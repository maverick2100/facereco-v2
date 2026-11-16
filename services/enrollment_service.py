"""Enrol people from photos.

Directory layout (one folder per person, folder name = person name):
    data/enrollment/
        Alice Smith/  img1.jpg img2.jpg ...
        Bob/          bob.png
Use 3-10 clear, frontal-ish photos per person, ideally from the real cameras.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

from constants import ENROLLMENT_DIR, IMAGE_EXTENSIONS
from database.embedding_repository import EmbeddingRepository
from database.person_repository import PersonRepository
from inference.inference_manager import InferenceManager
from processing.face_alignment import norm_crop
from processing.face_detection import faces_from_detections
from processing.face_quality import QualityConfig, assess_quality
from utils.image_utils import load_image, resize_max

log = logging.getLogger(__name__)

# Change this one value when the offender service assigns real login IDs.
OFFENDER_LOGIN_ID = 1


@dataclass
class EnrollResult:
    name: str
    person_id: Optional[int] = None
    added: int = 0
    skipped: int = 0
    errors: List[str] = field(default_factory=list)


class EnrollmentService:
    def __init__(self, inference: InferenceManager, persons: PersonRepository, embeddings: EmbeddingRepository,
                 min_face_size: int = 60, blur_threshold: float = 20.0) -> None:
        self.inference = inference
        self.persons = persons
        self.embeddings = embeddings
        self.quality_cfg = QualityConfig(min_face_size=min_face_size, blur_threshold=blur_threshold,
                                         max_yaw_ratio=0.6, min_quality=0.1)

    def _best_face_crop(self, img: np.ndarray):
        dets, kps = self.inference.detect([img])[0]
        faces = faces_from_detections(dets, kps)
        if not faces:
            return None, "no face"
        if len(faces) > 1:
            faces.sort(key=lambda f: f.area, reverse=True)
            if faces[1].area > 0.5 * faces[0].area:
                return None, "multiple faces of similar size"
        face = faces[0]
        crop = norm_crop(img, face.kps)
        q = assess_quality(face, crop, self.quality_cfg)
        if not q.passed:
            return None, f"low quality ({q.reason})"
        return (crop, q.score), ""

    def enroll_images(self, name: str, images: Iterable[Path], external_id: Optional[str] = None,
                      replace: bool = False) -> EnrollResult:
        res = EnrollResult(name=name)
        pid = self.persons.get_or_create(name, external_id)
        res.person_id = pid
        if replace:
            self.embeddings.delete_for_person(pid)

        crops, meta = [], []
        for path in images:
            path = Path(path)
            src = str(path.resolve())
            if not replace and self.embeddings.has_source(pid, src):
                res.skipped += 1
                continue
            img = load_image(path)
            if img is None:
                res.errors.append(f"{path.name}: unreadable")
                continue
            img = resize_max(img, 1920)
            found, err = self._best_face_crop(img)
            if found is None:
                res.errors.append(f"{path.name}: {err}")
                continue
            crops.append(found[0])
            meta.append((found[1], src))

        if crops:
            emb = self.inference.embed(crops)
            res.added = self.embeddings.add_many((pid, emb[i], meta[i][0], meta[i][1]) for i in range(len(crops)))
            if res.added and getattr(self.embeddings.db, "backend", "sqlite") == "postgres":
                self._sync_offender(pid, name, emb[0], meta[0][1], res, replace=replace)
        log.info("Enrolled %s (id=%s): +%d embeddings, %d skipped, %d errors",
                 name, pid, res.added, res.skipped, len(res.errors))
        for e in res.errors:
            log.warning("  %s: %s", name, e)
        return res

    def _sync_offender(self, person_id: int, name: str, embedding: np.ndarray, imagepath: str,
                       result: EnrollResult, replace: bool = False) -> None:
        """Store each enrolled image's embedding as an offender row in PostgreSQL."""
        db = self.embeddings.db
        try:
            mapping = db.fetchone(
                "SELECT offender_id FROM public.offender_embedding_sync WHERE person_id = ? AND imagepath = ?",
                (person_id, imagepath),
            )
            if mapping is None:
                # Migrate/link rows created by the earlier one-vector-per-person implementation.
                mapping = db.fetchone(
                    "SELECT offender_id FROM public.tbl_offenders WHERE offender_name = ? AND imagepath = ? "
                    "AND loginid = ? ORDER BY offender_id LIMIT 1",
                    (name, imagepath, OFFENDER_LOGIN_ID),
                )
            if mapping is not None:
                offender_id = int(mapping["offender_id"])
                if replace:
                    db.execute(
                        "UPDATE public.tbl_offenders SET embedding = ?, is_active = TRUE, updated_by = ?, "
                        "updated_date = CURRENT_TIMESTAMP WHERE offender_id = ?",
                        (np.asarray(embedding, dtype=np.float64).reshape(-1).tolist(),
                         OFFENDER_LOGIN_ID, offender_id),
                    )
                db.execute(
                    "INSERT INTO public.offender_embedding_sync(person_id, offender_id, imagepath, synced_at) "
                    "VALUES (?, ?, ?, ?) ON CONFLICT (person_id, imagepath) DO NOTHING",
                    (person_id, offender_id, imagepath, time.time()),
                )
                return

            row = db.fetchone(
                "SELECT * FROM public.fn_add_offenders(?, ?, ?, ?)",
                (name, np.asarray(embedding, dtype=np.float64).reshape(-1).tolist(),
                 OFFENDER_LOGIN_ID, imagepath),
            )
            if row is None or row.get("new_offender_id") is None:
                raise RuntimeError("fn_add_offenders returned no offender ID")
            offender_id = int(row["new_offender_id"])
            db.execute(
                "INSERT INTO public.offender_embedding_sync(person_id, offender_id, imagepath, synced_at) "
                "VALUES (?, ?, ?, ?) ON CONFLICT (person_id, imagepath) DO NOTHING",
                (person_id, offender_id, imagepath, time.time()),
            )
            # Retain a primary mapping for older consumers; one person can now have many vectors.
            db.execute(
                "INSERT INTO public.offender_sync(person_id, offender_id, synced_at) VALUES (?, ?, ?) "
                "ON CONFLICT (person_id) DO NOTHING",
                (person_id, offender_id, time.time()),
            )
            log.info("Synced offender embedding %s (person_id=%d, offender_id=%d, loginid=%d)",
                     name, person_id, offender_id, OFFENDER_LOGIN_ID)
        except Exception as exc:
            result.errors.append(f"offender sync: {exc}")
            log.exception("Could not sync offender %s (person_id=%d) to PostgreSQL", name, person_id)

    def enroll_directory(self, root: Path = ENROLLMENT_DIR, replace: bool = False) -> List[EnrollResult]:
        root = Path(root)
        results = []
        if not root.exists():
            log.warning("Enrolment directory %s does not exist", root)
            return results
        for person_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
            images = sorted(p for p in person_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
            if images:
                results.append(self.enroll_images(person_dir.name, images, replace=replace))
        return results

    def delete_person(self, person_id: int) -> bool:
        return self.persons.delete(person_id)
