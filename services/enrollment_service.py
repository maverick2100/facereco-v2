"""Enrol people from photos.

Directory layout (one folder per person, folder name = person name):
    data/enrollment/
        Alice Smith/  img1.jpg img2.jpg ...
        Bob/          bob.png
Use 3-10 clear, frontal-ish photos per person, ideally from the real cameras.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

from constants import ENROLLMENT_DIR, IMAGE_EXTENSIONS
from database.connection import Database
from database.offender_repository import offender_table_sql
from inference.inference_manager import InferenceManager
from processing.face_alignment import norm_crop
from processing.face_detection import faces_from_detections
from processing.face_quality import QualityConfig, assess_quality
from services import image_store
from utils.image_utils import crop_with_margin, load_image, resize_max

log = logging.getLogger(__name__)

# Change this one value when the offender service assigns real login IDs.
OFFENDER_LOGIN_ID = 1

THUMB_MAX_SIDE = 256


@dataclass
class EnrollResult:
    name: str
    offender_ids: List[int] = field(default_factory=list)
    added: int = 0
    skipped: int = 0
    errors: List[str] = field(default_factory=list)


class EnrollmentService:
    """Detects/aligns/embeds each photo and stores it as a row in the offender table (tbl_offenders)."""

    def __init__(self, inference: InferenceManager, db: Database,
                 min_face_size: int = 60, blur_threshold: float = 20.0) -> None:
        self.inference = inference
        self.db = db
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
        thumb = resize_max(crop_with_margin(img, face.bbox, 0.4), THUMB_MAX_SIDE)
        return (crop, q.score, thumb), ""

    def enroll_images(self, name: str, images: Iterable[Path],
                      replace: bool = False) -> EnrollResult:
        """Add one tbl_offenders row per usable photo.

        A name that already has active rows is skipped; with replace=True those rows are set
        is_active = FALSE (kept, not deleted) and the photos are added as new rows.
        """
        res = EnrollResult(name=name)
        images = [Path(p) for p in images]
        if getattr(self.db, "backend", "sqlite") != "postgres":
            res.errors.append("enrolment writes to the offender table and requires PostgreSQL")
            log.error("Cannot enrol %s: %s", name, res.errors[-1])
            return res
        if not image_store.base_url():
            res.errors.append("FR_OFFENDER_IMAGE_BASE_URL is not set in .env (the image server IP, e.g. 192.168.1.50)")
            log.error("Cannot enrol %s: %s", name, res.errors[-1])
            return res

        existing = self._active_count(name)
        if existing and not replace:
            res.skipped = len(images)
            log.info("%s already has %d active offender rows - skipped (use --replace to re-enrol)", name, existing)
            return res

        found_faces = []
        for path in images:
            img = load_image(path)
            if img is None:
                res.errors.append(f"{path.name}: unreadable")
                continue
            img = resize_max(img, 1920)
            found, err = self._best_face_crop(img)
            if found is None:
                res.errors.append(f"{path.name}: {err}")
                continue
            found_faces.append((path, found))

        if found_faces:
            emb = self.inference.embed([f[1][0] for f in found_faces])
            if existing and replace:
                n = self._deactivate(name)
                log.info("%s: deactivated %d old offender rows", name, n)
            for vec, (path, (_, _, thumb)) in zip(emb, found_faces):
                try:
                    res.offender_ids.append(self._store_offender(name, vec, thumb))
                    res.added += 1
                except Exception as exc:
                    res.errors.append(f"{path.name}: offender insert failed: {exc}")
                    log.exception("Could not store offender %s (%s)", name, path)
        log.info("Enrolled %s: +%d offender rows, %d skipped, %d errors",
                 name, res.added, res.skipped, len(res.errors))
        for e in res.errors:
            log.warning("  %s: %s", name, e)
        return res

    def _active_count(self, name: str) -> int:
        row = self.db.fetchone(
            f"SELECT COUNT(*) AS n FROM {offender_table_sql()} WHERE offender_name = ? AND is_active = TRUE",
            (name,),
        )
        return int(row["n"]) if row else 0

    def _deactivate(self, name: str) -> int:
        return self.db.execute(
            f"UPDATE {offender_table_sql()} SET is_active = FALSE, updated_by = ?, updated_date = CURRENT_TIMESTAMP "
            "WHERE offender_name = ? AND is_active = TRUE",
            (OFFENDER_LOGIN_ID, name),
        ).rowcount

    def _store_offender(self, name: str, embedding: np.ndarray, thumb: Optional[np.ndarray] = None) -> int:
        """Insert one row into the offender table (tbl_offenders):

        offender_id (generated), offender_name, embedding (double precision[]), loginid,
        imagepath (<FR_OFFENDER_IMAGE_BASE_URL>/images/offender/<epoch ms>_thumb.jpg), is_active,
        created_by, updated_by, updated_date. The face thumbnail is saved under that file name in
        FR_OFFENDER_IMAGE_DIR when it is set.
        """
        imagepath = image_store.store(image_store.OFFENDER, image_store.unique_ms(), "thumb", thumb)
        if imagepath is None:
            raise RuntimeError("FR_OFFENDER_IMAGE_BASE_URL is not set in .env")
        vec = np.asarray(embedding, dtype=np.float64).reshape(-1).tolist()
        table = offender_table_sql()
        row = self.db.fetchone(
            f"INSERT INTO {table} (offender_name, embedding, loginid, imagepath, is_active, "
            "created_by, updated_by, updated_date) "
            "VALUES (?, ?::double precision[], ?, ?, TRUE, ?, ?, CURRENT_TIMESTAMP) RETURNING offender_id",
            (name, vec, OFFENDER_LOGIN_ID, imagepath, OFFENDER_LOGIN_ID, OFFENDER_LOGIN_ID),
        )
        if row is None or row.get("offender_id") is None:
            raise RuntimeError(f"insert into {table} returned no offender_id")
        log.info("Added offender %s (offender_id=%s) imagepath=%s", name, row["offender_id"], imagepath)
        return int(row["offender_id"])

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
