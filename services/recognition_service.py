"""The real-time pipeline.

  FrameScheduler --(batch of frames)--> [detector thread]
        SCRFD (ANE) -> filter -> track-cache skip -> align -> quality gate
        --> InferenceQueue of face crops --> [recognizer thread]
        ArcFace (ANE, batched) -> best match in tbl_offenders (PostgreSQL) -> de-dup
        -> one row in tbl_offender_search_logs (+ frame / face images)

Only two Python threads touch the models; everything else is queues.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from config.settings import CameraConfig, DetectionSettings, RecognitionSettings
from constants import UNKNOWN_PERSON_ID
from database.offender_repository import OffenderRepository
from database.person_count_repository import device_key
from services.live_view import PENDING, REJECTED, UNKNOWN, FaceBox, LiveView
from inference.inference_manager import InferenceManager
from monitoring.performance import PerformanceTracker
from processing.face_alignment import norm_crop
from processing.face_detection import DetectedFace, faces_from_detections, filter_faces
from processing.face_quality import QualityConfig, assess_quality
from recognition.duplicate_filter import DuplicateFilter, TrackCache
from scheduler.batch_scheduler import BatchScheduler
from scheduler.camera_queue import Frame
from scheduler.frame_scheduler import FrameScheduler
from scheduler.inference_queue import InferenceQueue
from services import image_store
from utils.image_utils import crop_with_margin

log = logging.getLogger(__name__)


@dataclass
class FaceTask:
    camera_id: str
    frame_id: int
    timestamp: float
    face: DetectedFace
    crop: np.ndarray            # aligned 112x112 BGR
    quality: float
    snapshot: Optional[np.ndarray] = None   # face with margin  -> detected_face
    frame: Optional[np.ndarray] = None      # whole frame       -> detected_imagepath
    mono_ts: float = 0.0                    # arrival time, for time_taken


class RecognitionService:
    def __init__(self, inference: InferenceManager, frames: FrameScheduler, offenders: OffenderRepository,
                 duplicates: DuplicateFilter, tracks: TrackCache, cameras: List[CameraConfig], det_cfg: DetectionSettings, rec_cfg: RecognitionSettings,
                 perf: Optional[PerformanceTracker] = None, save_snapshots: bool = False,
                 live: Optional[LiveView] = None) -> None:
        self.inference = inference
        self.frames = frames
        self.duplicates = duplicates
        self.tracks = tracks
        self.offenders = offenders
        self.cameras: Dict[str, CameraConfig] = {c.camera_id: c for c in cameras}
        self.det_cfg = det_cfg
        self.rec_cfg = rec_cfg
        self.perf = perf or PerformanceTracker()
        self.save_snapshots = save_snapshots
        self.live = live
        self.quality_cfg = QualityConfig(
            min_face_size=det_cfg.min_face_size,
            blur_threshold=rec_cfg.blur_threshold,
            max_yaw_ratio=rec_cfg.max_yaw_ratio,
            min_quality=rec_cfg.min_quality,
        )
        self.rec_queue: InferenceQueue[FaceTask] = InferenceQueue(rec_cfg.queue_size)
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self._stop.clear()
        for target, name in ((self._detection_loop, "detector"), (self._recognition_loop, "recognizer")):
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)
        log.info("Recognition pipeline started (det batch %d, rec batch %d)",
                 self.det_cfg.batch_size, self.rec_cfg.batch_size)

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=10)
        self._threads.clear()

    # ------------------------------------------------------------------ detection
    def _detection_loop(self) -> None:
        batcher = BatchScheduler(self.frames.get_frames, self.det_cfg.batch_size, self.det_cfg.max_wait_ms)
        while not self._stop.is_set():
            frames = batcher.next_batch(self._stop)
            if not frames:
                continue
            try:
                results = self.inference.detect([f.image for f in frames])
            except Exception:
                log.exception("Detection failed")
                time.sleep(0.5)
                continue
            self.perf.incr("frames_processed", len(frames))
            now = time.monotonic()
            for frame, (dets, kps) in zip(frames, results):
                try:
                    self._handle_detections(frame, dets, kps, now)
                except Exception:
                    log.exception("[%s] post-detection failed", frame.camera_id)

    def _handle_detections(self, frame: Frame, dets: np.ndarray, kps: np.ndarray, now: float) -> None:
        live = self.live is not None
        boxes: List[FaceBox] = []
        if dets.shape[0] == 0:
            if live:
                self.live.publish_faces(frame.camera_id, frame.image, frame.timestamp, boxes)
            return
        faces = filter_faces(faces_from_detections(dets, kps), min_size=self.det_cfg.min_face_size,
                             max_faces=self.det_cfg.max_faces_per_frame)   # roi is the person count line
        self.perf.incr("faces_detected", len(faces))
        for face in faces:
            crop = norm_crop(frame.image, face.kps)
            q = assess_quality(face, crop, self.quality_cfg)
            if not q.passed:
                self.perf.incr(f"faces_rejected_{q.reason}")
                if live:
                    boxes.append(FaceBox(face.bbox_int(), q.reason, REJECTED))
                continue
            images = self.save_snapshots or image_store.base_url() != ""
            snap = crop_with_margin(frame.image, face.bbox, 0.4) if images else None
            task = FaceTask(frame.camera_id, frame.frame_id, frame.timestamp, face, crop, q.score, snap,
                            frame.image if images else None, frame.mono_ts)
            if not self.rec_queue.put(task):
                self.perf.incr("faces_dropped_backpressure")
            if live:
                boxes.append(FaceBox(face.bbox_int(), "", PENDING))
        if live:
            self.live.publish_faces(frame.camera_id, frame.image, frame.timestamp, boxes)

    # ------------------------------------------------------------------ recognition
    def _recognition_loop(self) -> None:
        batcher = BatchScheduler(self.rec_queue.get_batch, self.rec_cfg.batch_size, self.rec_cfg.max_wait_ms)
        while not self._stop.is_set():
            tasks = batcher.next_batch(self._stop)
            if not tasks:
                continue
            try:
                emb = self.inference.embed([t.crop for t in tasks])
            except Exception:
                log.exception("Embedding generation failed")
                time.sleep(0.5)
                continue
            self.perf.incr("faces_embedded", len(tasks))
            now = time.monotonic()
            threshold = self.rec_cfg.match_threshold
            for i, task in enumerate(tasks):
                try:
                    best = self.offenders.search(emb[i])
                except Exception:
                    log.exception("Offender search failed for camera %s", task.camera_id)
                    continue

                similarity = float(best["similarity"]) if best else 0.0
                is_known = best is not None and similarity >= threshold
                offender_id = int(best["offender_id"]) if is_known else None
                pid = offender_id if offender_id is not None else UNKNOWN_PERSON_ID
                label = str(best["offender_name"]) if is_known else UNKNOWN
                self.tracks.update(task.camera_id, task.face.bbox, pid, similarity, now)
                if not is_known and not self.rec_cfg.record_unknown:
                    continue
                if not self.duplicates.should_emit(task.camera_id, pid, emb[i], now):
                    self.perf.incr("events_suppressed")
                    continue
                self.perf.incr("recognized" if is_known else "unknown")
                try:
                    log_id = self._log_search(task, emb[i], offender_id, similarity)
                except Exception:
                    log.exception("Could not write tbl_offender_search_logs row for camera %s", task.camera_id)
                    log_id = None
                log.info("Offender search camera=%s %s name=%s score=%.2f%% log_id=%s",
                         task.camera_id, "matched" if is_known else "not_matched", label,
                         similarity * 100.0, log_id)

    def _log_search(self, task: FaceTask, embedding: np.ndarray, offender_id: Optional[int],
                    similarity: float) -> Optional[int]:
        """One row in tbl_offender_search_logs; offender_id is NULL for a face that matched nobody."""
        ts = image_store.unique_ms()
        frame_url = image_store.store(image_store.DETECTED, ts, "frame", task.frame)
        face_url = image_store.store(image_store.DETECTED, ts, "face", task.snapshot)
        cam = self.cameras.get(task.camera_id)
        time_taken_ms = (time.monotonic() - task.mono_ts) * 1000.0 if task.mono_ts else 0.0
        return self.offenders.log_search(
            device_id=device_key(task.camera_id),
            gate_id=getattr(cam, "gate_id", None),
            offender_id=offender_id,
            embedding=embedding,
            frameno=task.frame_id,
            detected_imagepath=frame_url,
            detected_face=face_url,
            match_percentage=similarity * 100.0,
            time_taken_ms=time_taken_ms,
            capture_time=datetime.fromtimestamp(task.timestamp),
        )

    def stats(self) -> dict:
        return {"rec_queue": self.rec_queue.stats(), "track_cache": self.tracks.stats()}
