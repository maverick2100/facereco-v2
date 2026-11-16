"""Real-time person detection + counting for cameras with `tasks: [person]`.

  FrameScheduler (person) --(batch)--> [person-detector thread]
      YOLOv7 (ANE) -> person filter / ROI -> tracker -> PersonCounter
      every FR_PERSON_FLUSH_S: one row per camera appended to `person_counts`
      cameras with a `count_zone`: IN / OUT crossings -> `person_crossings` (batched every ~2 s)

Runs alongside the face pipeline; both share the Neural Engine.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Optional

from config.settings import CameraConfig, ModelSettings, PersonSettings
from constants import MODEL_CACHE_DIR
from services.live_view import LiveView
from database.person_count_repository import PersonCountRepository
from inference.coreml_loader import CoreMLModel
from monitoring.performance import PerformanceTracker
from person_detection.person_counter import CountRecord, PersonCounter
from person_detection.yolo_engine import YOLOv7Engine, filter_persons
from scheduler.batch_scheduler import BatchScheduler
from scheduler.frame_scheduler import FrameScheduler

log = logging.getLogger(__name__)

CROSSING_WRITE_S = 2.0   # crossings reach the DB within ~2 s, batched into one INSERT


def load_yolo(models: ModelSettings, person: PersonSettings) -> YOLOv7Engine:
    cache = MODEL_CACHE_DIR if models.cache_compiled else None
    model = CoreMLModel(models.yolo_path, models.compute_units, cache, name="yolov7")
    return YOLOv7Engine(model, conf_threshold=person.conf_threshold, nms_threshold=person.nms_threshold,
                        class_id=person.class_id, num_classes=person.num_classes, input_std=person.input_std)


class PersonCountService:
    def __init__(self, engine: YOLOv7Engine, frames: FrameScheduler, repo: PersonCountRepository,
                 cameras: List[CameraConfig], settings: PersonSettings,
                 perf: Optional[PerformanceTracker] = None, retention_days: int = 0,
                 live: Optional[LiveView] = None) -> None:
        self.engine = engine
        self.frames = frames
        self.repo = repo
        self.cameras: Dict[str, CameraConfig] = {c.camera_id: c for c in cameras}
        self.settings = settings
        self.perf = perf or PerformanceTracker()
        self.retention_days = retention_days
        self.live = live
        self._last_retention = 0.0
        self.counter = PersonCounter(self.cameras, settings.flush_interval_s, settings.track_iou,
                                     settings.track_max_age_s, settings.track_min_hits,
                                     zones={c.camera_id: c.count_zone for c in cameras if c.count_zone},
                                     zone_anchor=settings.zone_anchor, zone_min_frames=settings.zone_min_frames)
        self._last_crossing_write = 0.0
        self._pushed: Dict[str, tuple] = {}   # camera_id -> (in, out) last sent to the update SP
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ lifecycle
    def warmup(self, iterations: int = 3) -> None:
        import numpy as np

        w, h = self.engine.input_size
        t0 = time.perf_counter()
        for _ in range(max(0, iterations)):
            self.engine.detect_batch([np.zeros((h, w, 3), np.uint8)] * self.settings.batch_size)
        if iterations:
            log.info("YOLOv7 warm-up done in %.2fs", time.perf_counter() - t0)

    def load_initial_totals(self) -> None:
        """Step 1: get each device's total so far (FR_PERSON_INIT_SQL) and count on from there."""
        sql = (self.settings.init_sql or "").strip()
        if not sql:
            return
        try:
            totals = self.repo.initial_totals(sql)
        except Exception as exc:
            log.warning("Starting totals query failed (%s) - counting starts at 0: %s", exc, sql)
            return
        seeded = self.counter.seed_totals(totals)
        self._pushed.update(self.counter.in_out_totals())   # already stored: don't send them back unchanged
        missing = sorted(set(self.cameras) - set(seeded))
        for cid in seeded:
            log.info("Camera %s: starting totals %s", cid, totals[cid])
        if missing:
            log.warning("No starting totals for camera(s) %s (SP returned ids: %s) - those start at 0",
                        ", ".join(missing), ", ".join(sorted(totals)) or "none")

    def start(self) -> None:
        self.load_initial_totals()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="person-detector", daemon=True)
        self._thread.start()
        log.info("Person counting started on %d cameras (batch %d, DB row every %.0fs)",
                 len(self.cameras), self.settings.batch_size, self.settings.flush_interval_s)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None
        self._write_crossings(force=True)
        self._write(self.counter.flush())  # keep the partial last interval

    # ------------------------------------------------------------------ loop
    def _loop(self) -> None:
        batcher = BatchScheduler(self.frames.get_frames, self.settings.batch_size, self.settings.max_wait_ms)
        while not self._stop.is_set():
            self._write_crossings()
            if self.counter.due():
                self._write(self.counter.flush())
                self._apply_retention()
            frames = batcher.next_batch(self._stop)
            if not frames:
                continue
            t0 = time.perf_counter()
            try:
                results = self.engine.detect_batch([f.image for f in frames])
            except Exception:
                log.exception("Person detection failed")
                time.sleep(0.5)
                continue
            dt = time.perf_counter() - t0
            self.perf.record_latency("yolo_batch_ms", dt * 1000.0)
            self.perf.record_latency("yolo_per_frame_ms", dt * 1000.0 / len(frames))
            self.perf.add_busy("inference", dt)
            self.perf.add_busy("yolo", dt)
            self.perf.incr("person_frames_processed", len(frames))
            for frame, dets in zip(frames, results):
                cam = self.cameras.get(frame.camera_id)
                dets = filter_persons(dets, self.settings.min_box_height, cam.roi if cam else None)
                self.perf.incr("persons_detected", int(dets.shape[0]))
                new = self.counter.update(frame.camera_id, dets, frame.timestamp, frame.mono_ts)
                if new:
                    self.perf.incr("persons_new", new)
                if self.live is not None:
                    self.live.publish_persons(frame.camera_id, frame.image, frame.timestamp, dets)

    def _write_crossings(self, force: bool = False) -> None:
        """One batched INSERT for every crossing since the last write (at most every CROSSING_WRITE_S)."""
        now = time.monotonic()
        if not force and now - self._last_crossing_write < CROSSING_WRITE_S:
            return
        self._last_crossing_write = now
        rows = self.counter.take_crossings()
        self._push_totals()
        if not rows:
            return
        try:
            self.repo.insert_crossings(c.as_row() for c in rows)
            self.perf.incr("person_crossings", len(rows))
        except Exception:
            log.exception("Failed to write %d zone crossings", len(rows))

    def _push_totals(self) -> None:
        """Step 2: send each camera's changed IN / OUT totals to the update SP (FR_PERSON_UPDATE_SQL)."""
        sql = (self.settings.update_sql or "").strip()
        if not sql:
            return
        changed = {cid: v for cid, v in self.counter.in_out_totals().items() if self._pushed.get(cid) != v}
        if not changed:
            return
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        for cid, (n_in, n_out) in changed.items():
            device = int(cid) if cid.isdigit() else cid
            try:
                self.repo.push_totals(sql, [(device, n_in, n_out, stamp)])
                self._pushed[cid] = (n_in, n_out)
            except Exception:
                log.exception("Count update failed for camera %s (in=%d out=%d) - retrying", cid, n_in, n_out)

    def _write(self, records: List[CountRecord]) -> None:
        if not records:
            return
        try:
            self.repo.insert_many(r.as_row() for r in records)
            self.perf.incr("person_count_rows", len(records))
            log.debug("Wrote %d person-count rows", len(records))
        except Exception:
            log.exception("Failed to write %d person-count rows", len(records))

    def _apply_retention(self) -> None:
        now = time.monotonic()
        if self.retention_days <= 0 or now - self._last_retention < 3600:
            return
        self._last_retention = now
        try:
            n = self.repo.delete_older_than(time.time() - self.retention_days * 86400)
            if n:
                log.info("Retention: deleted %d person-count rows older than %d days", n, self.retention_days)
        except Exception:
            log.exception("Person-count retention cleanup failed")

    def stats(self) -> dict:
        return self.counter.stats()
