"""Real-time person detection + counting for cameras with `tasks: [person]`.

  FrameScheduler (person) --(batch)--> [person-detector thread]
      YOLOv7 (ANE) -> person filter / ROI -> tracker -> PersonCounter
      every FR_PERSON_FLUSH_S: one row per camera appended to `person_counts`
      cameras with a roi line [x1, y1, x2, y2]: IN / OUT crossings -> `person_crossings` (batched every ~2 s)
      ...and the day's IN / OUT totals -> tbl_person_counts (one 'In' + one 'Out' row per device per
      business_date, updated in place within ~2 s of a change; FR_PERSON_COUNT_TABLE)

Runs alongside the face pipeline; both share the Neural Engine.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Set

from config.settings import CameraConfig, ModelSettings, PersonSettings
from constants import MODEL_CACHE_DIR
from services.live_view import LiveView
from database.person_count_repository import (FLAG_IN, FLAG_OUT, DeviceCountTable, PersonCountRepository,
                                               device_key)
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
                 live: Optional[LiveView] = None, table: Optional[DeviceCountTable] = None) -> None:
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
                                     lines={c.camera_id: c.roi for c in cameras if c.roi},
                                     anchor=settings.line_anchor, min_frames=settings.line_min_frames,
                                     min_distance=settings.line_min_distance,
                                     in_directions={c.camera_id: c.in_direction or settings.in_direction
                                                    for c in cameras})
        self._last_crossing_write = 0.0
        self._pushed: Dict[str, tuple] = {}   # camera_id -> (in, out) last sent to the update SP / table
        name = (settings.count_table or "").strip()
        if table is None and name.lower() not in ("", "off", "none", "-") and getattr(repo, "db", None) is not None:
            table = DeviceCountTable(repo.db, name)
        self.table = table                    # set -> write tbl_person_counts directly (instead of the SPs)
        self._day = date.today()              # business_date of the running totals
        self._has_rows: Set[str] = set()      # cameras that already have today's In / Out rows
        self._gates: Dict[str, Optional[int]] = {}
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
        if self.table is not None:
            self._load_day_totals()
            return
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

    def _load_day_totals(self) -> None:
        """Continue from today's In / Out rows in tbl_person_counts (cameras without rows start at 0)."""
        self._day = date.today()
        try:
            totals = self.table.day_totals(self._day)
        except Exception as exc:
            log.warning("Reading today's totals from %s failed (%s) - counting starts at 0", self.table.table, exc)
            return
        seeded = self.counter.seed_totals(totals)
        self._has_rows = set(seeded)
        self._pushed.update(self.counter.in_out_totals())
        for cid in seeded:
            log.info("Camera %s: today's totals so far %s (%s)", cid, totals[cid], self.table.table)

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
                dets = filter_persons(dets, self.settings.min_box_height)   # roi is the count line, not a crop
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
        """Step 2: send each camera's changed IN / OUT totals to tbl_person_counts or the update SP."""
        if self.table is not None:
            self._push_table()
            return
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

    def _push_table(self) -> None:
        today = date.today()
        if today != self._day:              # new business_date: totals restart at 0
            log.info("New business date %s - IN / OUT totals reset", today)
            self.counter.seed_totals({cid: {"in": 0, "out": 0} for cid in self.cameras})
            self._pushed = dict(self.counter.in_out_totals())
            self._has_rows.clear()
            self._day = today
        totals = self.counter.in_out_totals()
        changed = {cid: v for cid, v in totals.items() if self._pushed.get(cid, (0, 0)) != v}
        if not changed:
            return
        now = datetime.now().replace(microsecond=0)
        for cid, (n_in, n_out) in changed.items():
            old_in, old_out = self._pushed.get(cid, (0, 0))
            device = device_key(cid)
            try:
                gate = self._gate(cid, device)
                for flag, new, old in ((FLAG_IN, n_in, old_in), (FLAG_OUT, n_out, old_out)):
                    if new != old or cid not in self._has_rows:   # first write of the day: both rows
                        self.table.upsert(device, gate, flag, new, max(0, new - old), now, today)
                self._pushed[cid] = (n_in, n_out)
                self._has_rows.add(cid)
            except Exception:
                log.exception("Writing %s failed for camera %s (in=%d out=%d) - retrying",
                              self.table.table, cid, n_in, n_out)

    def _gate(self, cid: str, device: Any) -> Optional[int]:
        """gate_id from the camera config, else the gate this device was last stored with."""
        cam = self.cameras.get(cid)
        if cam is not None and cam.gate_id not in (None, ""):
            return int(cam.gate_id) if str(cam.gate_id).isdigit() else cam.gate_id
        if cid not in self._gates:
            self._gates[cid] = self.table.last_gate(device)
            if self._gates[cid] is None:
                log.warning("Camera %s: no gate_id (not in the camera config, none in %s) - gate_id left empty",
                            cid, self.table.table)
        return self._gates[cid]

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
