"""Per-camera count aggregation.

Each analysed frame updates the camera's current interval; every
`interval_s` the interval is closed into one `CountRecord` (one DB row).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from person_detection.person_tracker import PersonTracker
from person_detection.zone_counter import Crossing, ZoneCounter


@dataclass
class CountRecord:
    camera_id: str
    period_start: float      # epoch seconds
    period_end: float
    frames: int
    current_count: int
    min_count: int
    max_count: int
    avg_count: float
    new_persons: int
    in_count: int = 0        # zone crossings outside -> inside during the interval
    out_count: int = 0       # zone crossings inside -> outside
    zone_count: int = 0      # people inside the zone at the end of the interval

    def as_row(self) -> Tuple:
        return (self.camera_id, self.period_start, self.period_end, self.frames, self.current_count,
                self.min_count, self.max_count, round(self.avg_count, 3), self.new_persons,
                self.in_count, self.out_count, self.zone_count, self.period_end)


class _Interval:
    def __init__(self, start: float) -> None:
        self.start = start
        self.end = start
        self.frames = 0
        self.total = 0
        self.min = 0
        self.max = 0
        self.last = 0
        self.new = 0
        self.ins = 0
        self.outs = 0
        self.zone = 0

    def add(self, count: int, new: int, ts: float) -> None:
        self.min = count if self.frames == 0 else min(self.min, count)
        self.max = max(self.max, count)
        self.frames += 1
        self.total += count
        self.last = count
        self.new += new
        self.end = max(self.end, ts)


class CameraCounter:
    def __init__(self, camera_id: str, tracker: PersonTracker, zone: Optional[ZoneCounter] = None) -> None:
        self.camera_id = camera_id
        self.tracker = tracker
        self.zone = zone
        self.crossings: List[Crossing] = []   # confirmed but not yet written to the DB
        self.current = 0
        self.total_persons = 0     # since start-up
        self.peak = 0
        self.last_update: Optional[float] = None
        self._interval: Optional[_Interval] = None

    def update(self, boxes: np.ndarray, ts: float, mono_ts: float) -> int:
        new = self.tracker.update(boxes, mono_ts)
        count = int(boxes.shape[0])
        if self._interval is None:
            self._interval = _Interval(ts)
        self._interval.add(count, new, ts)
        if self.zone is not None:
            # Tracks matched or created on this frame carry this frame's timestamp.
            seen = [t for t in self.tracker.tracks if t.last_seen == mono_ts]
            boxes_seen = np.stack([t.bbox for t in seen]) if seen else np.zeros((0, 4), np.float32)
            crossed = self.zone.update([t.track_id for t in seen], boxes_seen, ts,
                                       alive_ids=[t.track_id for t in self.tracker.tracks])
            for c in crossed:
                if c.direction == "in":
                    self._interval.ins += 1
                else:
                    self._interval.outs += 1
            self._interval.zone = self.zone.inside_now
            self.crossings.extend(crossed)
        self.current = count
        self.total_persons += new
        self.peak = max(self.peak, count)
        self.last_update = ts
        return new

    def close(self, now: float) -> Optional[CountRecord]:
        iv, self._interval = self._interval, None
        if iv is None or iv.frames == 0:
            return None
        return CountRecord(self.camera_id, iv.start, max(iv.end, now), iv.frames, iv.last, iv.min, iv.max,
                           iv.total / iv.frames, iv.new, iv.ins, iv.outs, iv.zone)

    def take_crossings(self) -> List[Crossing]:
        out, self.crossings = self.crossings, []
        return out

    def stats(self) -> dict:
        d = {"current": self.current, "total_persons": self.total_persons, "peak": self.peak,
             "tracks": self.tracker.active, "last_update": self.last_update}
        if self.zone is not None:
            d.update({"in": self.zone.total_in, "out": self.zone.total_out, "in_zone": self.zone.inside_now})
        return d


class PersonCounter:
    """All cameras' counters plus the flush clock."""

    def __init__(self, camera_ids, interval_s: float = 60.0, track_iou: float = 0.3,
                 track_max_age_s: float = 2.0, track_min_hits: int = 2,
                 zones: Optional[Dict[str, Sequence[Sequence[float]]]] = None, zone_anchor: str = "bottom",
                 zone_min_frames: int = 2) -> None:
        """zones: camera_id -> polygon [[x, y], ...] in frame pixels (cameras without one count no IN/OUT)."""
        self.interval_s = interval_s
        self._lock = threading.Lock()
        zones = zones or {}
        self.cameras: Dict[str, CameraCounter] = {
            cid: CameraCounter(cid, PersonTracker(track_iou, track_max_age_s, track_min_hits),
                               ZoneCounter(cid, zones[cid], zone_anchor, zone_min_frames) if zones.get(cid) else None)
            for cid in camera_ids
        }
        self._last_flush = time.monotonic()

    def update(self, camera_id: str, boxes: np.ndarray, ts: float, mono_ts: float) -> int:
        with self._lock:
            return self.cameras[camera_id].update(boxes, ts, mono_ts)

    def seed_totals(self, totals: Dict[str, Dict[str, int]]) -> List[str]:
        """Continue from stored totals: {camera_id: {"total", "in", "out"}}. Returns the cameras seeded."""
        seeded = []
        with self._lock:
            for cid, vals in totals.items():
                cam = self.cameras.get(cid)
                if cam is None:
                    continue
                cam.total_persons = vals.get("total", cam.total_persons)
                if cam.zone is not None:
                    cam.zone.total_in = vals.get("in", cam.zone.total_in)
                    cam.zone.total_out = vals.get("out", cam.zone.total_out)
                seeded.append(cid)
        return seeded

    def in_out_totals(self) -> Dict[str, Tuple[int, int]]:
        """Running IN / OUT totals of the cameras that have a count_zone."""
        with self._lock:
            return {cid: (c.zone.total_in, c.zone.total_out) for cid, c in self.cameras.items() if c.zone is not None}

    def due(self) -> bool:
        return time.monotonic() - self._last_flush >= self.interval_s

    def flush(self) -> List[CountRecord]:
        now = time.time()
        with self._lock:
            self._last_flush = time.monotonic()
            return [r for r in (c.close(now) for c in self.cameras.values()) if r is not None]

    def take_crossings(self) -> List[Crossing]:
        """Crossings confirmed since the last call (written to the DB every few seconds)."""
        with self._lock:
            return [x for c in self.cameras.values() for x in c.take_crossings()]

    def stats(self) -> dict:
        with self._lock:
            per = {cid: c.stats() for cid, c in self.cameras.items()}
        return {
            "current_total": sum(p["current"] for p in per.values()),
            "persons_since_start": sum(p["total_persons"] for p in per.values()),
            "in_total": sum(p.get("in", 0) for p in per.values()),
            "out_total": sum(p.get("out", 0) for p in per.values()),
            "per_camera": per,
        }
