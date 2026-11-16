"""On-screen inference view: `python main.py run --show`.

The pipelines hand over each frame they ran inference on, with its detections. The main thread
draws boxes + names and shows every selected camera in a grid inside one OpenCV window
(cv2.imshow has to run on the main thread on macOS). Press q or Esc to stop.

When --show isn't used, nothing is created and the pipelines skip this entirely.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

KNOWN = "known"          # identified person
UNKNOWN = "unknown"      # face embedded but not in the gallery
PENDING = "pending"      # queued for ArcFace, no identity yet
REJECTED = "rejected"    # failed the quality gate (blurry, side-on, small)

# BGR
_FACE_COLORS = {KNOWN: (0, 200, 0), UNKNOWN: (0, 0, 230), PENDING: (200, 200, 200), REJECTED: (120, 120, 120)}
_PERSON_COLOR = (230, 160, 0)
_LINE_COLOR = (0, 220, 255)
_STALE_S = 1.5           # drop boxes that are older than this relative to the frame shown

WINDOW = "facereco - q / Esc to quit"


@dataclass
class FaceBox:
    bbox: Sequence[int]
    label: str
    kind: str
    similarity: float = 0.0


@dataclass
class _Slot:
    name: str
    image: Optional[np.ndarray] = None
    image_ts: float = 0.0
    faces: List[FaceBox] = field(default_factory=list)
    faces_ts: float = 0.0
    persons: Optional[np.ndarray] = None
    persons_ts: float = 0.0


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class LiveView:
    def __init__(self, cameras: Dict[str, str], window_width: int = 1600,
                 lines: Optional[Dict[str, Tuple[Sequence[float], str]]] = None) -> None:
        """cameras: camera_id -> display name (only these are shown).
        lines: camera_id -> ([x1, y1, x2, y2] count line in pixels, IN direction down|up|right|left)."""
        self.lines = {k: v for k, v in (lines or {}).items() if v and v[0]}
        self.window_width = window_width
        self._slots: Dict[str, _Slot] = {cid: _Slot(name) for cid, name in cameras.items()}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ called by the pipelines
    def publish_faces(self, camera_id: str, image: np.ndarray, ts: float, faces: List[FaceBox]) -> None:
        with self._lock:
            s = self._slots.get(camera_id)
            if s is None:
                return
            # A tracked face is re-checked every few seconds and is "pending" for that one frame:
            # keep the name it had on the previous frame so the label doesn't flicker.
            for f in faces:
                if f.kind == PENDING:
                    best = max(s.faces, key=lambda p: _iou(p.bbox, f.bbox), default=None)
                    if best is not None and best.kind in (KNOWN, UNKNOWN) and _iou(best.bbox, f.bbox) > 0.3:
                        f.label, f.kind, f.similarity = best.label, best.kind, best.similarity
            s.faces, s.faces_ts = faces, ts
            if ts >= s.image_ts:
                s.image, s.image_ts = image, ts

    def publish_persons(self, camera_id: str, image: np.ndarray, ts: float, boxes: np.ndarray) -> None:
        with self._lock:
            s = self._slots.get(camera_id)
            if s is None:
                return
            s.persons, s.persons_ts = np.asarray(boxes, dtype=np.float32), ts
            if ts >= s.image_ts:
                s.image, s.image_ts = image, ts

    # ------------------------------------------------------------------ drawing
    def _tile(self, s: _Slot, w: int, h: int, status: str, line=None) -> np.ndarray:
        if s.image is None:
            tile = np.zeros((h, w, 3), np.uint8)
            cv2.putText(tile, "waiting for frames...", (12, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (160, 160, 160), 1, cv2.LINE_AA)
        else:
            ih, iw = s.image.shape[:2]
            scale = min(w / iw, h / ih)
            img = cv2.resize(s.image, (int(iw * scale), int(ih * scale)), interpolation=cv2.INTER_AREA)
            tile = np.zeros((h, w, 3), np.uint8)
            oy, ox = (h - img.shape[0]) // 2, (w - img.shape[1]) // 2
            tile[oy:oy + img.shape[0], ox:ox + img.shape[1]] = img
            fs = max(0.4, min(0.7, w / 1100.0))

            def box(b, color, label: str = "") -> None:
                x1, y1, x2, y2 = (int(round(v * scale)) for v in b[:4])
                x1, x2, y1, y2 = x1 + ox, x2 + ox, y1 + oy, y2 + oy
                cv2.rectangle(tile, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
                if label:
                    (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
                    ty = y1 - 4 if y1 - th - base - 6 >= 0 else y2 + th + 6
                    cv2.rectangle(tile, (x1, ty - th - base - 2), (x1 + tw + 8, ty + 2), color, -1)
                    cv2.putText(tile, label, (x1 + 4, ty - base + 1), cv2.FONT_HERSHEY_SIMPLEX, fs,
                                (0, 0, 0) if color != _FACE_COLORS[UNKNOWN] else (255, 255, 255), 1, cv2.LINE_AA)

            if line is not None:
                _draw_line(tile, line[0], line[1], scale, ox, oy, fs)
            if s.persons is not None and abs(s.persons_ts - s.image_ts) <= _STALE_S:
                for d in s.persons:
                    box(d, _PERSON_COLOR, f"person {d[4]:.2f}" if len(d) > 4 else "person")
                status += f"  people: {len(s.persons)}"
            if abs(s.faces_ts - s.image_ts) <= _STALE_S:
                for f in s.faces:
                    text = {KNOWN: f"{f.label} {f.similarity:.2f}", UNKNOWN: "unknown", PENDING: "...",
                            REJECTED: ""}.get(f.kind, f.label)
                    box(f.bbox, _FACE_COLORS.get(f.kind, (200, 200, 200)), text)
        cv2.rectangle(tile, (0, 0), (w, 24), (0, 0, 0), -1)
        cv2.putText(tile, f"{s.name}  {status}", (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.rectangle(tile, (0, 0), (w - 1, h - 1), (60, 60, 60), 1)
        return tile

    def mosaic(self, status: Optional[Dict[str, str]] = None) -> np.ndarray:
        """All selected cameras in a grid, sized to `window_width`."""
        status = status or {}
        with self._lock:
            items = [(cid, _Slot(s.name, s.image, s.image_ts, list(s.faces), s.faces_ts, s.persons, s.persons_ts))
                     for cid, s in self._slots.items()]
        n = max(1, len(items))
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        tw = self.window_width // cols
        th = tw * 9 // 16
        canvas = np.zeros((rows * th, cols * tw, 3), np.uint8)
        for i, (cid, s) in enumerate(items):
            r, c = divmod(i, cols)
            canvas[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = self._tile(s, tw, th, status.get(cid, ""),
                                                                          self.lines.get(cid))
        return canvas

    # ------------------------------------------------------------------ main-thread loop
    def run_window(self, stop: threading.Event, status_fn=None, fps: float = 10.0) -> None:
        """Show the window until `stop` is set or q / Esc is pressed (then sets `stop`)."""
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        delay = max(1, int(1000 / fps))
        try:
            while not stop.is_set():
                t0 = time.monotonic()
                cv2.imshow(WINDOW, self.mosaic(status_fn() if status_fn else None))
                key = cv2.waitKey(max(1, delay - int((time.monotonic() - t0) * 1000))) & 0xFF
                if key in (ord("q"), 27):
                    stop.set()
                elif cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:  # window closed
                    stop.set()
        finally:
            cv2.destroyAllWindows()
            cv2.waitKey(1)


def _draw_line(tile: np.ndarray, line: Sequence[float], in_direction: str, scale: float, ox: int, oy: int,
               fs: float) -> None:
    """The count line plus an arrow from its middle pointing the IN way."""
    x1, y1, x2, y2 = (float(v) * scale for v in line)
    a, b = (int(round(x1 + ox)), int(round(y1 + oy))), (int(round(x2 + ox)), int(round(y2 + oy)))
    cv2.line(tile, a, b, _LINE_COLOR, 2, cv2.LINE_AA)
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / length, dx / length                       # normal; flip so it points down / right
    if (abs(dx) >= abs(dy) and ny < 0) or (abs(dx) < abs(dy) and nx < 0):
        nx, ny = -nx, -ny
    if in_direction in ("up", "left"):
        nx, ny = -nx, -ny
    mx, my = (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0
    k = max(18.0, min(40.0, length * 0.15))
    tip = (int(round(mx + nx * k)), int(round(my + ny * k)))
    cv2.arrowedLine(tile, (int(round(mx)), int(round(my))), tip, _LINE_COLOR, 2, cv2.LINE_AA, tipLength=0.35)
    cv2.putText(tile, "IN", (tip[0] + 4, tip[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, fs, _LINE_COLOR, 1, cv2.LINE_AA)
