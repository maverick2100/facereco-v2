"""IN / OUT counting across a line per camera: roi = [x1, y1, x2, y2] = the two ends of the line.

Each tracked person is reduced to an anchor point (by default the bottom-centre of the box, i.e.
the feet, which is what actually crosses a doorway on an angled camera). Every frame we check which
side of the line that point is on. When a person changes side, the crossing direction is:

    mostly horizontal line (|x2-x1| >= |y2-y1|):  top -> bottom = "down",  bottom -> top = "up"
    mostly vertical line:                          left -> right = "right", right -> left = "left"

`in_direction` (per camera, default "down") says which of these is an IN; the opposite one is an OUT.

Rules that keep the count clean:
  * a side change only counts once it has held for `min_frames` consecutive frames, so box jitter
    on the line doesn't produce IN/OUT/IN/OUT;
  * the point must also have moved at least `min_distance` px away from the line (a dead band);
  * the crossing must happen between the two ends of the line (people walking past the end of a
    short line aren't counted);
  * a person first seen already on one side sets their starting side but isn't counted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

IN = "in"
OUT = "out"
ANCHORS = ("bottom", "center")
DIRECTIONS = ("down", "up", "right", "left")
_OPPOSITE = {"down": "up", "up": "down", "right": "left", "left": "right"}
_SAME_AXIS = {"right": "down", "left": "up", "down": "right", "up": "left"}   # for a line of the other kind
END_MARGIN = 0.05        # crossings this far (fraction of the line length) past an end still count


def anchor_points(boxes: np.ndarray, anchor: str = "bottom") -> np.ndarray:
    boxes = np.asarray(boxes, np.float32)
    if boxes.size == 0:
        return np.zeros((0, 2), np.float32)
    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    cy = boxes[:, 3] if anchor == "bottom" else (boxes[:, 1] + boxes[:, 3]) / 2.0
    return np.stack([cx, cy], axis=1)


@dataclass
class Crossing:
    camera_id: str
    track_id: int
    direction: str           # "in" | "out"
    ts: float                # epoch seconds
    x: float                 # anchor point at the moment the crossing was confirmed
    y: float
    movement: str = ""       # "down" | "up" | "right" | "left"

    def as_row(self) -> Tuple:
        return (self.camera_id, self.track_id, self.direction, self.ts, round(self.x, 1), round(self.y, 1))


@dataclass
class _State:
    side: int                         # +1 / -1: last confirmed side
    pending: int = 0
    streak: int = 0
    last: Optional[Tuple[float, float]] = None   # last point on the confirmed side (to locate the crossing)


class LineCounter:
    def __init__(self, camera_id: str, line: Sequence[float], in_direction: str = "down",
                 anchor: str = "bottom", min_frames: int = 2, min_distance: float = 4.0) -> None:
        if anchor not in ANCHORS:
            raise ValueError(f"anchor must be one of {ANCHORS}, got {anchor!r}")
        in_direction = (in_direction or "down").strip().lower()
        if in_direction not in DIRECTIONS:
            raise ValueError(f"camera {camera_id}: in_direction must be one of {DIRECTIONS}, got {in_direction!r}")
        if line is None or len(line) != 4:
            raise ValueError(f"camera {camera_id}: count line must be [x1, y1, x2, y2]")
        x1, y1, x2, y2 = (float(v) for v in line)
        self.camera_id = camera_id
        self.a = np.array([x1, y1], np.float32)
        self.b = np.array([x2, y2], np.float32)
        d = self.b - self.a
        self.length = float(np.hypot(*d))
        if self.length < 1.0:
            raise ValueError(f"camera {camera_id}: the count line's two ends are the same point")
        self.horizontal = abs(d[0]) >= abs(d[1])
        # Unit normal pointing "down" (+y) for a horizontal-ish line, "right" (+x) for a vertical-ish one.
        n = np.array([-d[1], d[0]], np.float32) / self.length
        if (self.horizontal and n[1] < 0) or (not self.horizontal and n[0] < 0):
            n = -n
        self.normal = n
        self.positive, self.negative = ("down", "up") if self.horizontal else ("right", "left")
        if in_direction not in (self.positive, self.negative):     # e.g. "left" asked on a horizontal line
            in_direction = _SAME_AXIS[in_direction]
        self.in_direction = in_direction
        self.out_direction = _OPPOSITE[in_direction]
        self.anchor = anchor
        self.min_frames = max(1, int(min_frames))
        self.min_distance = max(0.0, float(min_distance))
        self._states: Dict[int, _State] = {}
        self.total_in = 0
        self.total_out = 0

    @property
    def inside_now(self) -> int:
        """Occupancy estimate: IN minus OUT (never below 0)."""
        return max(0, self.total_in - self.total_out)

    def _distance(self, pts: np.ndarray) -> np.ndarray:
        """Signed distance of each point from the line (> 0 = below / right of it)."""
        return (pts - self.a) @ self.normal

    def _between_ends(self, p: np.ndarray) -> bool:
        t = float((p - self.a) @ (self.b - self.a)) / (self.length ** 2)
        return -END_MARGIN <= t <= 1.0 + END_MARGIN

    def update(self, track_ids: Iterable[int], boxes: np.ndarray, ts: float,
               alive_ids: Optional[Iterable[int]] = None) -> List[Crossing]:
        """Feed the tracks matched in this frame. `alive_ids` = every track the tracker still keeps
        (state for the others is dropped). Returns the crossings confirmed on this frame."""
        ids = list(track_ids)
        pts = anchor_points(boxes, self.anchor) if ids else np.zeros((0, 2), np.float32)
        dist = self._distance(pts) if ids else np.zeros(0, np.float32)
        out: List[Crossing] = []
        for tid, p, dv in zip(ids, pts, dist):
            if abs(dv) < self.min_distance:           # on the line: wait until clearly on one side
                continue
            side = 1 if dv > 0 else -1
            st = self._states.get(tid)
            if st is None:
                self._states[tid] = _State(side, last=(float(p[0]), float(p[1])))
                continue
            if side == st.side:
                st.pending, st.streak, st.last = 0, 0, (float(p[0]), float(p[1]))
                continue
            if st.pending == side:
                st.streak += 1
            else:
                st.pending, st.streak = side, 1
            if st.streak < self.min_frames:
                continue
            prev = np.array(st.last if st.last is not None else p, np.float32)
            st.side, st.pending, st.streak, st.last = side, 0, 0, (float(p[0]), float(p[1]))
            if not self._between_ends((prev + p) / 2.0):  # walked around the end of the line
                continue
            movement = self.positive if side > 0 else self.negative
            if movement == self.in_direction:
                self.total_in += 1
                direction = IN
            else:
                self.total_out += 1
                direction = OUT
            out.append(Crossing(self.camera_id, int(tid), direction, ts, float(p[0]), float(p[1]), movement))
        if alive_ids is not None:
            alive = set(alive_ids)
            for tid in [t for t in self._states if t not in alive]:
                del self._states[tid]
        return out
