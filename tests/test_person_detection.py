import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import CameraConfig  # noqa: E402
from database.connection import Database  # noqa: E402
from database.person_count_repository import PersonCountRepository  # noqa: E402
from schema_sql import create_tables  # noqa: E402
from inference.coreml_loader import FeatureSpec  # noqa: E402
from ingestion.frame_receiver import FrameReceiver  # noqa: E402
from person_detection.person_count_service import PersonCountService  # noqa: E402
from person_detection.person_counter import PersonCounter  # noqa: E402
from person_detection.person_tracker import PersonTracker  # noqa: E402
from person_detection.yolo_engine import YOLOv7Engine, filter_persons, letterbox_center  # noqa: E402
from config.settings import PersonSettings  # noqa: E402,I001
from scheduler.camera_queue import CameraQueue  # noqa: E402
from scheduler.frame_scheduler import FrameScheduler  # noqa: E402

NO = 85  # 5 + 80 classes
# Person box in 640x640 model space: centre (320, 320), 100x200. A 1280x720 frame is letterboxed
# with scale 0.5 and pad_y 140, so in frame pixels it is x 540..740, y 160..560.


def _expected_frame_box(cx=320, cy=320, w=100, h=200, scale=0.5, px=0, py=140):
    x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    return [(x1 - px) / scale, (y1 - py) / scale, (x2 - px) / scale, (y2 - py) / scale]


class FakeYOLO:
    def __init__(self, layout="decoded", size=640):
        self.inputs = [FeatureSpec("images", "multiarray", shape=(1, 3, size, size), dtype=np.float32)]
        self.outputs = []
        self.layout = layout
        self.size = size

    def _decoded(self, transposed=False):
        a = np.zeros((1, 300, NO), np.float32)
        a[0, 7, :4] = [320, 320, 100, 200]
        a[0, 7, 4] = 0.9
        a[0, 7, 5] = 0.95          # person
        a[0, 8, :4] = [100, 100, 50, 50]
        a[0, 8, 4] = 0.9
        a[0, 8, 5 + 2] = 0.95      # a car - must be ignored
        a[0, 9, :4] = [322, 321, 102, 198]  # duplicate person -> NMS
        a[0, 9, 4:6] = 0.8
        return {"output": a.transpose(0, 2, 1) if transposed else a}

    def _raw(self):
        def logit(p):
            return float(np.log(p / (1 - p)))
        outs = {}
        for s in (8, 16, 32):
            g = self.size // s
            h = np.full((1, 3, g, g, NO), -10.0, np.float32)
            if s == 16:
                # anchor 2 at stride 16 = (72, 146); target w=100, h=200 at centre (320, 320):
                # xy = (sig*2 - 0.5 + grid) * stride  -> sig = 0.25 at grid cell 20
                # wh = (sig*2)^2 * anchor              -> sig = sqrt(target / anchor) / 2
                g_xy, (aw, ah) = 320 // s, (72, 146)
                h[0, 2, g_xy, g_xy, 0] = logit(0.25)
                h[0, 2, g_xy, g_xy, 1] = logit(0.25)
                h[0, 2, g_xy, g_xy, 2] = logit(np.sqrt(100 / aw) / 2)
                h[0, 2, g_xy, g_xy, 3] = logit(np.sqrt(200 / ah) / 2)
                h[0, 2, g_xy, g_xy, 4] = logit(0.9)
                h[0, 2, g_xy, g_xy, 5] = logit(0.95)
            outs[f"head{s}"] = h
        return outs

    def _end2end(self):
        return {"dets": np.array([[0, 270, 220, 370, 420, 0, 0.9], [0, 75, 75, 125, 125, 2, 0.9]], np.float32)}

    def _coreml_nms(self):
        return {"coordinates": np.array([[0.5, 0.5, 100 / 640, 200 / 640]], np.float32),
                "confidence": np.array([[0.9] + [0.0] * 79], np.float32)}

    def _split(self, names=("boxes", "scores", "class_idx")):
        # person (320,320,100x200) + its duplicate (-> NMS) + a car + a low-score person
        boxes = np.array([[[270, 220, 370, 420], [271, 221, 371, 419], [75, 75, 125, 125], [0, 0, 50, 90]]],
                         np.float32)
        return {names[0]: boxes, names[1]: np.array([[0.9, 0.8, 0.95, 0.1]], np.float32),
                names[2]: np.array([[0, 0, 2, 0]], np.float32)}

    def predict(self, feeds):
        return {"decoded": self._decoded, "transposed": lambda: self._decoded(True), "raw": self._raw,
                "end2end": self._end2end, "coreml_nms": self._coreml_nms, "split": self._split,
                "split_unnamed": lambda: self._split(("var_10", "var_11", "var_12"))}[self.layout]()

    def predict_batch(self, feeds_list):
        return [self.predict(f) for f in feeds_list]


def test_letterbox_center():
    canvas, (scale, px, py) = letterbox_center(np.full((720, 1280, 3), 200, np.uint8), (640, 640))
    assert canvas.shape == (640, 640, 3) and scale == pytest.approx(0.5)
    assert (px, py) == (0, 140)
    assert canvas[100, 0, 0] == 114 and canvas[320, 320, 0] == 200


@pytest.mark.parametrize("layout", ["decoded", "transposed", "raw", "end2end", "coreml_nms", "split",
                                    "split_unnamed"])
def test_yolo_layouts_return_one_person_in_frame_pixels(layout):
    eng = YOLOv7Engine(FakeYOLO(layout), conf_threshold=0.4)
    dets = eng.detect(np.zeros((720, 1280, 3), np.uint8))
    assert dets.shape == (1, 5)
    np.testing.assert_allclose(dets[0, :4], _expected_frame_box(), atol=1.0)
    assert dets[0, 4] > 0.4


def test_filter_persons_min_height_and_roi():
    dets = np.array([[0, 0, 50, 20, 0.9], [100, 100, 200, 400, 0.9], [900, 100, 1000, 400, 0.9]], np.float32)
    assert filter_persons(dets, min_height=30).shape[0] == 2
    assert filter_persons(dets, min_height=30, roi=[0, 0, 640, 720]).shape[0] == 1


def test_tracker_counts_distinct_people_once():
    tr = PersonTracker(iou_threshold=0.3, max_age_s=2.0, min_hits=2)
    a = np.array([[100, 100, 200, 400]], np.float32)
    assert tr.update(a, 0.0) == 0          # needs 2 hits
    assert tr.update(a + 10, 0.25) == 1    # confirmed
    for i in range(10):                    # same person standing still -> not re-counted
        assert tr.update(a + 10, 0.5 + i * 0.25) == 0
    b = np.array([[100, 100, 200, 400], [800, 100, 900, 400]], np.float32)
    tr.update(b, 3.0)
    assert tr.update(b, 3.25) == 1         # second person
    assert tr.update(np.zeros((0, 5), np.float32), 10.0) == 0 and tr.active == 0
    tr.update(a, 10.25)
    assert tr.update(a, 10.5) == 1         # came back after leaving -> counted again


def test_tracker_follows_fast_walker_at_low_fps():
    tr = PersonTracker(iou_threshold=0.3, max_age_s=2.0, min_hits=2)
    total = 0
    for i in range(8):  # moves 80 px (< 0.6 * 300 px height) between frames: IoU ~0.1
        x = 100 + i * 80
        total += tr.update(np.array([[x, 100, x + 100, 400]], np.float32), i * 0.25)
    assert total == 1


def test_counter_interval_rows():
    c = PersonCounter(["cam01", "cam02"], interval_s=60, track_min_hits=1)
    box = np.array([[100, 100, 200, 400]], np.float32)
    c.update("cam01", np.zeros((0, 5), np.float32), 1000.0, 0.0)
    c.update("cam01", box, 1001.0, 0.25)
    c.update("cam01", np.vstack([box, box + 600]), 1002.0, 0.5)
    rows = c.flush()
    assert len(rows) == 1  # cam02 saw no frames
    r = rows[0]
    assert (r.camera_id, r.frames, r.min_count, r.max_count, r.current_count, r.new_persons) == \
        ("cam01", 3, 0, 2, 2, 2)
    assert r.avg_count == pytest.approx(1.0)
    assert c.flush() == []
    assert c.stats()["persons_since_start"] == 2


def test_repository_insert_and_totals():
    db = Database(":memory:")
    create_tables(db)
    repo = PersonCountRepository(db)
    repo.insert_many([("cam01", 0, 60, 240, 1, 0, 3, 1.2, 5, 4, 1, 3, 60),
                      ("cam01", 60, 120, 240, 0, 0, 2, 0.5, 2, 0, 3, 0, 120),
                      ("cam02", 0, 60, 240, 0, 0, 1, 0.1, 1, 0, 0, 0, 60)])
    totals = {t["camera_id"]: t for t in repo.totals()}
    assert totals["cam01"]["total_persons"] == 7 and totals["cam01"]["peak_count"] == 3
    assert totals["cam02"]["total_persons"] == 1
    assert (totals["cam01"]["in_count"], totals["cam01"]["out_count"]) == (4, 4)
    assert len(repo.query(camera_id="cam01")) == 2
    assert repo.delete_older_than(61) == 2


def test_service_end_to_end_writes_rows():
    db = Database(":memory:")
    create_tables(db)
    repo = PersonCountRepository(db)
    cam = CameraConfig.from_dict({"camera_id": "cam01", "url": "test://", "tasks": ["person"]})
    sched = FrameScheduler(max_frame_age_s=5)
    rx = FrameReceiver("cam01", [sched.register("cam01")])
    svc = PersonCountService(YOLOv7Engine(FakeYOLO("decoded")), sched, repo, [cam],
                             PersonSettings(flush_interval_s=3600, track_min_hits=1, max_wait_ms=1))
    svc.start()
    import time
    for _ in range(3):
        rx(np.zeros((720, 1280, 3), np.uint8))
        time.sleep(0.1)
    svc.stop()  # flushes the partial interval
    rows = repo.query()
    assert len(rows) == 1 and rows[0]["max_count"] == 1 and rows[0]["new_persons"] == 1
    assert rows[0]["frames"] >= 1


def test_camera_tasks_parsing():
    assert CameraConfig.from_dict({"camera_id": "a", "url": "x"}).tasks == ["face"]
    c = CameraConfig.from_dict({"camera_id": "a", "url": "x", "tasks": "face, person"})
    assert c.runs_face and c.runs_person
    c = CameraConfig.from_dict({"camera_id": "a", "url": "x"}, defaults={"tasks": ["person"]})
    assert c.tasks == ["person"] and not c.runs_face
    with pytest.raises(ValueError):
        CameraConfig.from_dict({"camera_id": "a", "url": "x", "tasks": ["cars"]})


def test_frame_receiver_fans_out_to_each_task_queue():
    q1, q2 = CameraQueue("cam01"), CameraQueue("cam01")
    rx = FrameReceiver("cam01", [q1, q2])
    rx(np.zeros((4, 4, 3), np.uint8))
    f1, f2 = q1.pop_latest(), q2.pop_latest()
    assert f1 is f2 and f1.frame_id == 1


def _walk(lc, tid, ys, t0=0.0):
    """Feet of one tracked person at these y positions (x = 50), one frame each."""
    out = []
    for i, y in enumerate(ys):
        out += lc.update([tid], np.array([[40, y - 100, 60, y]], np.float32), t0 + i)
    return [c.direction for c in out]


def test_person_walking_in_counts_only_in_even_if_feet_bounce_back():
    from person_detection.line_counter import LineCounter

    lc = LineCounter("16", [0, 100, 200, 100], in_direction="down")
    # walks down across the line (IN), stops at the door, box bottom jumps back above the line, goes in
    dirs = _walk(lc, 1, [60, 80, 120, 140, 140, 90, 85, 130, 150])
    assert dirs == ["in"]
    assert (lc.total_in, lc.total_out) == (1, 0)
    assert _walk(lc, 2, [160, 140, 80, 60]) == ["out"]                  # someone else walks out
    assert (lc.total_in, lc.total_out) == (1, 1)


def test_count_once_off_keeps_old_behaviour():
    from person_detection.line_counter import LineCounter

    lc = LineCounter("16", [0, 100, 200, 100], in_direction="down", count_once=False)
    assert _walk(lc, 1, [60, 80, 120, 140, 140, 90, 85, 130, 150]) == ["in", "out", "in"]
