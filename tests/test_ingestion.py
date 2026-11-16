import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import CameraConfig, load_cameras  # noqa: E402
from ingestion.camera_health import CameraHealthRegistry  # noqa: E402
from ingestion.camera_pipeline import build_pipeline_description, mask_url  # noqa: E402
from ingestion.frame_receiver import FrameReceiver  # noqa: E402
from scheduler.batch_scheduler import BatchScheduler  # noqa: E402
from scheduler.frame_scheduler import FrameScheduler  # noqa: E402
from scheduler.inference_queue import InferenceQueue  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _img():
    return np.zeros((4, 4, 3), np.uint8)


def test_cameras_yaml_has_20_cameras():
    cams = load_cameras(ROOT / "config" / "cameras.yaml")
    assert len(cams) == 20
    assert all(c.decoder == "vtdec" for c in cams)


def test_pipeline_uses_hardware_decoder_and_drops_frames():
    cam = CameraConfig(camera_id="c", name="c", url="rtsp://u:p@1.2.3.4/s", fps=4, width=640, height=360)
    desc = build_pipeline_description(cam)
    assert "rtph264depay ! h264parse ! vtdec" in desc
    assert "framerate=4/1" in desc and "drop=true" in desc and "max-buffers=1" in desc
    assert "width=640,height=360" in desc and "format=BGR" in desc


def test_pipeline_h265_and_test_source():
    cam = CameraConfig(camera_id="c", name="c", url="rtsp://h/s", codec="h265")
    assert "rtph265depay ! h265parse" in build_pipeline_description(cam)
    cam = CameraConfig(camera_id="c", name="c", url="test://ball")
    assert "videotestsrc" in build_pipeline_description(cam)


def test_password_special_chars_quoted_and_masked():
    cam = CameraConfig(camera_id="c", name="c", url="rtsp://admin:p@ss#1@10.0.0.2:554/s")
    desc = build_pipeline_description(cam)
    assert "p%40ss%231@10.0.0.2" in desc
    assert mask_url(cam.url) == "rtsp://***@10.0.0.2:554/s"


def test_camera_queue_keeps_latest_only():
    sched = FrameScheduler()
    q = sched.register("c1")
    r = FrameReceiver("c1", q)
    for _ in range(5):
        r(_img())
    f = q.pop_latest()
    assert f.frame_id == 5 and q.dropped == 4 and q.pop_latest() is None


def test_scheduler_is_fair_and_one_frame_per_camera():
    sched = FrameScheduler()
    health = CameraHealthRegistry()
    recv = {c: FrameReceiver(c, sched.register(c), health.add(c)) for c in ("a", "b", "c")}
    for r in recv.values():
        r(_img())
    frames = sched.get_frames(10, 0.1)
    assert sorted(f.camera_id for f in frames) == ["a", "b", "c"]
    # 'a' produces a lot, but b and c must still be served first when they have frames
    recv["b"](_img())
    recv["c"](_img())
    time.sleep(0.01)
    recv["a"](_img())
    frames = sched.get_frames(2, 0.1)
    assert len(frames) == 2
    assert health.get("a").frames == 2


def test_scheduler_drops_stale_frames():
    sched = FrameScheduler(max_frame_age_s=0.05)
    r = FrameReceiver("c1", sched.register("c1"))
    r(_img())
    time.sleep(0.1)
    assert sched.get_frames(4, 0.01) == []
    assert sched.stale_dropped == 1


def test_scheduler_wakes_on_new_frame():
    sched = FrameScheduler()
    r = FrameReceiver("c1", sched.register("c1"))
    threading.Timer(0.05, lambda: r(_img())).start()
    t0 = time.monotonic()
    frames = sched.get_frames(1, 2.0)
    assert len(frames) == 1 and time.monotonic() - t0 < 1.0


def test_inference_queue_drops_oldest():
    q = InferenceQueue(maxsize=3)
    for i in range(5):
        q.put(i)
    assert q.get_batch(10, 0) == [2, 3, 4] and q.dropped == 2


def test_batch_scheduler_collects_until_max():
    q = InferenceQueue(100)
    for i in range(10):
        q.put(i)
    b = BatchScheduler(q.get_batch, max_batch=4, max_wait_ms=5)
    assert b.next_batch() == [0, 1, 2, 3]
    assert len(b.next_batch()) == 4
    assert b.next_batch() == [8, 9]
    assert b.next_batch() == []


def test_file_sources_use_a_non_leaky_queue():
    from ingestion.camera_pipeline import build_pipeline_description
    f = build_pipeline_description(CameraConfig.from_dict({"camera_id": "f", "url": "file:///tmp/a.mp4"}))
    live = build_pipeline_description(CameraConfig.from_dict({"camera_id": "l", "url": "rtsp://h/s"}))
    assert "leaky" not in f and "sync=true" in f          # files: paced by back-pressure, real time
    assert "leaky=downstream" in live and "sync=false" in live


def test_explain_error_points_at_codec():
    from ingestion.camera_pipeline import explain_error

    dbg = "../gst/rtsp/gstrtspsrc.c(3266): gst_rtspsrc_loop (): streaming stopped, reason not-linked (-1)"
    msg = explain_error("Internal data stream error.", dbg, "h264")
    assert "reason: not-linked" in msg and "codec: h265 or codec: auto" in msg
    assert "codec: h264 or codec: auto" in explain_error("Internal data stream error.", dbg, "h265")
    assert explain_error("Could not open resource for reading.", None) == "Could not open resource for reading."
    assert explain_error("x", "streaming stopped, reason error (-5)") == "x (reason: error)"
