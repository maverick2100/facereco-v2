#!/usr/bin/env python3
"""Check camera ingestion (GStreamer + VideoToolbox) without any models.

    python scripts/test_gstreamer.py                       # all enabled cameras, 20 s
    python scripts/test_gstreamer.py --cameras cam01,cam02 --seconds 30
    python scripts/test_gstreamer.py --url test://ball     # synthetic source
    python scripts/test_gstreamer.py --url rtsp://user:pass@ip/stream --codec auto   # h264 | h265 | auto
    python scripts/test_gstreamer.py --cameras cam01 --save-frame /tmp/cam01.jpg
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from config.settings import CameraConfig, load_settings  # noqa: E402
from ingestion.camera_health import CameraHealthRegistry  # noqa: E402
from ingestion.camera_pipeline import build_pipeline_description, mask_url  # noqa: E402
from ingestion.frame_receiver import FrameReceiver  # noqa: E402
from ingestion.gst_manager import GstManager, gstreamer_available, load_gstreamer, missing_elements  # noqa: E402
from monitoring.cpu_monitor import CPUMonitor  # noqa: E402
from scheduler.frame_scheduler import FrameScheduler  # noqa: E402
from utils.logger import setup_logging  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", help="comma-separated camera ids")
    ap.add_argument("--url", help="ad-hoc url (rtsp://, file://, test://)")
    ap.add_argument("--codec", choices=["h264", "h265", "auto"],
                    help="override the codec (for --url, and for --cameras instead of cameras.yaml)")
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--save-frame", help="save the last frame of the first camera to this path")
    args = ap.parse_args()

    settings = load_settings()
    setup_logging(settings.logging_config, "INFO")
    if not gstreamer_available():
        print("GStreamer/PyGObject not importable. See README (brew install gstreamer pygobject3).")
        return 1
    Gst, _ = load_gstreamer()
    print("GStreamer", Gst.version_string())
    missing = missing_elements()
    print("missing elements:", ", ".join(missing) if missing else "none")

    if args.url:
        # same defaults as cameras.yaml (codec, size, fps...), so the test matches a real run
        cams = [CameraConfig.from_dict({"camera_id": "adhoc", "url": args.url}, settings.camera_source.defaults)]
    else:
        cams = settings.enabled_cameras
        if args.cameras:
            wanted = set(args.cameras.split(","))
            cams = [c for c in settings.cameras if c.camera_id in wanted]
    if not cams:
        print("no cameras selected")
        return 1
    if args.codec:
        for c in cams:
            c.codec = args.codec
    print("pipeline example:", build_pipeline_description(cams[0]).replace(cams[0].url, mask_url(cams[0].url)))

    sched = FrameScheduler(max_frame_age_s=1e9)
    health = CameraHealthRegistry()
    receivers = {}
    for c in cams:
        q = sched.register(c.camera_id)
        receivers[c.camera_id] = FrameReceiver(c.camera_id, q, health.add(c.camera_id, c.name))

    cpu = CPUMonitor()
    mgr = GstManager(cams, receivers, health, settings.ingestion)
    mgr.start()
    last = None
    try:
        t_end = time.time() + args.seconds
        while time.time() < t_end:
            time.sleep(5)
            print(f"\n--- CPU process {cpu.sample()['process_pct_of_core']}% of one core")
            for s in health.snapshot():
                print(f"{s['camera_id']:<10} {s['status']:<13} fps={s['fps']:<6} frames={s['frames']:<6} "
                      f"reconnects={s['reconnects']} err={s['last_error'] or ''}")
            if args.save_frame:
                f = sched.queue(cams[0].camera_id).pop_latest()
                last = f or last
    finally:
        mgr.stop()
    if args.save_frame and last is not None:
        cv2.imwrite(args.save_frame, last.image)
        print("saved", args.save_frame, last.image.shape)
    return 0


if __name__ == "__main__":
    sys.exit(main())
