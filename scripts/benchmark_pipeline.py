#!/usr/bin/env python3
"""End-to-end pipeline benchmark with simulated cameras (no network / GStreamer needed).

Feeds N synthetic cameras at F fps into the real scheduler -> SCRFD -> ArcFace ->
matcher -> event pipeline and reports throughput, drops and CPU usage.

    python scripts/benchmark_pipeline.py --image group_photo.jpg --cameras 20 --fps 4 --seconds 60
    python scripts/benchmark_pipeline.py --video sample.mp4 --cameras 20
"""
import argparse
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

cv2.setNumThreads(1)

from config.settings import CameraConfig, load_settings  # noqa: E402
from ingestion.camera_health import CameraHealthRegistry  # noqa: E402
from ingestion.frame_receiver import FrameReceiver  # noqa: E402
from inference.inference_manager import InferenceManager  # noqa: E402
from monitoring.cpu_monitor import CPUMonitor  # noqa: E402
from monitoring.memory_monitor import MemoryMonitor  # noqa: E402
from monitoring.performance import PerformanceTracker  # noqa: E402
from recognition.duplicate_filter import DuplicateFilter, TrackCache  # noqa: E402
from recognition.embedding_manager import EmbeddingGallery  # noqa: E402
from recognition.face_matcher import FaceMatcher  # noqa: E402
from scheduler.frame_scheduler import FrameScheduler  # noqa: E402
from services.recognition_service import RecognitionService  # noqa: E402
from utils.logger import setup_logging  # noqa: E402


class _LocalOffenders:
    """In-process stand-in for the tbl_offenders search, so the benchmark measures the models, not PostgreSQL."""

    def __init__(self, matcher: FaceMatcher) -> None:
        self.matcher = matcher

    def search(self, embedding):
        r = self.matcher.match(embedding)
        return {"offender_id": r.person_id, "offender_name": f"person-{r.person_id}", "similarity": r.similarity}

    def log_search(self, **row):
        return None


def load_frames(args, size):
    w, h = size
    if args.video:
        cap = cv2.VideoCapture(args.video)
        frames = []
        while len(frames) < 200:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(cv2.resize(f, (w, h)))
        cap.release()
        if frames:
            return frames
    if args.image:
        img = cv2.imread(args.image)
        if img is not None:
            img = cv2.resize(img, (w, h))
            # slight shifts so frames are not byte-identical
            return [np.roll(img, s, axis=1) for s in range(0, 40, 4)]
    print("WARNING: no --image/--video: using noise (measures detection load only)")
    return [np.random.randint(0, 255, (h, w, 3), np.uint8) for _ in range(5)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", type=int, default=20)
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--image")
    ap.add_argument("--video")
    ap.add_argument("--no-track-cache", action="store_true", help="always run ArcFace (worst case)")
    args = ap.parse_args()

    settings = load_settings()
    setup_logging(settings.logging_config, "WARNING")
    perf = PerformanceTracker()

    inference = InferenceManager(settings.models, settings.detection, perf).load()
    gallery = EmbeddingGallery()
    # Fake gallery of 500 people so matching cost is realistic.
    rng = np.random.default_rng(0)
    gallery.replace(np.repeat(np.arange(1, 501), 5), rng.standard_normal((2500, 512)).astype(np.float32))
    offenders = _LocalOffenders(FaceMatcher(gallery, settings.recognition.match_threshold,
                                            settings.recognition.match_margin))

    cams = [CameraConfig(camera_id=f"sim{i:02d}", name=f"sim{i:02d}", url="test://", fps=args.fps,
                         width=args.width, height=args.height) for i in range(args.cameras)]
    sched = FrameScheduler(settings.detection.max_frame_age_s)
    health = CameraHealthRegistry()
    receivers = [FrameReceiver(c.camera_id, sched.register(c.camera_id), health.add(c.camera_id)) for c in cams]

    rc = settings.recognition
    if args.no_track_cache:
        rc.track_reverify_s = 0
    svc = RecognitionService(
        inference, sched, offenders,
        DuplicateFilter(rc.event_cooldown_s, rc.unknown_cooldown_s, rc.unknown_similarity),
        TrackCache(rc.track_iou, rc.track_reverify_s), cams, settings.detection, rc, perf,
    )
    frames = load_frames(args, (args.width, args.height))
    stop = threading.Event()

    def feeder():
        interval = 1.0 / args.fps
        # stagger cameras evenly inside each interval, like real independent streams
        offsets = [i * interval / len(receivers) for i in range(len(receivers))]
        start = time.monotonic()
        tick = 0
        while not stop.is_set():
            base = start + tick * interval
            for i, r in enumerate(receivers):
                delay = base + offsets[i] - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                r(frames[(tick + i) % len(frames)])
            tick += 1

    cpu, mem = CPUMonitor(), MemoryMonitor()
    svc.start()
    threading.Thread(target=feeder, daemon=True).start()
    print(f"running {args.cameras} cams x {args.fps} fps for {args.seconds:.0f}s ...")
    cpu_samples = []
    t_end = time.time() + args.seconds
    time.sleep(3)
    cpu.sample()  # reset the interval after warm-up
    while time.time() < t_end:
        time.sleep(5)
        s = cpu.sample()
        cpu_samples.append(s["process_pct_of_core"])
        c = perf.snapshot()["counters"]
        print(f"  frames {c.get('frames_processed', {}).get('per_s', 0):6.1f}/s  "
              f"faces {c.get('faces_detected', {}).get('per_s', 0):6.1f}/s  "
              f"embedded {c.get('faces_embedded', {}).get('per_s', 0):6.1f}/s  "
              f"busy {perf.utilization('inference') * 100:5.1f}%  CPU {s['process_pct_of_core']:5.1f}% of 1 core")
    stop.set()
    svc.stop()

    snap = perf.snapshot()
    offered = args.cameras * args.fps
    processed = snap["counters"].get("frames_processed", {}).get("per_s", 0)
    print("\n=== summary ===")
    print(f"offered load      : {offered:.0f} frames/s")
    print(f"processed         : {processed:.1f} frames/s  ({processed / offered * 100:.0f}%)")
    print(f"scheduler         : {json.dumps({k: v for k, v in sched.stats().items() if k != 'per_camera'})}")
    print(f"pipeline          : {json.dumps(svc.stats())}")
    print(f"avg process CPU   : {np.mean(cpu_samples) if cpu_samples else 0:.0f}% of one core "
          f"({cpu.ncpu} cores available)")
    print(f"memory            : {mem.sample()['process_rss_mb']} MB RSS")
    print("latency           :", json.dumps(snap["latency"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
