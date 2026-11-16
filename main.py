#!/usr/bin/env python3
"""Multi-camera face recognition + person counting for Apple Silicon (Neural Engine).

Each camera's `tasks` in config/cameras.yaml picks what runs on it:
    face   -> SCRFD + ArcFace recognition      person -> YOLOv7 person detection + counting

Usage:
    python main.py run                       # start all enabled cameras
    python main.py enroll                    # enrol everyone in data/enrollment/<name>/*.jpg
    python main.py enroll --name "Alice" --images a.jpg b.jpg
    python main.py persons                   # list enrolled people
    python main.py delete-person 3
    python main.py export --since 2026-11-13 # CSV into data/exports/
    python main.py counts --since 2026-11-13 # person totals per camera (--csv for all rows)
    python main.py info                      # model + environment info
    python main.py run --show                # same, plus a window with live boxes + names (q to quit)
    python main.py run --show cam01,cam02    # window shows only these cameras
    python main.py db-check                  # test the database connection (SQLite or PostgreSQL)
    python main.py cameras                   # what the camera source (cameras.yaml or your SP) returns
    python main.py api                       # HTTP API for the frontend (cameras in, dashboard out) on :8090
    python main.py run --api                 # pipelines + the API in one process
    python main.py db-setup                  # create camera_config + the SPs in PostgreSQL (--force: re-apply)

`run` follows the camera source: with FR_CAMERA_SOURCE=db it calls your stored procedure, starts face
recognition / person counting per camera as the SP says, and restarts the pipelines when the SP's
answer changes (NOTIFY camera_changed, or a re-check every FR_CAMERA_POLL_S).
"""
from __future__ import annotations

import os

# Keep CPU-side math libraries single-threaded: the heavy lifting belongs on the ANE,
# and 20 cameras' worth of small numpy/OpenCV ops are faster without thread-pool churn.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import time  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import signal  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Dict, List, Optional  # noqa: E402

import cv2  # noqa: E402

cv2.setNumThreads(1)
cv2.setUseOptimized(True)

from config.settings import Settings, load_settings  # noqa: E402
from constants import ENROLLMENT_DIR, EXPORTS_DIR  # noqa: E402
from database.camera_repository import CameraRepository  # noqa: E402
from database.connection import open_database  # noqa: E402
from database.embedding_repository import EmbeddingRepository  # noqa: E402
from database.event_repository import EventRepository  # noqa: E402
from database.person_count_repository import PersonCountRepository  # noqa: E402
from database.person_repository import PersonRepository  # noqa: E402
from database.schema import init_schema  # noqa: E402
from ingestion.camera_health import CameraHealthRegistry  # noqa: E402
from ingestion.frame_receiver import FrameReceiver, OpenCVManager  # noqa: E402
from inference.inference_manager import InferenceManager  # noqa: E402
from monitoring.performance import PerformanceTracker  # noqa: E402
from monitoring.system_health import SystemHealth  # noqa: E402
from person_detection.person_count_service import PersonCountService, load_yolo  # noqa: E402
from recognition.duplicate_filter import DuplicateFilter, TrackCache  # noqa: E402
from recognition.embedding_manager import EmbeddingGallery  # noqa: E402
from recognition.face_matcher import FaceMatcher  # noqa: E402
from recognition.identity_manager import IdentityManager  # noqa: E402
from scheduler.frame_scheduler import FrameScheduler  # noqa: E402
from services.enrollment_service import EnrollmentService  # noqa: E402
from services.event_service import EventService  # noqa: E402
from services.live_view import LiveView  # noqa: E402
from ingestion.camera_pipeline import mask_url  # noqa: E402
from services.camera_source import CameraSource, describe_change, fingerprint  # noqa: E402
from services.recognition_service import RecognitionService  # noqa: E402
from utils.logger import setup_logging  # noqa: E402
from utils.time_utils import iso, parse_datetime  # noqa: E402

log = logging.getLogger("main")


class FaceRecognitionApp:
    """Composes and runs the full system."""

    def __init__(self, settings: Settings, cameras=None, show: Optional[List[str]] = None) -> None:
        self.settings = settings
        self.cameras = list(cameras) if cameras is not None else settings.enabled_cameras
        self.face_cameras = [c for c in self.cameras if c.runs_face]
        self.person_cameras = [c for c in self.cameras if c.runs_person]
        self.perf = PerformanceTracker()
        self.db = open_database(settings.storage)
        init_schema(self.db)
        log.info("Database: %s", self.db.description)
        self.camera_repo = CameraRepository(self.db)
        self.person_repo = PersonRepository(self.db)
        self.embedding_repo = EmbeddingRepository(self.db)
        self.event_repo = EventRepository(self.db)
        self.person_count_repo = PersonCountRepository(self.db)

        self.inference = InferenceManager(settings.models, settings.detection, self.perf)
        self.gallery = EmbeddingGallery()
        self.identity = IdentityManager(self.person_repo, self.embedding_repo, self.gallery,
                                        settings.recognition.gallery_reload_s)
        self.events = EventService(self.event_repo, settings.storage, self.perf)
        self.health_registry = CameraHealthRegistry()
        self.scheduler = FrameScheduler(max_frame_age_s=settings.detection.max_frame_age_s)
        person_fps = settings.person.max_fps
        self.person_scheduler = FrameScheduler(max_frame_age_s=settings.detection.max_frame_age_s,
                                               min_interval_s=1.0 / person_fps if person_fps > 0 else 0.0)
        self.receivers: Dict[str, FrameReceiver] = {}
        self.ingestion = None
        self.recognition: Optional[RecognitionService] = None
        self.person_service: Optional[PersonCountService] = None
        self.system_health: Optional[SystemHealth] = None
        # --show: on-screen view (None = off, [] = all cameras, else these camera ids)
        self.live: Optional[LiveView] = None
        if show is not None:
            shown = [c for c in self.cameras if not show or c.camera_id in show]
            self.live = LiveView({c.camera_id: c.name for c in shown}, {c.camera_id: c.roi for c in shown},
                                 zones={c.camera_id: c.count_zone for c in shown if c.runs_person})
        self._stop = threading.Event()

    def _make_ingestion(self):
        backend = self.settings.ingestion.backend
        if backend in ("auto", "gstreamer"):
            from ingestion.gst_manager import GstManager, gstreamer_available

            if gstreamer_available():
                return GstManager(self.cameras, self.receivers, self.health_registry, self.settings.ingestion)
            if backend == "gstreamer":
                raise RuntimeError("GStreamer (PyGObject) not available - see README 'Install'")
            log.warning("GStreamer not available - falling back to OpenCV (higher CPU)")
        return OpenCVManager(self.cameras, self.receivers, self.health_registry, self.settings.ingestion)

    def start(self) -> None:
        s = self.settings
        if not self.cameras:
            raise RuntimeError("No enabled cameras in config/cameras.yaml")
        log.info("Starting with %d cameras (face: %d, person: %d), compute units=%s", len(self.cameras),
                 len(self.face_cameras), len(self.person_cameras), s.models.compute_units)
        self.camera_repo.sync(self.cameras)

        if self.face_cameras:
            self.inference.load()
            self.identity.reload()
            if self.gallery.size == 0:
                log.warning("Gallery is empty - every face will be 'unknown'. Run: python main.py enroll")
        if self.person_cameras:
            self.person_service = PersonCountService(
                load_yolo(s.models, s.person), self.person_scheduler, self.person_count_repo,
                self.person_cameras, s.person, self.perf, retention_days=s.storage.retention_days,
                live=self.live,
            )
            self.person_service.warmup(s.models.warmup_iterations)

        for cam in self.cameras:
            queues = []
            if cam.runs_face:
                queues.append(self.scheduler.register(cam.camera_id, cam.priority, s.ingestion.queue_depth))
            if cam.runs_person:
                queues.append(self.person_scheduler.register(cam.camera_id, cam.priority, s.ingestion.queue_depth))
            h = self.health_registry.add(cam.camera_id, cam.name)
            self.receivers[cam.camera_id] = FrameReceiver(cam.camera_id, queues, h)

        if self.face_cameras:
            matcher = FaceMatcher(self.gallery, s.recognition.match_threshold, s.recognition.match_margin)
            duplicates = DuplicateFilter(s.recognition.event_cooldown_s, s.recognition.unknown_cooldown_s,
                                         s.recognition.unknown_similarity)
            tracks = TrackCache(s.recognition.track_iou, s.recognition.track_reverify_s)
            self.events.start()
            self.recognition = RecognitionService(
                self.inference, self.scheduler, self.identity, matcher, duplicates, tracks, self.events,
                self.face_cameras, s.detection, s.recognition, self.perf, s.storage.save_snapshots,
                live=self.live,
            )
            self.recognition.start()
        if self.person_service is not None:
            self.person_service.start()

        self.ingestion = self._make_ingestion()
        self.ingestion.start()

        extra = {"models": self._models_info}
        if self.recognition is not None:
            extra.update({
                "scheduler": lambda: {k: v for k, v in self.scheduler.stats().items() if k != "per_camera"},
                "pipeline": self.recognition.stats,
                "gallery": lambda: {"persons": self.gallery.num_persons, "embeddings": self.gallery.size},
            })
        if self.person_service is not None:
            extra.update({
                "person_scheduler": lambda: {k: v for k, v in self.person_scheduler.stats().items()
                                             if k != "per_camera"},
                "person_counts": self.person_service.stats,
            })
        self.system_health = SystemHealth(s.monitoring, self.perf, self.health_registry, extra=extra)
        self.system_health.start()
        log.info("System running - Ctrl+C to stop")

    def _models_info(self) -> dict:
        info = self.inference.info()
        if self.person_service is not None:
            info["yolov7"] = self.person_service.engine.model.info()
        return info

    def run_forever(self) -> None:
        signal.signal(signal.SIGINT, lambda *_: self._stop.set())
        signal.signal(signal.SIGTERM, lambda *_: self._stop.set())
        self.start()
        try:
            if self.live is not None:
                self.live.run_window(self._stop, self._camera_status)  # blocks until q / Esc / Ctrl+C
            while not self._stop.wait(1.0):
                pass
        finally:
            self.stop()

    def _camera_status(self) -> Dict[str, str]:
        st = {h["camera_id"]: f'{h["status"]} {h["fps"]:.1f} fps' for h in self.health_registry.snapshot()}
        if self.person_service is not None:
            for cid, p in self.person_service.stats()["per_camera"].items():
                if "in" in p and cid in st:
                    st[cid] += f'  IN {p["in"]} OUT {p["out"]} inside {p["in_zone"]}'
        return st

    def stop(self) -> None:
        log.info("Shutting down...")
        for component in (self.ingestion, self.recognition, self.person_service, self.system_health):
            if component is not None:
                try:
                    component.stop()
                except Exception:
                    log.exception("error stopping %s", type(component).__name__)
        self.events.stop()
        self.db.close()
        log.info("Stopped")


# ---------------------------------------------------------------------- commands
def _repos(settings: Settings):
    db = open_database(settings.storage)
    init_schema(db)
    return db, PersonRepository(db), EmbeddingRepository(db), EventRepository(db)


class Supervisor:
    """Runs whatever the camera source says, and follows it when it changes.

    1. load the cameras (cameras.yaml, or your SP with FR_CAMERA_SOURCE=db)
    2. start face recognition / person counting on each camera according to its use case
    3. watch for changes (NOTIFY camera_changed, or a re-check every FR_CAMERA_POLL_S); when the
       list differs, flush what was counted so far and restart the pipelines with the new list
    """

    SETTLE_S = 1.0          # wait for a burst of changes (frontend saving several cameras) to finish
    RETRY_S = 30.0          # after a failed start (model missing, DB down), try again this often

    def __init__(self, settings: Settings, source: CameraSource, show: Optional[List[str]] = None,
                 only: Optional[set] = None) -> None:
        self.settings = settings
        self.source = source
        self.show = show
        self.only = only
        self._stop = threading.Event()
        self._cycle = threading.Event()
        self._next = None

    def _load(self):
        cams = self.source.load()
        if self.only:
            cams = [c for c in cams if c.camera_id in self.only]
        return cams

    def _load_until_ok(self):
        while not self._stop.is_set():
            try:
                return self._load()
            except Exception as exc:
                log.error("Could not load cameras from %s: %s - retrying in %.0fs", self.source.description,
                          exc, self.settings.camera_source.poll_s)
                self._stop.wait(self.settings.camera_source.poll_s)
        return []

    def _watch(self, current, cycle: threading.Event) -> None:
        fp = fingerprint(current)
        poll = self.settings.camera_source.poll_s
        while not cycle.is_set():
            notified = self.source.wait_for_change(poll, cycle)
            if cycle.is_set():
                return
            if notified:
                cycle.wait(self.SETTLE_S)
            try:
                new = self._load()
            except Exception as exc:
                log.warning("Camera re-check failed (%s) - keeping the current cameras", exc)
                continue
            if fingerprint(new) != fp:
                log.info("Camera list changed (%s) - restarting pipelines", describe_change(current, new))
                self._next = new
                cycle.set()
                return

    def _summary(self, cams) -> str:
        face = [c.camera_id for c in cams if c.runs_face]
        person = [c.camera_id for c in cams if c.runs_person]
        return f"{len(cams)} cameras - face: {', '.join(face) or '-'} | person: {', '.join(person) or '-'}"

    def _run_once(self, cams, cycle: threading.Event, watcher: threading.Thread) -> bool:
        """Run the pipelines for `cams` until `cycle` is set. False if they failed to start."""
        log.info("Running %s", self._summary(cams))
        show = None                                     # None = no window, [] = every camera
        if self.show is not None:
            running = {c.camera_id for c in cams}
            show = [c for c in self.show if c in running]
        app = None
        try:
            app = FaceRecognitionApp(self.settings, cams, show)
            app.start()
            watcher.start()
            if app.live is not None:
                app.live.run_window(cycle, app._camera_status)   # q / Esc / closing the window sets cycle
                if self._next is None:                            # closed by the user, not a reload
                    self._stop.set()
            else:
                cycle.wait()
            return True
        except Exception:
            log.exception("Pipelines failed to start - retrying in %.0fs (or when the cameras change)",
                          self.RETRY_S)
            return False
        finally:
            if app is not None:
                try:
                    app.stop()     # flushes the partial person-count interval and pending crossings
                except Exception:
                    log.exception("Error while stopping pipelines")

    def run_forever(self) -> None:
        def _signal(*_):
            self._stop.set()
            self._cycle.set()

        signal.signal(signal.SIGINT, _signal)
        signal.signal(signal.SIGTERM, _signal)
        log.info("Camera source: %s", self.source.description)
        cams = self._load_until_ok()
        try:
            while not self._stop.is_set():
                self._cycle = cycle = threading.Event()
                if self._stop.is_set():
                    break
                self._next = None
                watcher = threading.Thread(target=self._watch, args=(cams, cycle), name="camera-watch", daemon=True)
                if not cams:
                    log.warning("No enabled cameras from %s - waiting for one to be added", self.source.description)
                    watcher.start()
                    cycle.wait()
                elif not self._run_once(cams, cycle, watcher):
                    # start failed: retry after RETRY_S, or sooner if the camera list changes
                    if not watcher.is_alive() and watcher.ident is None:
                        watcher.start()
                    cycle.wait(self.RETRY_S)
                cycle.set()
                if watcher.ident is not None:
                    watcher.join(timeout=5)
                if self._next is not None:
                    cams = self._next
                elif not self._stop.is_set():
                    cams = self._load_until_ok()
        finally:
            self.source.close()


def _auto_setup_sql(settings: Settings) -> None:
    """FR_CAMERA_SOURCE=db on PostgreSQL: create camera_config + the SPs if they're missing."""
    if not settings.api.auto_setup or settings.camera_source.source != "db" or \
            settings.storage.backend != "postgres" or "sp_get_cameras" not in settings.camera_source.sql:
        return                                       # your own SP (e.g. fn_get_device_configuration): leave the DB alone
    from database.sql_setup import ensure_sql_objects

    db = None
    try:
        db = open_database(settings.storage)
        ensure_sql_objects(db)
    except Exception as exc:
        log.warning("Could not create camera_config / SPs (%s) - run: python main.py db-setup", exc)
    finally:
        if db is not None:
            db.close()


def cmd_run(settings: Settings, args) -> int:
    only = set(args.cameras.split(",")) if args.cameras else None
    show = None
    if args.show is not None:
        show = [c for c in args.show.split(",") if c]
    _auto_setup_sql(settings)
    api_server = None
    if getattr(args, "api", False):
        from api.server import serve_in_thread

        api_server = serve_in_thread(settings)
    try:
        Supervisor(settings, CameraSource(settings.camera_source, settings.storage), show, only).run_forever()
    finally:
        if api_server is not None:
            api_server.should_exit = True
    return 0


def cmd_api(settings: Settings, args) -> int:
    from api.server import serve

    serve(settings, args.host, args.port)
    return 0


def cmd_db_setup(settings: Settings, args) -> int:
    from database.sql_setup import ensure_sql_objects

    try:
        db = open_database(settings.storage)
        applied = ensure_sql_objects(db, force=args.force)
    except Exception as exc:
        print(f"FAILED: {exc}")
        return 1
    print(f"OK  {db.description}")
    print("    applied: " + (", ".join(applied) if applied else "nothing (all objects exist; --force re-applies)"))
    db.close()
    return 0


def cmd_cameras(settings: Settings, args) -> int:
    """Show what the camera source (cameras.yaml or your SP) returns, without starting anything."""
    src = CameraSource(settings.camera_source, settings.storage)
    try:
        cams = src.load()
    except Exception as exc:
        print(f"FAILED to load cameras from {src.description}: {exc}")
        return 1
    finally:
        src.close()
    print(f"source: {src.description}")
    print(f"{'camera_id':<14} {'name':<22} {'face':<5} {'person':<7} {'zone':<5} url")
    for c in cams:
        print(f"{c.camera_id:<14} {c.name[:22]:<22} {'yes' if c.runs_face else '-':<5} "
              f"{'yes' if c.runs_person else '-':<7} {'yes' if c.count_zone else '-':<5} {mask_url(c.url)}")
    print(f"{len(cams)} enabled cameras: face {sum(c.runs_face for c in cams)}, "
          f"person {sum(c.runs_person for c in cams)}")
    return 0


def cmd_enroll(settings: Settings, args) -> int:
    db, persons, embeddings, _ = _repos(settings)
    inference = InferenceManager(settings.models, settings.detection).load()
    svc = EnrollmentService(inference, persons, embeddings)
    if args.name:
        if not args.images:
            print("--images is required with --name")
            return 2
        results = [svc.enroll_images(args.name, [Path(p) for p in args.images], args.external_id, args.replace)]
    else:
        results = svc.enroll_directory(Path(args.dir), replace=args.replace)
    for r in results:
        print(f"{r.name:<30} id={r.person_id:<5} added={r.added:<3} skipped={r.skipped:<3} errors={len(r.errors)}")
    db.close()
    return 0 if results else 1


def cmd_persons(settings: Settings, args) -> int:
    db, persons, embeddings, _ = _repos(settings)
    for p in persons.list(active_only=not args.all):
        print(f"{p['id']:>5}  {p['name']:<30} embeddings={embeddings.count_for(p['id']):<4} "
              f"active={bool(p['active'])}  external_id={p['external_id'] or ''}")
    db.close()
    return 0


def cmd_delete_person(settings: Settings, args) -> int:
    db, persons, _, _ = _repos(settings)
    ok = persons.delete(args.person_id)
    print("deleted" if ok else "not found")
    db.close()
    return 0 if ok else 1


def cmd_export(settings: Settings, args) -> int:
    db, _, _, events = _repos(settings)
    svc = EventService(events, settings.storage)
    path = svc.export_csv(Path(args.out) if args.out else None, parse_datetime(args.since),
                          parse_datetime(args.until), args.camera, args.known_only)
    print(path)
    db.close()
    return 0


def cmd_counts(settings: Settings, args) -> int:
    db = open_database(settings.storage)
    init_schema(db)
    repo = PersonCountRepository(db)
    since, until = parse_datetime(args.since), parse_datetime(args.until)
    if args.csv is not None:
        rows = repo.query(since, until, args.camera, limit=0)
        path = Path(args.csv) if args.csv else EXPORTS_DIR / f"person_counts_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        cols = ["period_start_time", "period_end_time", "camera_id", "camera_name", "camera_location", "frames",
                "current_count", "min_count", "max_count", "avg_count", "new_persons", "in_count", "out_count",
                "zone_count"]
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in reversed(rows):
                r["period_start_time"], r["period_end_time"] = iso(r["period_start"]), iso(r["period_end"])
                w.writerow(r)
        print(path)
    else:
        totals = repo.totals(since, until, args.camera)
        print(f"{'camera':<12} {'name':<24} {'persons':>8} {'peak':>6} {'in':>6} {'out':>6} {'frames':>9}  from -> to")
        for t in totals:
            print(f"{t['camera_id']:<12} {(t['camera_name'] or ''):<24} {t['total_persons']:>8} "
                  f"{t['peak_count']:>6} {t['in_count'] or 0:>6} {t['out_count'] or 0:>6} {t['frames']:>9}  "
                  f"{iso(t['first_ts'])} -> {iso(t['last_ts'])}")
        print(f"{'TOTAL':<12} {'':<24} {sum(t['total_persons'] for t in totals):>8} {'':>6} "
              f"{sum(t['in_count'] or 0 for t in totals):>6} {sum(t['out_count'] or 0 for t in totals):>6}")
    db.close()
    return 0


def cmd_info(settings: Settings, args) -> int:
    if settings.face_cameras or not settings.person_cameras:
        inference = InferenceManager(settings.models, settings.detection).load()
        print(json.dumps(inference.info(), indent=2))
    if settings.person_cameras or settings.models.yolo_path.exists():
        print(json.dumps({"yolov7": load_yolo(settings.models, settings.person).model.info()}, indent=2))
    print(f"cameras: {len(settings.cameras)} configured, {len(settings.enabled_cameras)} enabled "
          f"(face: {len(settings.face_cameras)}, person: {len(settings.person_cameras)})")
    return 0


def cmd_db_check(settings: Settings, args) -> int:
    try:
        db = open_database(settings.storage)
        init_schema(db)
        counts = {t: db.fetchone(f"SELECT COUNT(*) AS n FROM {t}")["n"] for t in ("persons", "embeddings", "cameras")}
    except Exception as exc:
        print(f"FAILED: {exc}")
        return 1
    print(f"OK  {db.description}")
    print(f"    persons={counts['persons']} embeddings={counts['embeddings']} cameras={counts['cameras']}")
    db.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", help="path to .env file")
    p.add_argument("--cameras-file", help="path to cameras.yaml")
    sub = p.add_subparsers(dest="command")

    r = sub.add_parser("run", help="run real-time recognition")
    r.add_argument("--cameras", help="comma-separated camera ids (default: all enabled)")
    r.add_argument("--show", nargs="?", const="", metavar="CAMS",
                   help="open a window with live detections (optionally only these comma-separated cameras)")
    r.add_argument("--api", action="store_true", help="also serve the HTTP API (FR_API_HOST:FR_API_PORT)")

    e = sub.add_parser("enroll", help="enrol faces")
    e.add_argument("--dir", default=str(ENROLLMENT_DIR), help="folder with one sub-folder per person")
    e.add_argument("--name", help="person name (enrol specific images)")
    e.add_argument("--images", nargs="*", help="image files for --name")
    e.add_argument("--external-id", help="optional external/employee id")
    e.add_argument("--replace", action="store_true", help="replace existing embeddings")

    ps = sub.add_parser("persons", help="list persons")
    ps.add_argument("--all", action="store_true", help="include inactive")

    d = sub.add_parser("delete-person", help="delete a person and their embeddings")
    d.add_argument("person_id", type=int)

    x = sub.add_parser("export", help="export events to CSV")
    x.add_argument("--since")
    x.add_argument("--until")
    x.add_argument("--camera")
    x.add_argument("--known-only", action="store_true")
    x.add_argument("--out")

    c = sub.add_parser("counts", help="person-count totals per camera (or --csv for every row)")
    c.add_argument("--since")
    c.add_argument("--until")
    c.add_argument("--camera")
    c.add_argument("--csv", nargs="?", const="", help="export rows to CSV (optional output path)")

    sub.add_parser("info", help="show model/environment info")
    sub.add_parser("cameras", help="list the cameras + use cases the camera source (yaml or SP) returns")

    sub.add_parser("db-check", help="test the database connection and create tables")

    a = sub.add_parser("api", help="HTTP API for the frontend (docs at /docs)")
    a.add_argument("--host", help="bind address (default FR_API_HOST)")
    a.add_argument("--port", type=int, help="port (default FR_API_PORT)")

    ds = sub.add_parser("db-setup", help="create camera_config + stored procedures in PostgreSQL")
    ds.add_argument("--force", action="store_true", help="re-apply the SQL files even if the objects exist")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings(Path(args.env) if args.env else None,
                             Path(args.cameras_file) if args.cameras_file else None)
    setup_logging(settings.logging_config, settings.log_level)
    commands = {
        None: cmd_run, "run": cmd_run, "enroll": cmd_enroll, "persons": cmd_persons,
        "delete-person": cmd_delete_person, "export": cmd_export, "counts": cmd_counts, "info": cmd_info,
        "cameras": cmd_cameras,
        "db-check": cmd_db_check, "api": cmd_api, "db-setup": cmd_db_setup,
    }
    if args.command is None:
        args.cameras = None
        args.show = None
        args.api = False
    return commands[args.command](settings, args)


if __name__ == "__main__":
    sys.exit(main())
