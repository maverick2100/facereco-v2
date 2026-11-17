#!/usr/bin/env python3
"""End-to-end test: your SP -> supervisor -> GStreamer -> models -> database, on PostgreSQL.

It adds two temporary cameras through the SPs, runs the real `main.py run` supervisor on them, and
checks every step by looking at the database:

  1. database connection + SPs present
  2. SP says: e2e_face = face, e2e_person = person (with an IN/OUT line)
     -> pipelines start automatically
     -> person_counts rows, IN and OUT crossings reach the DB
  3. SP change: e2e_face switched to person  -> NOTIFY -> restarted within seconds, counts arrive
  4. SP change: e2e_person disabled          -> stops running
  5. clean-up: the e2e cameras are deleted (results too, unless --keep)

Only the e2e_* cameras run; other cameras in your camera_config are not touched or started.

    # throwaway PostgreSQL, fake models (no .mlpackage needed), built-in test video:
    FR_DATABASE_URL=postgresql://user:pass@127.0.0.1:55432/db python scripts/e2e_test.py --apply-sql --fake-models

    # real models, your video files (a clip with people walking across the frame for --person-url):
    python scripts/e2e_test.py --face-url file:///path/faces.mp4 --person-url file:///path/people.mp4

--apply-sql runs database/sql/camera_config_postgres.sql first (it replaces sp_get_cameras /
sp_upsert_camera / sp_delete_camera, so don't use it on a database that has your own SPs with those names).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FACE, PERSON = "e2e_face", "e2e_person"
LINE = "[0.5, 0, 0.5, 1]"                     # vertical line down the middle


# --------------------------------------------------------------------------- fake models
def install_fake_models() -> None:
    """Replace Core ML loading with in-process fakes: one face at a fixed spot, one person walking
    left <-> right across the frame (so it crosses the line: IN, then OUT, every 6 s)."""
    import numpy as np

    import inference.inference_manager as im
    import person_detection.person_count_service as pcs
    from inference.coreml_loader import FeatureSpec
    from tests.fakes import FakeArcFaceModel, FakeSCRFDModel

    class WalkingPersonYOLO:
        def __init__(self) -> None:
            self.inputs = [FeatureSpec("images", "multiarray", shape=(1, 3, 640, 640), dtype=np.float32)]
            self.outputs = []

        def info(self) -> dict:
            return {"model": "fake-yolo"}

        def predict(self, feeds):
            t = time.monotonic() % 6.0
            frac = t / 3.0 if t < 3.0 else (6.0 - t) / 3.0
            a = np.zeros((1, 10, 85), np.float32)
            a[0, 0, :4] = [80 + 480 * frac, 320, 100, 200]   # cx, cy, w, h in 640x640 model space
            a[0, 0, 4], a[0, 0, 5] = 0.9, 0.95
            return {"output": a}

        def predict_batch(self, feeds_list):
            return [self.predict(f) for f in feeds_list]

    def fake_coreml(path, compute_units=None, cache_dir=None, name=""):
        m = FakeSCRFDModel() if name == "scrfd" else FakeArcFaceModel()
        m.info = lambda: {"model": f"fake-{name}"}
        return m

    im.CoreMLModel = fake_coreml
    pcs.CoreMLModel = lambda *a, **k: WalkingPersonYOLO()
    # synthetic frames are flat: let the fake face through the quality gate
    os.environ.update({"FR_BLUR_THRESHOLD": "0", "FR_MIN_QUALITY": "0", "FR_MAX_YAW": "10"})


# --------------------------------------------------------------------------- checks
class Report:
    def __init__(self) -> None:
        self.rows = []

    def add(self, status: str, name: str, detail: str = "") -> None:
        self.rows.append((status, name, detail))
        print(f"  [{status}] {name}" + (f" - {detail}" if detail else ""), flush=True)

    def ok(self, cond: bool, name: str, detail: str = "") -> bool:
        self.add("PASS" if cond else "FAIL", name, detail)
        return cond

    @property
    def failed(self) -> int:
        return sum(r[0] == "FAIL" for r in self.rows)


def wait_for(fn, timeout: float, every: float = 0.5):
    deadline = time.monotonic() + timeout
    while True:
        v = fn()
        if v or time.monotonic() >= deadline:
            return v
        time.sleep(every)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fake-models", action="store_true", help="no .mlpackage needed (tests everything else)")
    ap.add_argument("--apply-sql", action="store_true", help="run database/sql/camera_config_postgres.sql first")
    ap.add_argument("--face-url", default="test://ball")
    ap.add_argument("--person-url", default="test://ball")
    ap.add_argument("--timeout", type=float, default=30.0, help="seconds to wait for each check")
    ap.add_argument("--keep", action="store_true", help="keep the e2e result rows in the DB")
    ap.add_argument("--skip-face", action="store_true",
                    help="person counting only (e.g. while the SCRFD model is missing): e2e_face is added later "
                         "as a second person camera instead")
    args = ap.parse_args()

    # Settings for the run (environment wins over .env). Fast flush so counts show up in seconds;
    # slow polling so a quick restart proves NOTIFY works.
    os.environ.update({"FR_CAMERA_SOURCE": "db", "FR_PERSON_FLUSH_S": "5", "FR_CAMERA_POLL_S": "60",
                       "FR_HEALTH_PORT": "0", "FR_RECORD_UNKNOWN": "true", "FR_POWERMETRICS": "false"})
    if args.fake_models:
        install_fake_models()

    import main as app_main
    from config.settings import load_settings
    from database.connection import open_database
    from services.camera_source import CameraSource

    settings = load_settings()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("main").setLevel(logging.INFO)
    logging.getLogger("services.camera_source").setLevel(logging.INFO)
    rep = Report()
    real_video = not (args.face_url.startswith("test://") and args.person_url.startswith("test://"))
    detections_checked = args.fake_models or real_video

    print("1. Database")
    is_pg = settings.storage.backend == "postgres"
    if not rep.ok(is_pg, "PostgreSQL configured",
                  "" if is_pg else "set FR_DATABASE_URL or FR_PG_* (in .env or the environment)"):
        return 1
    try:
        db = open_database(settings.storage)
    except Exception as exc:
        rep.ok(False, "connect", str(exc))
        return 1
    rep.ok(True, "connect", db.description)
    if args.apply_sql:
        db.executescript((ROOT / "database/sql/camera_config_postgres.sql").read_text())
        rep.ok(True, "camera_config SQL applied")
    have_sp = db.fetchone("SELECT COUNT(*) AS n FROM pg_proc WHERE proname IN "
                          "('sp_get_cameras', 'sp_upsert_camera', 'sp_delete_camera')")["n"]
    if not rep.ok(have_sp >= 3, "SPs sp_get_cameras / sp_upsert_camera / sp_delete_camera exist",
                  "" if have_sp >= 3 else "run with --apply-sql, or create them"):
        return 1

    def upsert(cid, url, face, person, line=None, enabled=True):
        db.execute("SELECT sp_upsert_camera(p_camera_id => ?, p_name => ?, p_url => ?, p_run_face => ?, "
                   "p_run_person => ?, p_location => 'e2e', p_enabled => ?, p_roi => ?::jsonb)",
                   (cid, cid, url, face, person, enabled, line))

    def rows(table, cid, since, extra=""):
        return db.fetchone(f"SELECT COUNT(*) AS n FROM {table} WHERE camera_id = ? AND ts >= ?{extra}",
                           (cid, since))["n"]

    # Record every pipeline (re)start the supervisor makes.
    starts = []

    class RecordingApp(app_main.FaceRecognitionApp):
        def start(self):
            super().start()
            starts.append((time.monotonic(), {c.camera_id: tuple(c.tasks) for c in self.cameras}))

    app_main.FaceRecognitionApp = RecordingApp
    source = CameraSource(settings.camera_source, settings.storage)
    sup = app_main.Supervisor(settings, source, None, only={FACE, PERSON})

    def started_with(expected, after):
        return any(t >= after and cams == expected for t, cams in starts)

    def scenario():
        try:
            t0, since = time.monotonic(), time.time()
            print("\n2. SP -> use cases start automatically")
            if args.skip_face:
                upsert(PERSON, args.person_url, False, True, LINE)
                rep.ok(wait_for(lambda: started_with({PERSON: ("person",)}, t0), args.timeout),
                       "supervisor started e2e_person=person from sp_get_cameras()")
            else:
                upsert(FACE, args.face_url, True, False)
                upsert(PERSON, args.person_url, False, True, LINE)
                rep.ok(wait_for(lambda: started_with({FACE: ("face",), PERSON: ("person",)}, t0), args.timeout),
                       "supervisor started e2e_face=face, e2e_person=person from sp_get_cameras()")
            if detections_checked:
                rep.ok(wait_for(lambda: rows("person_counts", PERSON, since), args.timeout) > 0,
                       "person_counts rows written")
                rep.ok(wait_for(lambda: rows("person_crossings", PERSON, since, " AND direction = 'in'"),
                                args.timeout) > 0, "IN crossing written (person_crossings)")
                rep.ok(wait_for(lambda: rows("person_crossings", PERSON, since, " AND direction = 'out'"),
                                args.timeout) > 0, "OUT crossing written (person_crossings)")
                r = db.fetchone("SELECT SUM(in_count) AS i, SUM(out_count) AS o FROM person_counts "
                                "WHERE camera_id = ? AND ts >= ?", (PERSON, since))
                rep.ok(True, "person_counts totals", f"in={r['i']} out={r['o']}")
            else:
                rep.ok(wait_for(lambda: rows("person_counts", PERSON, since), args.timeout) > 0,
                       "person_counts rows written (frames flowing)")
                rep.add("SKIP", "IN-OUT crossings",
                        "test video has no faces or people: use --fake-models, or --face-url/--person-url")

            t1, since = time.monotonic(), time.time()
            if args.skip_face:
                print("\n3. SP change: new camera e2e_face added (person)")
                upsert(FACE, args.person_url, False, True, LINE)
            else:
                print("\n3. SP change: e2e_face switched face -> person")
                upsert(FACE, args.face_url, False, True)
            ok = wait_for(lambda: started_with({FACE: ("person",), PERSON: ("person",)}, t1), args.timeout)
            took = next((t - t1 for t, c in starts if t >= t1), None)
            rep.ok(bool(ok), "pipelines restarted with the new use case",
                   f"{took:.1f}s after the SP call" if took is not None else "")
            rep.ok(took is not None and took < 20, "change arrived via NOTIFY (not the 60 s poll)")
            rep.ok(wait_for(lambda: rows("person_counts", FACE, since), args.timeout) > 0,
                   "e2e_face now writes person_counts")

            print("\n4. SP change: e2e_person disabled")
            t2 = time.monotonic()
            upsert(PERSON, args.person_url, False, True, LINE, enabled=False)
            rep.ok(wait_for(lambda: started_with({FACE: ("person",)}, t2), args.timeout),
                   "e2e_person no longer running")
            print(f"\n   total scenario time {time.monotonic() - t0:.0f}s")
        except Exception as exc:
            rep.ok(False, "scenario crashed", repr(exc))
        finally:
            sup._stop.set()
            sup._cycle.set()

    threading.Thread(target=scenario, name="e2e-scenario", daemon=True).start()
    try:
        sup.run_forever()               # main thread: real signal handling, like `main.py run`
    finally:
        print("\n5. Clean-up")
        for cid in (FACE, PERSON):
            db.execute("SELECT sp_delete_camera(?)", (cid,))
        if not args.keep:
            for table in ("person_counts", "person_crossings"):
                db.execute(f"DELETE FROM {table} WHERE camera_id IN (?, ?)", (FACE, PERSON))
            db.execute("DELETE FROM cameras WHERE id IN (?, ?)", (FACE, PERSON))
        rep.ok(db.fetchone("SELECT COUNT(*) AS n FROM camera_config WHERE camera_id IN (?, ?)",
                           (FACE, PERSON))["n"] == 0, "e2e cameras removed")
        db.close()

    print(f"\nRESULT: {'FAILED' if rep.failed else 'PASSED'} - "
          f"{sum(r[0] == 'PASS' for r in rep.rows)} passed, {rep.failed} failed, "
          f"{sum(r[0] == 'SKIP' for r in rep.rows)} skipped"
          + ("   (fake models: real SCRFD/ArcFace/YOLOv7 not tested)" if args.fake_models else ""))
    return 1 if rep.failed else 0


if __name__ == "__main__":
    sys.exit(main())
