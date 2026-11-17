import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main  # noqa: E402
from config.settings import CameraSourceSettings, Settings  # noqa: E402
from services.camera_source import (CameraSource, describe_change, fingerprint, parse_tasks,  # noqa: E402
                                    rows_to_cameras)

DEFAULTS = {"fps": 4, "width": 1280, "height": 720, "tasks": ["face"]}


def test_rows_with_tasks_text_flags_and_aliases():
    rows = [
        {"camera_id": "cam01", "name": "Gate", "url": "rtsp://a", "tasks": "face", "enabled": True},
        {"id": 2, "camera_name": "Hall", "rtsp_url": "rtsp://b", "use_case": "Person Detection",
         "roi": "[0.5, 0, 0.5, 1]", "fps": None, "created_by": "admin"},
        {"camera_id": "cam03", "url": "rtsp://c", "run_face": 1, "run_person": "true"},
        {"camera_id": "cam04", "url": "rtsp://d", "tasks": ["face", "person"], "is_active": "false"},  # off
        {"camera_id": "cam05", "url": "rtsp://e", "face": False, "person": False},                  # no use case
        {"camera_id": "cam06", "url": "rtsp://f", "tasks": "both", "roi": "[0, 0, 640, 360]"},
    ]
    cams = {c.camera_id: c for c in rows_to_cameras(rows, DEFAULTS)}
    assert sorted(cams) == ["2", "cam01", "cam03", "cam06"]
    assert cams["cam01"].tasks == ["face"]
    assert cams["2"].tasks == ["person"] and cams["2"].name == "Hall" and cams["2"].url == "rtsp://b"
    assert cams["2"].fps == 4                                       # NULL -> yaml default
    assert cams["2"].roi == [640.0, 0.0, 640.0, 720.0]               # fractions -> pixels
    assert cams["cam03"].tasks == ["face", "person"]
    assert cams["cam06"].tasks == ["face", "person"] and cams["cam06"].roi == [0, 0, 640, 360]


def test_bad_row_is_skipped_not_fatal():
    rows = [{"camera_id": "ok", "url": "x", "tasks": "person"},
            {"camera_id": "bad", "url": "x", "tasks": "cars"},
            {"name": "no id"},
            {"camera_id": "ok", "url": "y", "tasks": "face"}]           # duplicate id
    assert [(c.camera_id, c.tasks) for c in rows_to_cameras(rows)] == [("ok", ["person"])]


def test_parse_tasks():
    assert parse_tasks("face, person") == ["face", "person"]
    assert parse_tasks("Face Recognition|person_count") == ["face", "person"]
    with pytest.raises(ValueError):
        parse_tasks("vehicle")


def test_change_detection():
    a = rows_to_cameras([{"camera_id": "c1", "url": "x", "tasks": "face"}])
    b = rows_to_cameras([{"camera_id": "c1", "url": "x", "tasks": "person"},
                         {"camera_id": "c2", "url": "x", "tasks": "face"}])
    assert fingerprint(a) == fingerprint(rows_to_cameras([{"camera_id": "c1", "url": "x", "tasks": "face"}]))
    assert fingerprint(a) != fingerprint(b)
    assert describe_change(a, b) == "added: c2; changed: c1"


def test_db_source_runs_the_sql(tmp_path):
    from config.settings import StorageSettings
    from database.connection import Database

    db_path = tmp_path / "cams.db"
    db = Database(db_path)
    db.executescript("CREATE TABLE my_cams (cam TEXT, link TEXT, face INTEGER, person INTEGER);"
                     "INSERT INTO my_cams VALUES ('c1', 'rtsp://1', 1, 0), ('c2', 'rtsp://2', 0, 1);")
    db.close()
    src = CameraSource(CameraSourceSettings(source="db", defaults=DEFAULTS,
                                            sql="SELECT cam AS camera_id, link AS url, face, person FROM my_cams"),
                       StorageSettings(db_path=db_path))
    cams = src.load()
    src.close()
    assert [(c.camera_id, c.tasks) for c in cams] == [("c1", ["face"]), ("c2", ["person"])]


class _FakeSource:
    description = "fake"

    def __init__(self, versions):
        self.versions = versions          # list of camera lists returned on successive loads
        self.i = 0

    def load(self):
        v = self.versions[min(self.i, len(self.versions) - 1)]
        self.i += 1
        return v

    def wait_for_change(self, timeout, stop):
        stop.wait(0.05)
        return False

    def close(self):
        pass


class _FakeApp:
    runs = []

    def __init__(self, settings, cams, show):
        self.cams = cams
        self.live = None

    def start(self):
        _FakeApp.runs.append(("start", [(c.camera_id, tuple(c.tasks)) for c in self.cams]))

    def stop(self):
        _FakeApp.runs.append(("stop",))


def test_supervisor_restarts_pipelines_when_sp_result_changes(monkeypatch):
    v1 = rows_to_cameras([{"camera_id": "c1", "url": "x", "tasks": "face"}])
    v2 = rows_to_cameras([{"camera_id": "c1", "url": "x", "tasks": "face"},
                          {"camera_id": "c2", "url": "x", "tasks": "person"}])
    # load #1 = initial; re-checks: unchanged, then the frontend adds c2.
    source = _FakeSource([v1, v1, v2])
    monkeypatch.setattr(main, "FaceRecognitionApp", _FakeApp)
    monkeypatch.setattr(main.signal, "signal", lambda *a: None)
    _FakeApp.runs = []
    sup = main.Supervisor(Settings(camera_source=CameraSourceSettings(poll_s=0.01)), source)
    t = threading.Thread(target=sup.run_forever, daemon=True)
    t.start()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and sum(r[0] == "start" for r in _FakeApp.runs) < 2:
        time.sleep(0.02)
    sup._stop.set()
    sup._cycle.set()
    t.join(timeout=5)
    starts = [r[1] for r in _FakeApp.runs if r[0] == "start"]
    assert starts[0] == [("c1", ("face",))]
    assert starts[1] == [("c1", ("face",)), ("c2", ("person",))]
    assert _FakeApp.runs[1] == ("stop",)          # old pipelines flushed + stopped before the new start
    assert not t.is_alive()
