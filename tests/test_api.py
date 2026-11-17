"""HTTP API: cameras in (both through the API and by calling the SPs directly), dashboard out.

The URL helpers always run. Everything else needs a scratch PostgreSQL (its public schema is wiped):

    docker run -d --rm -p 55432:5432 -e POSTGRES_PASSWORD=pw postgres:16-alpine
    FR_TEST_DATABASE_URL=postgresql://postgres:pw@localhost:55432/postgres pytest tests/test_api.py
"""
import os
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import HTTPException  # noqa: E402

from api.server import create_app, mask_password, restore_password  # noqa: E402
from config.settings import CameraSourceSettings, Settings, StorageSettings  # noqa: E402
from services.camera_source import CameraSource  # noqa: E402

PG_URL = os.environ.get("FR_TEST_DATABASE_URL", "")
GATE = {"camera_id": "gate1", "name": "Main Gate", "url": "rtsp://admin:s3cr%40t@10.0.0.5:554/stream1",
        "run_person": True, "roi": [0.5, 0, 0.5, 1]}


# ------------------------------------------------------------------------------------------- URL helpers
def test_mask_and_restore_password():
    url = "rtsp://admin:s3cr%40t@10.0.0.5:554/stream1"
    masked = mask_password(url)
    assert masked == "rtsp://admin:***@10.0.0.5:554/stream1"
    assert restore_password(masked, url) == url                       # round trip keeps the real one
    assert restore_password("rtsp://admin:new@10.0.0.5/s", url) == "rtsp://admin:new@10.0.0.5/s"
    assert mask_password("rtsp://10.0.0.5/s") == "rtsp://10.0.0.5/s"   # nothing to hide
    with pytest.raises(HTTPException):
        restore_password(masked, None)                                 # new camera can't send ***


# ------------------------------------------------------------------------------------------- PostgreSQL
@pytest.fixture
def pg():
    if not PG_URL:
        pytest.skip("set FR_TEST_DATABASE_URL to test the API against PostgreSQL")
    from database.connection import PostgresDatabase

    db = PostgresDatabase(PG_URL, min_size=1, max_size=4)
    db.executescript("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
    from database.sql_setup import CAMERA_SQL
    from schema_sql import create_tables

    create_tables(db)                       # facereco itself never creates tables
    db.executescript(CAMERA_SQL.read_text(encoding="utf-8"))
    yield db
    db.close()


def _settings(api_key=""):
    s = Settings()
    s.storage = StorageSettings(database_url=PG_URL)
    s.camera_source = CameraSourceSettings(source="db")
    s.monitoring.health_port = 0          # facereco not running in tests
    s.api.api_key = api_key
    return s


@pytest.fixture
def client(pg):
    from fastapi.testclient import TestClient

    with TestClient(create_app(_settings(), db=pg)) as c:   # start-up installs the SPs
        yield c


def _sp_cameras(pg):
    return {r["camera_id"]: r["tasks"] for r in pg.fetchall("SELECT * FROM sp_get_cameras()")}


def test_startup_creates_table_and_sps(client, pg):
    names = {r["proname"] for r in pg.fetchall("SELECT proname FROM pg_proc WHERE proname LIKE 'sp_%'")}
    assert {"sp_get_cameras", "sp_upsert_camera", "sp_delete_camera", "sp_dash_summary"} <= names


def test_save_camera_starts_use_case(client, pg):
    r = client.post("/cameras", json=GATE)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["url"] == "rtsp://admin:***@10.0.0.5:554/stream1"   # password never sent back
    assert body["use_cases"] == ["person"] and body["state"] == "starting"
    assert _sp_cameras(pg) == {"gate1": "person"}                     # what facereco reads

    # the real reader facereco uses, against the same table
    cams = CameraSource(_settings().camera_source, _settings().storage).load()
    assert [(c.camera_id, c.tasks, c.url) for c in cams] == [("gate1", ["person"], GATE["url"])]
    assert cams[0].roi == [640.0, 0.0, 640.0, 720.0]                  # fractions -> pixels (1280x720)


def test_patch_start_stop_delete(client, pg):
    client.post("/cameras", json=GATE)
    r = client.patch("/cameras/gate1", json={"run_face": True})
    assert r.status_code == 200 and r.json()["use_cases"] == ["face", "person"]
    assert _sp_cameras(pg) == {"gate1": "face,person"}

    assert client.post("/cameras/gate1/stop").json()["state"] == "stopped"
    assert _sp_cameras(pg) == {}                                      # facereco stops it
    assert client.post("/cameras/gate1/start").json()["enabled"] is True
    assert _sp_cameras(pg) == {"gate1": "face,person"}

    client.patch("/cameras/gate1", json={"run_face": False, "run_person": False})
    assert _sp_cameras(pg) == {}                                      # no use case = not running
    assert client.get("/cameras/gate1").json()["state"] == "stopped"

    assert client.delete("/cameras/gate1").status_code == 204
    assert client.get("/cameras/gate1").status_code == 404
    assert client.delete("/cameras/gate1").status_code == 404
    assert client.patch("/cameras/nope", json={"enabled": True}).status_code == 404


def test_masked_password_round_trip_keeps_stored_password(client, pg):
    client.post("/cameras", json=GATE)
    shown = client.get("/cameras/gate1").json()
    r = client.put("/cameras/gate1", json={"name": "Gate (renamed)", "url": shown["url"], "run_person": True})
    assert r.status_code == 200
    row = pg.fetchone("SELECT name, url FROM camera_config WHERE camera_id = 'gate1'")
    assert row["name"] == "Gate (renamed)" and row["url"] == GATE["url"]
    assert client.put("/cameras/new1", json={"name": "x", "url": shown["url"]}).status_code == 422


@pytest.mark.parametrize("change", [
    {"url": "ftp://10.0.0.5/x"},
    {"url": "not a url"},
    {"count_zone": [[0.5, 0], [1, 0], [1, 1]]},           # polygons are not accepted
    {"roi": [10, 10, 5, 5]},
    {"fps": 0},
    {"codec": "mpeg2"},
    {"camera_id": "bad id!"},
    {"unknown_field": 1},
])
def test_rejects_bad_input(client, pg, change):
    r = client.post("/cameras", json={**GATE, **change})
    assert r.status_code == 422, r.text
    assert _sp_cameras(pg) == {}


def test_backend_calling_sp_directly_is_seen_by_api(client, pg):
    pg.execute("SELECT sp_upsert_camera('hall2', 'Hall', 'rtsp://10.0.0.6/s', true, false)")
    cams = {c["camera_id"]: c for c in client.get("/cameras").json()["cameras"]}
    assert cams["hall2"]["use_cases"] == ["face"]
    assert client.post("/cameras/hall2/stop").status_code == 200       # and the API can manage it
    assert _sp_cameras(pg) == {}


def test_api_change_wakes_facereco_via_notify(client, pg):
    src = CameraSource(_settings().camera_source, _settings().storage)
    stop = threading.Event()
    assert src.wait_for_change(0.5, stop) is False                     # opens LISTEN, nothing yet
    got = {}
    t = threading.Thread(target=lambda: got.setdefault("notified", src.wait_for_change(10, stop)))
    t.start()
    time.sleep(0.3)
    t0 = time.monotonic()
    client.post("/cameras", json=GATE)
    t.join(12)
    src.close()
    assert got.get("notified") is True
    assert time.monotonic() - t0 < 2.0                                 # instant, not the 30 s poll


def test_dashboard_endpoints(client, pg):
    now = time.time()
    pg.executemany("INSERT INTO person_crossings (camera_id, track_id, direction, ts) VALUES (?, ?, ?, ?)",
                   [("gate1", 1, "in", now - 5), ("gate1", 2, "in", now - 4), ("gate1", 1, "out", now - 3)])
    s = client.get("/dashboard/summary", params={"tz": "UTC"}).json()
    assert (s["in_count"], s["out_count"], s["inside_now"]) == (2, 1, 1)
    hourly = client.get("/dashboard/hourly", params={"tz": "UTC"}).json()
    assert len(hourly) == 24 and sum(h["in_count"] for h in hourly) == 2
    assert client.get("/dashboard/summary", params={"cameras": "other", "tz": "UTC"}).json()["in_count"] == 0
    assert len(client.get("/dashboard/daily").json()) == 7
    assert client.get("/dashboard/daily", params={"from": "2026-01-10", "to": "2026-01-01"}).status_code == 422
    assert client.get("/dashboard/cameras").status_code == 200
    r = client.get("/dashboard/summary", params={"tz": "Mars/Base"})
    assert r.status_code == 400, r.text


def test_api_key(pg):
    from fastapi.testclient import TestClient

    with TestClient(create_app(_settings(api_key="k1"), db=pg)) as c:
        assert c.get("/health").status_code == 200                     # open for load balancers
        assert c.get("/cameras").status_code == 401
        assert c.get("/cameras", headers={"X-API-Key": "wrong"}).status_code == 401
        assert c.get("/cameras", headers={"X-API-Key": "k1"}).status_code == 200
