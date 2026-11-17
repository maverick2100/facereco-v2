import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import CameraConfig  # noqa: E402
from database.camera_repository import CameraRepository  # noqa: E402
from database.connection import Database  # noqa: E402
from database.person_repository import PersonRepository  # noqa: E402
from schema_sql import create_tables  # noqa: E402


@pytest.fixture()
def db(tmp_path):
    d = Database(tmp_path / "t.db")
    create_tables(d)
    yield d
    d.close()


def test_camera_url_is_masked(db):
    repo = CameraRepository(db)
    repo.sync([CameraConfig(camera_id="c1", name="Door", url="rtsp://admin:secret@10.0.0.1/stream")])
    row = repo.get("c1")
    assert "secret" not in row["url_masked"] and "10.0.0.1" in row["url_masked"]


def test_person_roundtrip(db):
    persons = PersonRepository(db)
    pid = persons.get_or_create("Alice")
    assert persons.get_or_create("Alice") == pid
    assert persons.delete(pid)


def test_no_events_or_embeddings_tables(db):
    names = {r["name"] for r in db.fetchall("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert not names & {"events", "embeddings"}


def test_memory_db_shared_across_threads():
    import threading

    d = Database(":memory:")
    create_tables(d)
    repo = PersonRepository(d)
    t = threading.Thread(target=lambda: repo.create("Bob"))
    t.start()
    t.join()
    assert repo.get_by_name("Bob") is not None


class _RecordingPg:
    """Stands in for PostgreSQL: records the SQL the enrolment sends to tbl_offenders."""

    backend = "postgres"

    def __init__(self):
        self.calls = []

    def fetchone(self, sql, params=()):
        self.calls.append((sql, params))
        if sql.startswith("INSERT"):
            return {"log_id": 9} if "search_logs" in sql else {"offender_id": 41}
        return None

    def execute(self, sql, params=()):
        self.calls.append((sql, params))


def test_enrolment_inserts_full_tbl_offenders_row(monkeypatch, tmp_path):
    import numpy as np

    from services.enrollment_service import OFFENDER_LOGIN_ID, EnrollmentService

    import re

    monkeypatch.setenv("FR_OFFENDER_IMAGE_BASE_URL", "https://faces.example.com/")
    monkeypatch.setenv("FR_OFFENDER_IMAGE_DIR", str(tmp_path / "offender"))
    pg = _RecordingPg()
    svc = EnrollmentService(inference=None, db=pg)
    vec = np.arange(512, dtype=np.float32)
    thumb = np.zeros((40, 30, 3), np.uint8)
    assert svc._store_offender("Alice", vec, thumb) == 41
    sql, params = pg.calls[-1]
    assert '"public"."tbl_offenders"' in sql
    for col in ("offender_name", "embedding", "loginid", "imagepath", "is_active",
                "created_by", "updated_by", "updated_date"):
        assert col in sql
    url = params[3]
    m = re.fullmatch(r"https://faces\.example\.com/images/offender/(\d{13})_thumb\.jpg", url)
    assert m, url
    assert (tmp_path / "offender" / f"{m.group(1)}_thumb.jpg").exists()     # thumbnail saved
    assert params[0] == "Alice" and params[2] == OFFENDER_LOGIN_ID
    assert params[4:] == (OFFENDER_LOGIN_ID, OFFENDER_LOGIN_ID)
    svc._store_offender("Alice", vec)
    assert pg.calls[-1][1][3] != url                                        # unique per photo
    assert len(params[1]) == 512 and params[1][3] == 3.0


def test_search_log_row_has_all_tbl_offender_search_logs_columns():
    from datetime import datetime

    import numpy as np

    from database.offender_repository import OffenderRepository

    pg = _RecordingPg()
    log_id = OffenderRepository(pg).log_search(
        device_id=27, gate_id=11, offender_id=101, embedding=np.ones(512, np.float32), frameno=5321,
        detected_imagepath="https://x/images/detected/1_frame.jpg",
        detected_face="https://x/images/detected/1_face.jpg", match_percentage=78.4321,
        time_taken_ms=142.37, capture_time=datetime(2026, 11, 17, 14, 45, 12))
    assert log_id == 9
    sql, params = pg.calls[-1]
    assert '"public"."tbl_offender_search_logs"' in sql and "RETURNING log_id" in sql
    for col in ("device_id", "gate_id", "offender_id", "offender_embedding", "frameno", "detected_imagepath",
                "detected_face", "match_percentage", "time_taken", "capture_time", "is_active",
                "created_date", "updated_date"):
        assert col in sql
    assert params[:3] == (27, 11, 101) and len(params[3]) == 512 and params[4] == 5321
    assert params[7:9] == (78.43, 142.4)
