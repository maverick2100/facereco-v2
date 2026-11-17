"""Repository queries on every database backend.

SQLite always runs. PostgreSQL runs when FR_TEST_DATABASE_URL points at a scratch database
(its tables are dropped and recreated), e.g.

    docker run -d --rm -p 55432:5432 -e POSTGRES_PASSWORD=pw postgres:16-alpine
    FR_TEST_DATABASE_URL=postgresql://postgres:pw@localhost:55432/postgres pytest tests/test_backends.py
"""
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import CameraConfig, StorageSettings  # noqa: E402
from database.camera_repository import CameraRepository  # noqa: E402
from database.connection import Database, mask_dsn  # noqa: E402
from database.person_count_repository import PersonCountRepository  # noqa: E402
from database.person_repository import PersonRepository  # noqa: E402
from schema_sql import create_tables  # noqa: E402

PG_URL = os.environ.get("FR_TEST_DATABASE_URL", "")
TABLES = ["person_counts", "person_crossings", "persons", "cameras"]


@pytest.fixture(params=["sqlite", "postgres"])
def db(request, tmp_path):
    if request.param == "sqlite":
        d = Database(tmp_path / "t.db")
    else:
        if not PG_URL:
            pytest.skip("set FR_TEST_DATABASE_URL to test PostgreSQL")
        from database.connection import PostgresDatabase

        d = PostgresDatabase(PG_URL, max_size=4)
        d.executescript("DROP TABLE IF EXISTS " + ", ".join(TABLES) + " CASCADE")
    create_tables(d)
    yield d
    d.close()


def test_repositories_roundtrip(db):
    cams = CameraRepository(db)
    cams.sync([CameraConfig(camera_id="c1", name="Door", url="rtsp://admin:secret@10.0.0.1/s")])
    cams.sync([CameraConfig(camera_id="c1", name="Front door", url="rtsp://admin:secret@10.0.0.1/s")])
    assert cams.get("c1")["name"] == "Front door" and "secret" not in cams.get("c1")["url_masked"]

    persons = PersonRepository(db)
    a = persons.get_or_create("Alice")
    b = persons.create("Bob", external_id="E2", metadata={"team": "x"})
    assert a != b and persons.get_or_create("Alice") == a
    assert persons.get(b)["metadata"] == {"team": "x"}

    now = time.time()
    counts = PersonCountRepository(db)
    counts.insert_many([("c1", now - 120, now - 60, 100, 2, 0, 3, 1.5, 4, 3, 1, 2, now - 60),
                        ("c1", now - 60, now, 100, 1, 0, 2, 1.0, 2, 0, 2, 0, now)])
    counts.insert_crossings([("c1", 5, "in", now - 70, 640.0, 600.0), ("c1", 5, "out", now - 10, 300.0, 600.0)])
    assert [c["direction"] for c in counts.crossings()] == ["out", "in"]
    t = counts.totals()
    assert t[0]["total_persons"] == 6 and t[0]["peak_count"] == 3 and t[0]["camera_name"] == "Front door"
    assert (t[0]["in_count"], t[0]["out_count"]) == (3, 3)

    assert counts.delete_older_than(now - 30) == 1
    assert persons.delete(a)


def test_storage_settings_conninfo():
    pytest.importorskip("psycopg")
    assert StorageSettings().backend == "sqlite" and StorageSettings().postgres_conninfo() == ""
    s = StorageSettings(database_url="postgresql://u:old@db.local:5432/fr", pg_password="p@ss:w/rd #1")
    info = s.postgres_conninfo()
    from psycopg.conninfo import conninfo_to_dict

    d = conninfo_to_dict(info)
    assert d["password"] == "p@ss:w/rd #1" and d["host"] == "db.local" and d["dbname"] == "fr"
    s = StorageSettings(pg_host="10.0.0.5", pg_user="fr", pg_password="x", pg_database="faces", pg_sslmode="require")
    d = conninfo_to_dict(s.postgres_conninfo())
    assert d == {"host": "10.0.0.5", "port": "5432", "user": "fr", "password": "x", "dbname": "faces",
                 "sslmode": "require"}


def test_mask_dsn():
    assert mask_dsn("postgresql://u:secret@h:5432/db") == "postgresql://u:***@h:5432/db"
    assert "secret" not in mask_dsn("host=h user=u password=secret dbname=db")
    assert mask_dsn("postgresql://h/db") == "postgresql://h/db"
