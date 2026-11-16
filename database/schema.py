"""Database schema + initialisation (SQLite and PostgreSQL)."""
from __future__ import annotations

import logging
import time

from constants import SCHEMA_VERSION
from database.connection import Database

log = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER PRIMARY KEY,
    applied_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS cameras (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    url_masked  TEXT,
    location    TEXT,
    enabled     INTEGER NOT NULL DEFAULT 1,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS persons (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    external_id TEXT UNIQUE,
    metadata    TEXT,
    active      INTEGER NOT NULL DEFAULT 1,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_persons_name ON persons(name);

CREATE TABLE IF NOT EXISTS embeddings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id   INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    vector      BLOB NOT NULL,
    dim         INTEGER NOT NULL,
    model       TEXT,
    quality     REAL,
    source      TEXT,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_embeddings_person ON embeddings(person_id);

CREATE TABLE IF NOT EXISTS events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id     TEXT NOT NULL,
    person_id     INTEGER REFERENCES persons(id) ON DELETE SET NULL,
    label         TEXT NOT NULL,
    similarity    REAL,
    det_score     REAL,
    quality       REAL,
    bbox          TEXT,
    snapshot_path TEXT,
    ts            REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_camera_ts ON events(camera_id, ts);
CREATE INDEX IF NOT EXISTS idx_events_person_ts ON events(person_id, ts);

-- One row per camera per interval (FR_PERSON_FLUSH_S), appended by person_detection/.
CREATE TABLE IF NOT EXISTS person_counts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id     TEXT NOT NULL,
    period_start  REAL NOT NULL,
    period_end    REAL NOT NULL,
    frames        INTEGER NOT NULL,   -- frames analysed in the interval
    current_count INTEGER NOT NULL,   -- people in the last analysed frame
    min_count     INTEGER NOT NULL,
    max_count     INTEGER NOT NULL,   -- peak occupancy in the interval
    avg_count     REAL NOT NULL,
    new_persons   INTEGER NOT NULL,   -- distinct people that appeared (tracked) in the interval
    in_count      INTEGER NOT NULL DEFAULT 0,  -- count_zone: outside -> inside crossings in the interval
    out_count     INTEGER NOT NULL DEFAULT 0,  -- count_zone: inside -> outside crossings
    zone_count    INTEGER NOT NULL DEFAULT 0,  -- people inside the zone at the end of the interval
    ts            REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_person_counts_ts ON person_counts(ts);
CREATE INDEX IF NOT EXISTS idx_person_counts_camera_ts ON person_counts(camera_id, ts);

-- One row per confirmed zone crossing (written every few seconds, in batches).
CREATE TABLE IF NOT EXISTS person_crossings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id   TEXT NOT NULL,
    track_id    INTEGER NOT NULL,
    direction   TEXT NOT NULL CHECK (direction IN ('in', 'out')),
    ts          REAL NOT NULL,
    x           REAL,
    y           REAL
);
CREATE INDEX IF NOT EXISTS idx_person_crossings_camera_ts ON person_crossings(camera_id, ts);
CREATE INDEX IF NOT EXISTS idx_person_crossings_ts ON person_crossings(ts);
"""


# Same tables for PostgreSQL. Timestamps stay epoch seconds, as DOUBLE PRECISION
# (PostgreSQL REAL is only 4 bytes, which would round timestamps to minutes).
PG_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER PRIMARY KEY,
    applied_at  DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS cameras (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    url_masked  TEXT,
    location    TEXT,
    enabled     INTEGER NOT NULL DEFAULT 1,
    created_at  DOUBLE PRECISION NOT NULL,
    updated_at  DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS persons (
    id          BIGSERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    external_id TEXT UNIQUE,
    metadata    TEXT,
    active      INTEGER NOT NULL DEFAULT 1,
    created_at  DOUBLE PRECISION NOT NULL,
    updated_at  DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_persons_name ON persons(name);

CREATE TABLE IF NOT EXISTS embeddings (
    id          BIGSERIAL PRIMARY KEY,
    person_id   BIGINT NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    vector      BYTEA NOT NULL,
    dim         INTEGER NOT NULL,
    model       TEXT,
    quality     DOUBLE PRECISION,
    source      TEXT,
    created_at  DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_embeddings_person ON embeddings(person_id);

CREATE TABLE IF NOT EXISTS events (
    id            BIGSERIAL PRIMARY KEY,
    camera_id     TEXT NOT NULL,
    person_id     BIGINT REFERENCES persons(id) ON DELETE SET NULL,
    label         TEXT NOT NULL,
    similarity    DOUBLE PRECISION,
    det_score     DOUBLE PRECISION,
    quality       DOUBLE PRECISION,
    bbox          TEXT,
    snapshot_path TEXT,
    ts            DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_camera_ts ON events(camera_id, ts);
CREATE INDEX IF NOT EXISTS idx_events_person_ts ON events(person_id, ts);

CREATE TABLE IF NOT EXISTS person_counts (
    id            BIGSERIAL PRIMARY KEY,
    camera_id     TEXT NOT NULL,
    period_start  DOUBLE PRECISION NOT NULL,
    period_end    DOUBLE PRECISION NOT NULL,
    frames        INTEGER NOT NULL,
    current_count INTEGER NOT NULL,
    min_count     INTEGER NOT NULL,
    max_count     INTEGER NOT NULL,
    avg_count     DOUBLE PRECISION NOT NULL,
    new_persons   INTEGER NOT NULL,
    in_count      INTEGER NOT NULL DEFAULT 0,
    out_count     INTEGER NOT NULL DEFAULT 0,
    zone_count    INTEGER NOT NULL DEFAULT 0,
    ts            DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_person_counts_ts ON person_counts(ts);
CREATE INDEX IF NOT EXISTS idx_person_counts_camera_ts ON person_counts(camera_id, ts);

CREATE TABLE IF NOT EXISTS person_crossings (
    id          BIGSERIAL PRIMARY KEY,
    camera_id   TEXT NOT NULL,
    track_id    INTEGER NOT NULL,
    direction   TEXT NOT NULL CHECK (direction IN ('in', 'out')),
    ts          DOUBLE PRECISION NOT NULL,
    x           REAL,
    y           REAL
);
CREATE INDEX IF NOT EXISTS idx_person_crossings_camera_ts ON person_crossings(camera_id, ts);
CREATE INDEX IF NOT EXISTS idx_person_crossings_ts ON person_crossings(ts);
"""

# Columns added after a table first shipped: (table, column, type). CREATE TABLE IF NOT EXISTS
# doesn't touch an existing table, so these are added to older databases on start-up.
_ADDED_COLUMNS = [
    ("person_counts", "in_count", "INTEGER NOT NULL DEFAULT 0"),
    ("person_counts", "out_count", "INTEGER NOT NULL DEFAULT 0"),
    ("person_counts", "zone_count", "INTEGER NOT NULL DEFAULT 0"),
]


def _add_missing_columns(db: Database) -> None:
    pg = getattr(db, "backend", "sqlite") == "postgres"
    for table, col, typ in _ADDED_COLUMNS:
        if pg:
            db.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col} {typ}")
        elif col not in {r["name"] for r in db.fetchall(f"PRAGMA table_info({table})")}:
            db.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
            log.info("Added column %s.%s", table, col)


def init_schema(db: Database) -> None:
    db.executescript(PG_SCHEMA_SQL if getattr(db, "backend", "sqlite") == "postgres" else SCHEMA_SQL)
    if getattr(db, "backend", "sqlite") == "postgres":
        db.executescript(OFFENDERS_PG_SQL)
    _add_missing_columns(db)
    row = db.fetchone("SELECT MAX(version) AS v FROM schema_version")
    current = row["v"] if row and row["v"] is not None else 0
    if current < SCHEMA_VERSION:
        db.execute("INSERT INTO schema_version(version, applied_at) VALUES (?, ?) "
                   "ON CONFLICT(version) DO NOTHING", (SCHEMA_VERSION, time.time()))
        log.info("Database schema at version %d (%s)", SCHEMA_VERSION, db.path)


OFFENDERS_PG_SQL = """
CREATE TABLE IF NOT EXISTS public.tbl_offenders (
    offender_id   SERIAL PRIMARY KEY,
    offender_name VARCHAR NOT NULL,
    embedding     DOUBLE PRECISION[] NOT NULL,
    loginid       INTEGER NOT NULL,
    imagepath     TEXT,
    is_active     BOOLEAN NOT NULL DEFAULT TRUE,
    created_by    INTEGER,
    updated_by    INTEGER,
    updated_date  TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE public.tbl_offenders ADD COLUMN IF NOT EXISTS imagepath TEXT;
ALTER TABLE public.tbl_offenders ADD COLUMN IF NOT EXISTS is_active BOOLEAN NOT NULL DEFAULT TRUE;
ALTER TABLE public.tbl_offenders ADD COLUMN IF NOT EXISTS created_by INTEGER;
ALTER TABLE public.tbl_offenders ADD COLUMN IF NOT EXISTS updated_by INTEGER;
ALTER TABLE public.tbl_offenders ADD COLUMN IF NOT EXISTS updated_date TIMESTAMP WITHOUT TIME ZONE
    NOT NULL DEFAULT CURRENT_TIMESTAMP;

CREATE TABLE IF NOT EXISTS public.offender_sync (
    person_id       BIGINT PRIMARY KEY REFERENCES public.persons(id) ON DELETE CASCADE,
    offender_id     INTEGER NOT NULL UNIQUE,
    synced_at       DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS public.offender_embedding_sync (
    person_id       BIGINT NOT NULL REFERENCES public.persons(id) ON DELETE CASCADE,
    offender_id     INTEGER NOT NULL UNIQUE,
    imagepath       TEXT NOT NULL,
    synced_at       DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (person_id, imagepath)
);

CREATE TABLE IF NOT EXISTS public.offender_search_log (
    logid                   SERIAL PRIMARY KEY,
    camera_id               INTEGER NOT NULL,
    capture_time            VARCHAR NOT NULL,
    matched_offender_id     INTEGER,
    matched_percentage      DOUBLE PRECISION NOT NULL,
    matched_status          VARCHAR NOT NULL
);

CREATE OR REPLACE FUNCTION public.fn_add_offenders(
    p_offender_name CHARACTER VARYING,
    p_embedding DOUBLE PRECISION[],
    p_loginid INTEGER,
    p_imagepath TEXT
)
RETURNS TABLE(p_message CHARACTER VARYING, new_offender_id INTEGER,
              new_offender_name CHARACTER VARYING, new_loginid INTEGER)
LANGUAGE plpgsql
COST 100
VOLATILE
PARALLEL UNSAFE
ROWS 1000
AS $$
BEGIN
    INSERT INTO public.tbl_offenders
        (offender_name, embedding, loginid, imagepath, is_active, created_by, updated_by, updated_date)
    VALUES (p_offender_name, p_embedding, p_loginid, p_imagepath, TRUE, p_loginid, p_loginid, CURRENT_TIMESTAMP)
    RETURNING offender_id INTO new_offender_id;

    RETURN QUERY SELECT 'offender added'::character varying, new_offender_id,
                        p_offender_name, p_loginid;
END;
$$;

-- Keep the original three-argument API available for direct callers.
CREATE OR REPLACE FUNCTION public.fn_add_offenders(
    p_offender_name CHARACTER VARYING,
    p_embedding DOUBLE PRECISION[],
    p_loginid INTEGER
)
RETURNS TABLE(p_message CHARACTER VARYING, new_offender_id INTEGER,
              new_offender_name CHARACTER VARYING, new_loginid INTEGER)
LANGUAGE plpgsql
COST 100
VOLATILE
PARALLEL UNSAFE
ROWS 1000
AS $$
BEGIN
    RETURN QUERY SELECT * FROM public.fn_add_offenders(
        p_offender_name, p_embedding, p_loginid, NULL::TEXT);
END;
$$;

CREATE OR REPLACE FUNCTION public.fn_search_offender(
    _camera_id INTEGER,
    _input_embedding TEXT,
    _capture_time CHARACTER VARYING
)
RETURNS TABLE(name CHARACTER VARYING, matched_offender_id INTEGER,
              matched_percentage DOUBLE PRECISION, matched_status CHARACTER VARYING,
              new_logid INTEGER)
LANGUAGE plpgsql
COST 100
VOLATILE
PARALLEL UNSAFE
ROWS 1000
AS $$
DECLARE
    v_embedding DOUBLE PRECISION[] := _input_embedding::DOUBLE PRECISION[];
    v_name CHARACTER VARYING;
    v_offender_id INTEGER;
    v_loginid INTEGER;
    v_similarity DOUBLE PRECISION;
    v_status CHARACTER VARYING;
    v_logid INTEGER;
BEGIN
    SELECT o.offender_name, o.offender_id, o.loginid, score.cosine_similarity
      INTO v_name, v_offender_id, v_loginid, v_similarity
      FROM public.tbl_offenders o
      CROSS JOIN LATERAL (
          SELECT SUM(q.value * o.embedding[q.ordinality::INTEGER]) /
                 NULLIF(SQRT(SUM(q.value * q.value) * SUM(o.embedding[q.ordinality::INTEGER] *
                       o.embedding[q.ordinality::INTEGER])), 0) AS cosine_similarity
            FROM UNNEST(v_embedding) WITH ORDINALITY AS q(value, ordinality)
           WHERE CARDINALITY(o.embedding) = CARDINALITY(v_embedding)
      ) score
     WHERE score.cosine_similarity IS NOT NULL
       AND o.is_active = TRUE
     ORDER BY score.cosine_similarity DESC
     LIMIT 1;

    v_status := CASE WHEN v_offender_id IS NOT NULL AND v_similarity >= 0.45
                     THEN 'matched' ELSE 'not_matched' END;
    IF v_status = 'not_matched' THEN
        v_name := NULL;
        v_offender_id := NULL;
        v_loginid := NULL;
    END IF;

    INSERT INTO public.offender_search_log
        (camera_id, capture_time, matched_offender_id, matched_percentage, matched_status)
    VALUES (_camera_id, _capture_time, v_offender_id,
            COALESCE(v_similarity, 0) * 100.0, v_status)
    RETURNING logid INTO v_logid;

    RETURN QUERY SELECT v_name, v_offender_id, COALESCE(v_similarity, 0) * 100.0,
                        v_status, v_logid;
END;
$$;
"""
