"""Table DDL for tests only.

The application never creates tables in your database: they must already exist. Tests build a
scratch database, so they create the tables the repositories expect with this helper.
"""

SQLITE_SQL = """
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
    in_count      INTEGER NOT NULL DEFAULT 0,  -- IN crossings of the roi line in the interval
    out_count     INTEGER NOT NULL DEFAULT 0,  -- OUT crossings of the roi line
    zone_count    INTEGER NOT NULL DEFAULT 0,  -- IN - OUT occupancy at the end of the interval
    ts            REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_person_counts_ts ON person_counts(ts);
CREATE INDEX IF NOT EXISTS idx_person_counts_camera_ts ON person_counts(camera_id, ts);

-- One row per confirmed line crossing (written every few seconds, in batches).
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

PG_SQL = """
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


def create_tables(db) -> None:
    db.executescript(PG_SQL if getattr(db, "backend", "sqlite") == "postgres" else SQLITE_SQL)
