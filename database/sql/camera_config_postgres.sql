-- Camera configuration driven by your frontend, for FR_CAMERA_SOURCE=db.
--
--   psql "$FR_DATABASE_URL" -f database/sql/camera_config_postgres.sql
--
-- Frontend -> your API -> SELECT sp_upsert_camera(...) / sp_delete_camera(...)
--   -> trigger sends NOTIFY camera_changed
--   -> facereco (LISTEN camera_changed) calls sp_get_cameras() and starts/stops the use cases.
--
-- Already have your own tables/SP? Skip this file. Make your SP return one row per camera with
-- the columns camera_id, url and tasks (or boolean face / person), point FR_CAMERA_SQL at it, and
-- add a NOTIFY camera_changed (or the trigger below) wherever cameras are changed.
-- Safe to run again: everything is CREATE ... IF NOT EXISTS / CREATE OR REPLACE.
-- facereco itself only applies this file when camera_config already exists; it never creates tables.

CREATE TABLE IF NOT EXISTS camera_config (
    camera_id    TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    location     TEXT NOT NULL DEFAULT '',
    url          TEXT NOT NULL,                        -- full RTSP URL incl. credentials: restrict who can read it
    run_face     BOOLEAN NOT NULL DEFAULT FALSE,       -- use case: face recognition
    run_person   BOOLEAN NOT NULL DEFAULT FALSE,       -- use case: person detection + IN/OUT counting
    roi          JSONB,                                -- IN/OUT line [x1,y1,x2,y2], pixels or 0..1 fractions
    fps          REAL,                                 -- NULL = cameras.yaml defaults
    width        INTEGER,
    height       INTEGER,
    codec        TEXT,
    enabled      BOOLEAN NOT NULL DEFAULT TRUE,
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Older versions returned / accepted a polygon count_zone: drop those signatures first.
DROP FUNCTION IF EXISTS sp_get_cameras();
DROP FUNCTION IF EXISTS sp_upsert_camera(TEXT, TEXT, TEXT, BOOLEAN, BOOLEAN, JSONB, TEXT, BOOLEAN, REAL, INTEGER,
                                         INTEGER, JSONB, TEXT);

-- What facereco calls (FR_CAMERA_SQL default). Only enabled cameras with at least one use case.
CREATE OR REPLACE FUNCTION sp_get_cameras()
RETURNS TABLE (camera_id TEXT, name TEXT, location TEXT, url TEXT, tasks TEXT, roi TEXT,
               fps REAL, width INTEGER, height INTEGER, codec TEXT, enabled BOOLEAN)
LANGUAGE sql STABLE AS $$
    SELECT c.camera_id, c.name, c.location, c.url,
           concat_ws(',', CASE WHEN c.run_face THEN 'face' END, CASE WHEN c.run_person THEN 'person' END),
           CASE WHEN c.run_person THEN c.roi::text END,
           c.fps, c.width, c.height, c.codec, c.enabled
    FROM camera_config c
    WHERE c.enabled AND (c.run_face OR c.run_person)
    ORDER BY c.camera_id
$$;

-- What your API calls when the frontend saves a camera (insert or update in one call).
CREATE OR REPLACE FUNCTION sp_upsert_camera(
    p_camera_id TEXT, p_name TEXT, p_url TEXT, p_run_face BOOLEAN, p_run_person BOOLEAN,
    p_location TEXT DEFAULT '', p_enabled BOOLEAN DEFAULT TRUE,
    p_fps REAL DEFAULT NULL, p_width INTEGER DEFAULT NULL, p_height INTEGER DEFAULT NULL,
    p_roi JSONB DEFAULT NULL, p_codec TEXT DEFAULT NULL)
RETURNS VOID LANGUAGE sql AS $$
    INSERT INTO camera_config AS c (camera_id, name, url, run_face, run_person, location, enabled,
                                    fps, width, height, roi, codec, updated_at)
    VALUES (p_camera_id, p_name, p_url, p_run_face, p_run_person, COALESCE(p_location, ''),
            p_enabled, p_fps, p_width, p_height,
            CASE WHEN p_run_person THEN p_roi ELSE NULL END, p_codec, now())
    ON CONFLICT (camera_id) DO UPDATE SET
        name = excluded.name, url = excluded.url, run_face = excluded.run_face, run_person = excluded.run_person,
        location = excluded.location, enabled = excluded.enabled,
        fps = excluded.fps, width = excluded.width, height = excluded.height, roi = excluded.roi,
        codec = excluded.codec, updated_at = now()
$$;

CREATE OR REPLACE FUNCTION sp_delete_camera(p_camera_id TEXT)
RETURNS VOID LANGUAGE sql AS $$ DELETE FROM camera_config WHERE camera_id = p_camera_id $$;

-- Any change to camera_config (through the SPs or a direct UPDATE) wakes facereco up at once.
CREATE OR REPLACE FUNCTION trg_camera_config_notify() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    PERFORM pg_notify('camera_changed', COALESCE(NEW.camera_id, OLD.camera_id));
    RETURN NULL;
END $$;

DROP TRIGGER IF EXISTS camera_config_notify ON camera_config;
CREATE TRIGGER camera_config_notify AFTER INSERT OR UPDATE OR DELETE ON camera_config
    FOR EACH ROW EXECUTE FUNCTION trg_camera_config_notify();
