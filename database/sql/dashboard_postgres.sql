-- Dashboard stored procedures (read-only) for the people-count pages.
--
--   psql "$FR_DATABASE_URL" -f database/sql/dashboard_postgres.sql
--
-- The tables it reads (person_crossings / person_counts / cameras) must already exist.
-- Safe to run again (CREATE OR REPLACE).
--
-- Common parameters:
--   p_day      the day to show (NULL = today in p_tz)
--   p_cameras  only these camera ids, e.g. ARRAY['gate1','gate2'] (NULL = all cameras)
--   p_tz       time zone for days and hours (the app stores UTC epoch seconds)
--
-- IN / OUT only exist for cameras with a roi line. Live numbers come from person_crossings
-- (~2 s behind); unique people and peaks come from person_counts (written every FR_PERSON_FLUSH_S).

-- 1. Hourly IN / OUT for one day: always 24 rows (00:00 .. 23:00), zeros included.
--    SELECT * FROM sp_dash_hourly_in_out();                          -- today, all cameras
--    SELECT * FROM sp_dash_hourly_in_out('2026-11-13', ARRAY['gate1']);
CREATE OR REPLACE FUNCTION sp_dash_hourly_in_out(
    p_day DATE DEFAULT NULL, p_cameras TEXT[] DEFAULT NULL, p_tz TEXT DEFAULT 'Asia/Kolkata')
RETURNS TABLE (hour INTEGER, hour_label TEXT, in_count BIGINT, out_count BIGINT)
LANGUAGE sql STABLE AS $$
    WITH d AS (SELECT COALESCE(p_day, (now() AT TIME ZONE p_tz)::date) AS day),
    r AS (SELECT extract(epoch FROM (d.day::timestamp AT TIME ZONE p_tz)) AS t0,
                 extract(epoch FROM ((d.day + 1)::timestamp AT TIME ZONE p_tz)) AS t1 FROM d),
    c AS (
        SELECT extract(hour FROM to_timestamp(x.ts) AT TIME ZONE p_tz)::int AS hour,
               count(*) FILTER (WHERE x.direction = 'in')  AS i,
               count(*) FILTER (WHERE x.direction = 'out') AS o
        FROM person_crossings x, r
        WHERE x.ts >= r.t0 AND x.ts < r.t1
          AND (p_cameras IS NULL OR x.camera_id = ANY (p_cameras))
        GROUP BY 1)
    SELECT h.hour, lpad(h.hour::text, 2, '0') || ':00', COALESCE(c.i, 0), COALESCE(c.o, 0)
    FROM generate_series(0, 23) AS h(hour)
    LEFT JOIN c ON c.hour = h.hour
    ORDER BY h.hour
$$;

-- 2. Daily totals for a date range: one row per day, zeros included.
--    SELECT * FROM sp_dash_daily_totals();                           -- last 7 days incl. today
--    SELECT * FROM sp_dash_daily_totals('2026-11-07', '2026-11-13');
--    peak_count = highest number of people seen at once by any single camera that day.
CREATE OR REPLACE FUNCTION sp_dash_daily_totals(
    p_from DATE DEFAULT NULL, p_to DATE DEFAULT NULL, p_cameras TEXT[] DEFAULT NULL,
    p_tz TEXT DEFAULT 'Asia/Kolkata')
RETURNS TABLE (day DATE, day_label TEXT, in_count BIGINT, out_count BIGINT, unique_people BIGINT,
               peak_count INTEGER)
LANGUAGE sql STABLE AS $$
    WITH b AS (SELECT COALESCE(p_to, (now() AT TIME ZONE p_tz)::date) AS d_to),
    rng AS (SELECT COALESCE(p_from, b.d_to - 6) AS d_from, b.d_to FROM b),
    r AS (SELECT rng.*, extract(epoch FROM (rng.d_from::timestamp AT TIME ZONE p_tz)) AS t0,
                 extract(epoch FROM ((rng.d_to + 1)::timestamp AT TIME ZONE p_tz)) AS t1 FROM rng),
    c AS (
        SELECT (to_timestamp(p.ts) AT TIME ZONE p_tz)::date AS day,
               sum(p.in_count) AS i, sum(p.out_count) AS o, sum(p.new_persons) AS u, max(p.max_count) AS pk
        FROM person_counts p, r
        WHERE p.ts >= r.t0 AND p.ts < r.t1
          AND (p_cameras IS NULL OR p.camera_id = ANY (p_cameras))
        GROUP BY 1)
    SELECT g.day::date, to_char(g.day, 'DD Mon'), COALESCE(c.i, 0)::bigint, COALESCE(c.o, 0)::bigint,
           COALESCE(c.u, 0)::bigint, COALESCE(c.pk, 0)
    FROM r, generate_series(r.d_from, r.d_to, interval '1 day') AS g(day)
    LEFT JOIN c ON c.day = g.day::date
    ORDER BY 1
$$;

-- 3. Summary tiles for one day (one row).
--    SELECT * FROM sp_dash_summary();
--    inside_now      = IN - OUT since midnight (never below 0): people inside the counted area
--    in_zone_now     = IN - OUT occupancy estimate at the latest interval per camera
--    cameras_live    = cameras that reported in the last 3 minutes
CREATE OR REPLACE FUNCTION sp_dash_summary(
    p_day DATE DEFAULT NULL, p_cameras TEXT[] DEFAULT NULL, p_tz TEXT DEFAULT 'Asia/Kolkata')
RETURNS TABLE (day DATE, in_count BIGINT, out_count BIGINT, inside_now BIGINT, in_zone_now BIGINT,
               unique_people BIGINT, peak_count INTEGER, cameras_live BIGINT, last_update TIMESTAMPTZ)
LANGUAGE sql STABLE AS $$
    WITH d AS (SELECT COALESCE(p_day, (now() AT TIME ZONE p_tz)::date) AS day),
    r AS (SELECT d.day, extract(epoch FROM (d.day::timestamp AT TIME ZONE p_tz)) AS t0,
                 extract(epoch FROM ((d.day + 1)::timestamp AT TIME ZONE p_tz)) AS t1 FROM d),
    x AS (
        SELECT count(*) FILTER (WHERE c.direction = 'in') AS i, count(*) FILTER (WHERE c.direction = 'out') AS o,
               max(c.ts) AS last_ts
        FROM person_crossings c, r
        WHERE c.ts >= r.t0 AND c.ts < r.t1 AND (p_cameras IS NULL OR c.camera_id = ANY (p_cameras))),
    p AS (
        SELECT sum(pc.new_persons) AS u, max(pc.max_count) AS pk, max(pc.ts) AS last_ts
        FROM person_counts pc, r
        WHERE pc.ts >= r.t0 AND pc.ts < r.t1 AND (p_cameras IS NULL OR pc.camera_id = ANY (p_cameras))),
    latest AS (
        SELECT DISTINCT ON (pc.camera_id) pc.camera_id, pc.zone_count, pc.ts
        FROM person_counts pc
        WHERE pc.ts >= extract(epoch FROM now()) - 180 AND pc.ts <= extract(epoch FROM now()) + 60
          AND (p_cameras IS NULL OR pc.camera_id = ANY (p_cameras))
        ORDER BY pc.camera_id, pc.ts DESC)
    SELECT r.day, x.i, x.o, GREATEST(x.i - x.o, 0),
           COALESCE((SELECT sum(zone_count) FROM latest), 0)::bigint,
           COALESCE(p.u, 0)::bigint, COALESCE(p.pk, 0),
           (SELECT count(*) FROM latest),
           to_timestamp(GREATEST(x.last_ts, p.last_ts))
    FROM r, x, p
$$;

-- 4. Per-camera breakdown for one day (one row per camera that has data or is registered).
--    SELECT * FROM sp_dash_camera_counts();
CREATE OR REPLACE FUNCTION sp_dash_camera_counts(
    p_day DATE DEFAULT NULL, p_tz TEXT DEFAULT 'Asia/Kolkata')
RETURNS TABLE (camera_id TEXT, camera_name TEXT, location TEXT, in_count BIGINT, out_count BIGINT,
               unique_people BIGINT, peak_count INTEGER, current_count INTEGER, in_zone_now INTEGER,
               last_update TIMESTAMPTZ, is_live BOOLEAN)
LANGUAGE sql STABLE AS $$
    WITH d AS (SELECT COALESCE(p_day, (now() AT TIME ZONE p_tz)::date) AS day),
    r AS (SELECT extract(epoch FROM (d.day::timestamp AT TIME ZONE p_tz)) AS t0,
                 extract(epoch FROM ((d.day + 1)::timestamp AT TIME ZONE p_tz)) AS t1 FROM d),
    x AS (
        SELECT c.camera_id, count(*) FILTER (WHERE c.direction = 'in') AS i,
               count(*) FILTER (WHERE c.direction = 'out') AS o
        FROM person_crossings c, r WHERE c.ts >= r.t0 AND c.ts < r.t1 GROUP BY 1),
    p AS (
        SELECT pc.camera_id, sum(pc.new_persons) AS u, max(pc.max_count) AS pk
        FROM person_counts pc, r WHERE pc.ts >= r.t0 AND pc.ts < r.t1 GROUP BY 1),
    latest AS (
        SELECT DISTINCT ON (pc.camera_id) pc.camera_id, pc.current_count, pc.zone_count, pc.ts
        FROM person_counts pc WHERE pc.ts <= extract(epoch FROM now()) + 60   -- ignore clock-skewed rows
        ORDER BY pc.camera_id, pc.ts DESC),
    ids AS (SELECT camera_id FROM x UNION SELECT camera_id FROM p UNION SELECT id FROM cameras)
    SELECT ids.camera_id, COALESCE(cam.name, ids.camera_id), COALESCE(cam.location, ''),
           COALESCE(x.i, 0), COALESCE(x.o, 0), COALESCE(p.u, 0)::bigint, COALESCE(p.pk, 0),
           COALESCE(l.current_count, 0), COALESCE(l.zone_count, 0), to_timestamp(l.ts),
           COALESCE(l.ts >= extract(epoch FROM now()) - 180, false)
    FROM ids
    LEFT JOIN cameras cam ON cam.id = ids.camera_id
    LEFT JOIN x ON x.camera_id = ids.camera_id
    LEFT JOIN p ON p.camera_id = ids.camera_id
    LEFT JOIN latest l ON l.camera_id = ids.camera_id
    ORDER BY ids.camera_id
$$;
