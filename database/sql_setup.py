"""Create the PostgreSQL objects the frontend flow needs, so nobody has to run psql by hand.

    database/sql/camera_config_postgres.sql  camera_config + sp_get_cameras / sp_upsert_camera /
                                             sp_delete_camera + the NOTIFY camera_changed trigger
    database/sql/dashboard_postgres.sql      sp_dash_* (read-only dashboard queries)

`ensure_sql_objects(db)` applies a file only when its objects are missing, so a database where you
already created (or customised) them is left alone. `force=True` re-applies both files, e.g. after
pulling a newer version of them. Both files are CREATE ... IF NOT EXISTS / CREATE OR REPLACE.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from database.schema import init_schema

log = logging.getLogger(__name__)

SQL_DIR = Path(__file__).resolve().parent / "sql"
CAMERA_SQL = SQL_DIR / "camera_config_postgres.sql"
DASHBOARD_SQL = SQL_DIR / "dashboard_postgres.sql"

_CAMERA_FUNCS = ("sp_get_cameras", "sp_upsert_camera", "sp_delete_camera")
_DASH_FUNCS = ("sp_dash_hourly_in_out", "sp_dash_daily_totals", "sp_dash_summary", "sp_dash_camera_counts")


def _missing_functions(db, names) -> List[str]:
    rows = db.fetchall("SELECT proname FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace "
                       "WHERE n.nspname = current_schema() AND proname = ANY (?)", (list(names),))
    have = {r["proname"] for r in rows}
    return [n for n in names if n not in have]


def _table_exists(db, name: str) -> bool:
    row = db.fetchone("SELECT to_regclass(?) IS NOT NULL AS ok", (name,))
    return bool(row and row["ok"])


def ensure_sql_objects(db, force: bool = False) -> List[str]:
    """Create facereco's tables, then any missing camera / dashboard SPs. Returns the files applied."""
    if getattr(db, "backend", "sqlite") != "postgres":
        raise RuntimeError("The camera table and stored procedures need PostgreSQL "
                           "(set FR_DATABASE_URL or FR_PG_HOST in .env)")
    init_schema(db)                                  # cameras / events / person_counts / ... (dashboard SPs read them)
    applied = []
    if force or not _table_exists(db, "camera_config") or _missing_functions(db, _CAMERA_FUNCS):
        db.executescript(CAMERA_SQL.read_text(encoding="utf-8"))
        applied.append(CAMERA_SQL.name)
    if force or _missing_functions(db, _DASH_FUNCS):
        db.executescript(DASHBOARD_SQL.read_text(encoding="utf-8"))
        applied.append(DASHBOARD_SQL.name)
    if applied:
        log.info("Applied %s", ", ".join(applied))
    return applied
