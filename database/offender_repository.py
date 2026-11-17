"""Offender matching against tbl_offenders and the search log (tbl_offender_search_logs).

Both tables must already exist - facereco never creates tables.
"""
from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

from database.connection import Database


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")


def _configured_table_sql(env_name: str, default: str) -> str:
    raw = os.getenv(env_name, default).strip()
    parts = raw.split(".")
    if len(parts) == 1:
        parts.insert(0, "public")
    if len(parts) != 2 or any(not _IDENTIFIER.fullmatch(part) for part in parts):
        raise ValueError(f"{env_name} must be a table name or schema.table using SQL identifiers")
    return ".".join('"' + part.replace('"', '""') + '"' for part in parts)


def offender_table_sql() -> str:
    """Return the configured offender embeddings table (default public.tbl_offenders)."""
    return _configured_table_sql("FR_OFFENDER_TABLE", "public.tbl_offenders")


def offender_search_log_table_sql() -> str:
    """Return the configured offender search log table (default public.tbl_offender_search_logs)."""
    return _configured_table_sql("FR_OFFENDER_SEARCH_LOG_TABLE", "public.tbl_offender_search_logs")


def pg_array(vec: np.ndarray) -> list:
    return np.asarray(vec, dtype=np.float64).reshape(-1).tolist()


class OffenderRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def search(self, embedding: np.ndarray) -> Optional[Dict[str, Any]]:
        """Best active tbl_offenders row by cosine similarity: {offender_id, offender_name, similarity}.

        None when the table has no active row with an embedding of the same length.
        """
        return self.db.fetchone(
            "WITH q AS (SELECT ?::double precision[] AS v) "
            "SELECT o.offender_id, o.offender_name, s.similarity "
            f"FROM {offender_table_sql()} o CROSS JOIN q "
            "CROSS JOIN LATERAL ("
            "    SELECT SUM(p.val * o.embedding[p.i::int]) / "
            "           NULLIF(SQRT(SUM(p.val * p.val) * SUM(o.embedding[p.i::int] * o.embedding[p.i::int])), 0)"
            "           AS similarity "
            "    FROM UNNEST(q.v) WITH ORDINALITY AS p(val, i)"
            ") s "
            "WHERE o.is_active = TRUE AND CARDINALITY(o.embedding) = CARDINALITY(q.v) AND s.similarity IS NOT NULL "
            "ORDER BY s.similarity DESC LIMIT 1",
            (pg_array(embedding),),
        )

    def log_search(self, device_id: Any, gate_id: Optional[int], offender_id: Optional[int],
                   embedding: np.ndarray, frameno: int, detected_imagepath: Optional[str],
                   detected_face: Optional[str], match_percentage: float, time_taken_ms: float,
                   capture_time: datetime) -> Optional[int]:
        """Insert one row into tbl_offender_search_logs and return its log_id."""
        row = self.db.fetchone(
            f"INSERT INTO {offender_search_log_table_sql()} (device_id, gate_id, offender_id, offender_embedding, "
            "frameno, detected_imagepath, detected_face, match_percentage, time_taken, capture_time, is_active, "
            "created_date, updated_date) "
            "VALUES (?, ?, ?, ?::double precision[], ?, ?, ?, ?, ?, ?, TRUE, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP) "
            "RETURNING log_id",
            (device_id, gate_id, offender_id, pg_array(embedding), int(frameno), detected_imagepath,
             detected_face, round(float(match_percentage), 2), round(float(time_taken_ms), 1), capture_time),
        )
        return int(row["log_id"]) if row and row.get("log_id") is not None else None
