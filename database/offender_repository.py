"""PostgreSQL bridge to the offender search function."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict

import numpy as np

from database.connection import Database


class OffenderRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def search(self, camera_id: str, embedding: np.ndarray, capture_ts: float) -> Dict[str, Any]:
        """Call fn_search_offender; camera IDs like cam01 map to integer 1."""
        match = re.search(r"(\d+)$", str(camera_id))
        if not match:
            raise ValueError(f"camera_id {camera_id!r} has no numeric suffix for fn_search_offender")
        values = np.asarray(embedding, dtype=np.float64).reshape(-1)
        pg_array_text = "{" + ",".join(format(float(v), ".17g") for v in values) + "}"
        capture_time = datetime.fromtimestamp(capture_ts, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        row = self.db.fetchone(
            "SELECT * FROM public.fn_search_offender(?, ?, ?)",
            (int(match.group(1)), pg_array_text, capture_time),
        )
        if row is None:
            raise RuntimeError("fn_search_offender returned no row")
        result = dict(row)
        if result.get("matched_offender_id") is not None:
            mapping = self.db.fetchone(
                "SELECT person_id FROM public.offender_embedding_sync WHERE offender_id = ?",
                (int(result["matched_offender_id"]),),
            )
            if mapping is None:
                mapping = self.db.fetchone(
                    "SELECT person_id FROM public.offender_sync WHERE offender_id = ?",
                    (int(result["matched_offender_id"]),),
                )
            result["person_id"] = int(mapping["person_id"]) if mapping else None
        else:
            result["person_id"] = None
        return result
