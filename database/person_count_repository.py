"""Person-count rows (appended once per camera per interval)."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from database.connection import Database

COUNT_COLUMNS = ("camera_id", "period_start", "period_end", "frames", "current_count", "min_count",
                 "max_count", "avg_count", "new_persons", "in_count", "out_count", "zone_count", "ts")
CROSSING_COLUMNS = ("camera_id", "track_id", "direction", "ts", "x", "y")

# Column names accepted from the "starting totals" SP (FR_PERSON_INIT_SQL), lower-case.
_INIT_ID = ("camera_id", "device_id", "cam_id", "camera_code", "device_code", "id")
_INIT_TOTAL = ("total_count", "total", "person_count", "personcount", "total_persons", "count")
_INIT_IN = ("in_count", "total_in", "incount", "entry_count", "entries")
_INIT_OUT = ("out_count", "total_out", "outcount", "exit_count", "exits")


def _pick(row: Mapping[str, Any], names: Sequence[str]) -> Any:
    for n in names:
        if row.get(n) is not None:
            return row[n]
    return None


def parse_initial_totals(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, int]]:
    """SP rows -> {camera_id: {"total": n, "in": n, "out": n}} (only the columns the SP returned)."""
    out: Dict[str, Dict[str, int]] = {}
    for raw in rows:
        row = {str(k).strip().lower(): v for k, v in dict(raw).items()}
        cid = _pick(row, _INIT_ID)
        if cid is None:
            continue
        vals = {}
        for key, names in (("total", _INIT_TOTAL), ("in", _INIT_IN), ("out", _INIT_OUT)):
            v = _pick(row, names)
            if v is not None:
                vals[key] = int(v)
        out[str(cid).strip()] = vals
    return out


class PersonCountRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def initial_totals(self, sql: str) -> Dict[str, Dict[str, int]]:
        """Run the starting-totals SP (e.g. fn_initialize_device_personcount) once."""
        return parse_initial_totals(self.db.fetchall(sql))

    def push_totals(self, sql: str, rows: Iterable[Sequence]) -> int:
        """Call the update SP (e.g. sp_upsert_person_count_update) once per (device, in, out, time) row."""
        n = 0
        for row in rows:
            self.db.execute(sql, tuple(row))
            n += 1
        return n

    def insert_many(self, rows: Iterable[Sequence]) -> int:
        """rows: tuples in COUNT_COLUMNS order."""
        payload = [tuple(r) for r in rows]
        if payload:
            cols = ", ".join(COUNT_COLUMNS)
            marks = ", ".join("?" * len(COUNT_COLUMNS))
            self.db.executemany(f"INSERT INTO person_counts({cols}) VALUES ({marks})", payload)
        return len(payload)

    def insert_crossings(self, rows: Iterable[Sequence]) -> int:
        """rows: tuples in CROSSING_COLUMNS order (one per IN/OUT crossing)."""
        payload = [tuple(r) for r in rows]
        if payload:
            self.db.executemany(f"INSERT INTO person_crossings({', '.join(CROSSING_COLUMNS)}) VALUES (?, ?, ?, ?, ?, ?)",
                                payload)
        return len(payload)

    def crossings(self, since: Optional[float] = None, until: Optional[float] = None,
                  camera_id: Optional[str] = None, limit: int = 1000) -> List[dict]:
        where, params = self._where(since, until, camera_id)
        sql = "SELECT p.* FROM person_crossings p" + where + " ORDER BY p.ts DESC"
        if limit:
            sql += f" LIMIT {int(limit)}"
        return [dict(r) for r in self.db.fetchall(sql, params)]

    @staticmethod
    def _where(since: Optional[float], until: Optional[float], camera_id: Optional[str]):
        where, params = [], []
        if since is not None:
            where.append("p.ts >= ?")
            params.append(since)
        if until is not None:
            where.append("p.ts < ?")
            params.append(until)
        if camera_id:
            where.append("p.camera_id = ?")
            params.append(camera_id)
        return (" WHERE " + " AND ".join(where)) if where else "", params

    def query(self, since: Optional[float] = None, until: Optional[float] = None,
              camera_id: Optional[str] = None, limit: int = 1000) -> List[dict]:
        where, params = self._where(since, until, camera_id)
        sql = ("SELECT p.*, c.name AS camera_name, c.location AS camera_location FROM person_counts p "
               "LEFT JOIN cameras c ON c.id = p.camera_id" + where + " ORDER BY p.ts DESC")
        if limit:
            sql += f" LIMIT {int(limit)}"
        return [dict(r) for r in self.db.fetchall(sql, params)]

    def totals(self, since: Optional[float] = None, until: Optional[float] = None,
               camera_id: Optional[str] = None) -> List[dict]:
        """Per camera: total distinct people, peak occupancy, frames analysed."""
        where, params = self._where(since, until, camera_id)
        sql = ("SELECT p.camera_id, c.name AS camera_name, SUM(p.new_persons) AS total_persons, "
               "MAX(p.max_count) AS peak_count, SUM(p.frames) AS frames, "
               "SUM(p.in_count) AS in_count, SUM(p.out_count) AS out_count, "
               "MIN(p.period_start) AS first_ts, MAX(p.period_end) AS last_ts "
               "FROM person_counts p LEFT JOIN cameras c ON c.id = p.camera_id" + where +
               " GROUP BY p.camera_id, c.name ORDER BY p.camera_id")
        return [dict(r) for r in self.db.fetchall(sql, params)]

    def delete_older_than(self, ts: float) -> int:
        n = self.db.execute("DELETE FROM person_counts WHERE ts < ?", (ts,)).rowcount
        self.db.execute("DELETE FROM person_crossings WHERE ts < ?", (ts,))
        return n
