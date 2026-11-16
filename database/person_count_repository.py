"""Person-count rows (appended once per camera per interval)."""
from __future__ import annotations

import re
from datetime import date, datetime
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


_TABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$")
FLAG_IN, FLAG_OUT = "In", "Out"


def device_key(camera_id: str) -> Any:
    """camera_id '27' -> 27 (device_id is an integer column); anything else stays text."""
    return int(camera_id) if str(camera_id).isdigit() else camera_id


class DeviceCountTable:
    """IN / OUT totals kept in the customer's tbl_person_counts, one row per device / flag / day:

      personcountid | gate_id | device_id | person_count | lastupdateddate | ml_givencount | device_flag | detected_personcount | business_date
              8083  |      11 |        16 |          799 | 2026-10-07 ...  |           799 | In          |                    1 | 2026-10-07

    person_count = ml_givencount = the day's running total for that direction;
    detected_personcount = how many were added by this update. Rows are updated in place during the day.
    """

    def __init__(self, db: Database, table: str = "tbl_person_counts") -> None:
        if not _TABLE_NAME.match(table or ""):
            raise ValueError(f"invalid person-count table name: {table!r}")
        self.db = db
        self.table = table

    def day_totals(self, business_date: date) -> Dict[str, Dict[str, int]]:
        """{device_id: {"in": n, "out": n}} already stored for that business_date."""
        rows = self.db.fetchall(
            f"SELECT device_id, device_flag, MAX(person_count) AS n FROM {self.table} "
            "WHERE business_date = ? AND device_flag IN (?, ?) GROUP BY device_id, device_flag",
            (business_date, FLAG_IN, FLAG_OUT))
        out: Dict[str, Dict[str, int]] = {}
        for r in rows:
            r = {str(k).lower(): v for k, v in dict(r).items()}
            if r.get("device_id") is None:
                continue
            key = "in" if str(r["device_flag"]) == FLAG_IN else "out"
            out.setdefault(str(r["device_id"]).strip(), {})[key] = int(r["n"] or 0)
        return out

    def last_gate(self, device_id: Any) -> Optional[int]:
        """The gate this device was last recorded on (None if it never had one)."""
        row = self.db.fetchone(
            f"SELECT gate_id FROM {self.table} WHERE device_id = ? AND gate_id IS NOT NULL "
            "ORDER BY personcountid DESC LIMIT 1", (device_id,))
        return None if row is None else dict(row)["gate_id"]

    def upsert(self, device_id: Any, gate_id: Optional[int], flag: str, total: int, detected: int,
               when: datetime, business_date: date) -> None:
        """Update today's row for (device, flag); insert it if it isn't there yet."""
        n = self.db.execute(
            f"UPDATE {self.table} SET person_count = ?, ml_givencount = ?, detected_personcount = ?, "
            "lastupdateddate = ?, gate_id = COALESCE(gate_id, ?) "
            "WHERE device_id = ? AND device_flag = ? AND business_date = ?",
            (total, total, detected, when, gate_id, device_id, flag, business_date)).rowcount
        if n == 0:
            self.db.execute(
                f"INSERT INTO {self.table} (gate_id, device_id, person_count, lastupdateddate, ml_givencount, "
                "device_flag, detected_personcount, business_date) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (gate_id, device_id, total, when, total, flag, detected, business_date))


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
