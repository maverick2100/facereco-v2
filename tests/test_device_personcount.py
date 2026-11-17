"""Starting totals (fn_initialize_device_personcount) and count updates (sp_upsert_person_count_update)."""
from config.settings import CameraConfig, PersonSettings
from database.person_count_repository import parse_initial_totals
from person_detection.person_count_service import PersonCountService

LINE = [0, 50, 100, 50]          # count line [x1, y1, x2, y2]


class FakeRepo:
    def __init__(self, init_rows, fail_init=False):
        self.init_rows, self.fail_init = init_rows, fail_init
        self.init_sql, self.calls = None, []

    def initial_totals(self, sql):
        self.init_sql = sql
        if self.fail_init:
            raise RuntimeError("function does not exist")
        return parse_initial_totals(self.init_rows)

    def push_totals(self, sql, rows):
        self.calls += [(sql, r) for r in rows]
        return len(rows)

    def take_crossings(self):
        return []


def _service(repo, cams=None):
    cams = cams or [CameraConfig.from_dict({"camera_id": "2", "url": "x", "tasks": ["person"], "roi": LINE}),
                    CameraConfig.from_dict({"camera_id": "3", "url": "y", "tasks": ["person"], "roi": LINE})]
    return PersonCountService(engine=None, frames=None, repo=repo, cameras=cams, settings=PersonSettings())


def test_parse_initial_totals_accepts_common_column_names():
    rows = [{"Device_ID": 2, "In_Count": 10, "Out_Count": 15, "Total_Count": 25},
            {"camera_id": "gate1", "total": 7},
            {"something": 1}]                                   # no id -> ignored
    assert parse_initial_totals(rows) == {"2": {"in": 10, "out": 15, "total": 25}, "gate1": {"total": 7}}


def test_starts_from_sp_totals_then_calls_update_sp_only_on_change():
    repo = FakeRepo([{"device_id": 2, "in_count": 10, "out_count": 15}])
    svc = _service(repo)
    svc.load_initial_totals()
    assert repo.init_sql == "SELECT * FROM public.fn_initialize_device_personcount()"
    assert svc.counter.in_out_totals() == {"2": (10, 15), "3": (0, 0)}

    svc._push_totals()                       # nothing changed since start -> no SP call
    assert repo.calls == []

    svc.counter.cameras["2"].line.total_in += 1   # someone walks in on device 2
    svc._push_totals()
    sql, (device, n_in, n_out, stamp) = repo.calls[-1]
    assert sql == "CALL sp_upsert_person_count_update(?, ?, ?, ?)"
    assert (device, n_in, n_out) == (2, 11, 15) and len(stamp) == 19   # 'YYYY-MM-DD HH:MM:SS'
    n = len(repo.calls)
    svc._push_totals()
    assert len(repo.calls) == n              # nothing changed -> no call


def test_init_sp_failure_starts_at_zero():
    repo = FakeRepo([], fail_init=True)
    svc = _service(repo)
    svc.load_initial_totals()
    assert svc.counter.in_out_totals() == {"2": (0, 0), "3": (0, 0)}


# ------------------------------------------------------------------ tbl_person_counts (direct write)
import datetime as _dt

from database.connection import Database
from database.person_count_repository import DeviceCountTable


class FakeTable:
    table = "tbl_person_counts"

    def __init__(self, day_rows=None, gate=None):
        self.day_rows, self.gate, self.rows = day_rows or {}, gate, []

    def day_totals(self, business_date):
        return self.day_rows

    def last_gate(self, device_id):
        return self.gate

    def upsert(self, device_id, gate_id, flag, total, detected, when, business_date):
        self.rows.append((gate_id, device_id, flag, total, detected, business_date))


def _table_service(table, cams=None):
    cams = cams or [CameraConfig.from_dict({"camera_id": "16", "url": "x", "tasks": ["person"],
                                            "roi": LINE, "gate_id": 11})]
    return PersonCountService(engine=None, frames=None, repo=FakeRepo([]), cameras=cams,
                              settings=PersonSettings(), table=table)


def test_table_mode_continues_from_today_and_writes_in_out_rows():
    table = FakeTable({"16": {"in": 798, "out": 141}})
    svc = _table_service(table)
    svc.load_initial_totals()
    assert svc.counter.in_out_totals() == {"16": (798, 141)}
    svc._push_totals()
    assert table.rows == []                                   # nothing new yet

    svc.counter.cameras["16"].line.total_in += 1
    svc._push_totals()
    today = _dt.date.today()
    assert table.rows == [(11, 16, "In", 799, 1, today)]     # only the direction that changed


def test_table_mode_first_write_of_day_creates_both_rows_and_uses_last_gate():
    table = FakeTable(gate=3)
    cams = [CameraConfig.from_dict({"camera_id": "4", "url": "x", "tasks": ["person"], "roi": LINE})]
    svc = _table_service(table, cams)
    svc.load_initial_totals()
    svc.counter.cameras["4"].line.total_out += 1
    svc._push_totals()
    today = _dt.date.today()
    assert table.rows == [(3, 4, "In", 0, 0, today), (3, 4, "Out", 1, 1, today)]


def test_table_mode_resets_on_new_business_date():
    table = FakeTable({"16": {"in": 50, "out": 20}})
    svc = _table_service(table)
    svc.load_initial_totals()
    svc._day = _dt.date.today() - _dt.timedelta(days=1)       # midnight passed
    svc.counter.cameras["16"].line.total_in += 1              # crossing just after midnight
    svc._push_totals()
    assert svc.counter.in_out_totals() == {"16": (0, 0)}       # reset; next crossing counts from 0
    svc.counter.cameras["16"].line.total_in += 1
    svc._push_totals()
    today = _dt.date.today()
    assert table.rows == [(11, 16, "In", 1, 1, today), (11, 16, "Out", 0, 0, today)]


def test_device_count_table_sql_upserts_one_row_per_device_flag_day(tmp_path):
    db = Database(tmp_path / "t.db")
    db.executescript("""CREATE TABLE tbl_person_counts (personcountid INTEGER PRIMARY KEY AUTOINCREMENT,
        gate_id INTEGER, device_id INTEGER, person_count INTEGER, lastupdateddate TEXT, ml_givencount INTEGER,
        device_flag TEXT, detected_personcount INTEGER, business_date TEXT)""")
    t = DeviceCountTable(db)
    day = _dt.date(2026, 10, 7)
    db.execute("INSERT INTO tbl_person_counts (gate_id, device_id, device_flag, person_count, business_date) "
               "VALUES (11, 16, 'In', 5, '2026-10-06')")
    assert t.last_gate(16) == 11
    t.upsert(16, 11, "In", 798, 1, _dt.datetime(2026, 10, 7, 11, 50), day)
    t.upsert(16, 11, "In", 799, 1, _dt.datetime(2026, 10, 7, 11, 55, 15), day)
    t.upsert(16, 11, "Out", 141, 1, _dt.datetime(2026, 10, 7, 11, 16, 12), day)
    rows = [dict(r) for r in db.fetchall("SELECT gate_id, device_id, person_count, ml_givencount, device_flag, "
                                         "detected_personcount FROM tbl_person_counts WHERE business_date = ? "
                                         "ORDER BY personcountid", (day,))]
    assert rows == [
        {"gate_id": 11, "device_id": 16, "person_count": 799, "ml_givencount": 799, "device_flag": "In",
         "detected_personcount": 1},
        {"gate_id": 11, "device_id": 16, "person_count": 141, "ml_givencount": 141, "device_flag": "Out",
         "detected_personcount": 1},
    ]
    assert t.day_totals(day) == {"16": {"in": 799, "out": 141}}


def test_device_count_table_rejects_bad_table_name():
    import pytest
    with pytest.raises(ValueError):
        DeviceCountTable(None, "tbl; DROP TABLE x")
