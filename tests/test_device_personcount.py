"""Starting totals (fn_initialize_device_personcount) and count updates (sp_upsert_person_count_update)."""
from config.settings import CameraConfig, PersonSettings
from database.person_count_repository import parse_initial_totals
from person_detection.person_count_service import PersonCountService

ZONE = [[0, 0], [100, 0], [100, 100], [0, 100]]


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
    cams = cams or [CameraConfig.from_dict({"camera_id": "2", "url": "x", "tasks": ["person"], "count_zone": ZONE}),
                    CameraConfig.from_dict({"camera_id": "3", "url": "y", "tasks": ["person"], "count_zone": ZONE})]
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

    svc.counter.cameras["2"].zone.total_in += 1   # someone walks in on device 2
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
