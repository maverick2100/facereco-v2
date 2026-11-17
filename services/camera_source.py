"""Where the camera list comes from: config/cameras.yaml or your stored procedure (SP).

With FR_CAMERA_SOURCE=db the service runs FR_CAMERA_SQL (default `SELECT * FROM sp_get_cameras()`).
Each row returned is one camera, and its columns say which use case runs on it:

    camera_id | name  | url            | tasks          | enabled | roi                 | fps ...
    cam01     | Gate  | rtsp://...     | face           | true    |                     |
    cam02     | Hall  | rtsp://...     | person         | true    | [0.5, 0, 0.5, 1]    |
    cam03     | Lobby | rtsp://...     | face,person    | true    |                     |

`tasks` may be text ("face", "person", "face,person", "both"), a text[] array, or replaced by two
boolean columns `face` / `person` (also accepted: run_face / face_enabled, run_person / person_enabled).
Common column names are accepted too (id, rtsp_url, use_case, is_active, ...). Columns that aren't
camera settings are ignored, and NULLs fall back to cameras.yaml `defaults:`.

Changes are picked up by polling every FR_CAMERA_POLL_S, and instantly on PostgreSQL when something
runs `NOTIFY camera_changed` (the trigger in database/sql/camera_config_postgres.sql does this).
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import fields
from typing import Any, Dict, List, Mapping, Optional

from config.settings import CameraConfig, CameraSourceSettings, StorageSettings, load_cameras
from constants import TASK_FACE, TASK_PERSON

log = logging.getLogger(__name__)

_CAMERA_FIELDS = {f.name for f in fields(CameraConfig)}
_ALIASES = {
    "id": "camera_id", "cam_id": "camera_id", "camera_code": "camera_id",
    "camera_name": "name", "cam_name": "name",
    "rtsp_url": "url", "stream_url": "url", "rtsp": "url", "camera_url": "url",
    "use_case": "tasks", "usecase": "tasks", "use_cases": "tasks", "usecases": "tasks", "task": "tasks",
    "is_enabled": "enabled", "is_active": "enabled", "active": "enabled",
    "line": "roi", "count_line": "roi", "counting_line": "roi",
    "direction": "in_direction", "count_direction": "in_direction",
    "gateid": "gate_id", "gate": "gate_id",
}
_FACE_FLAGS = ("face", "run_face", "face_enabled", "is_face", "face_recognition")
_PERSON_FLAGS = ("person", "run_person", "person_enabled", "is_person", "person_detection", "person_count")
_TASK_WORDS = {
    "face": [TASK_FACE], "face_recognition": [TASK_FACE], "facerecognition": [TASK_FACE], "fr": [TASK_FACE],
    "person": [TASK_PERSON], "person_detection": [TASK_PERSON], "persondetection": [TASK_PERSON],
    "person_count": [TASK_PERSON], "person_counting": [TASK_PERSON], "counting": [TASK_PERSON],
    "people": [TASK_PERSON], "pd": [TASK_PERSON],
    "both": [TASK_FACE, TASK_PERSON], "all": [TASK_FACE, TASK_PERSON],
}


def _truthy(v: Any) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "t", "yes", "y", "on", "active", "enabled")
    return bool(v)


def parse_tasks(value: Any) -> List[str]:
    if isinstance(value, str):
        value = value.replace(";", ",").replace("+", ",").replace("|", ",").split(",")
    out: List[str] = []
    for item in value or []:
        word = str(item).strip().lower().replace(" ", "_").replace("-", "_")
        if not word:
            continue
        if word not in _TASK_WORDS:
            raise ValueError(f"unknown use case {item!r} (use face, person or both)")
        out += _TASK_WORDS[word]
    return list(dict.fromkeys(out))


def row_to_camera(row: Mapping[str, Any], defaults: Optional[Dict[str, Any]] = None) -> Optional[CameraConfig]:
    """One SP row -> CameraConfig. Returns None for a disabled camera or one with no use case."""
    data: Dict[str, Any] = {}
    flags: Dict[str, bool] = {}
    row_lc = {str(k).strip().lower(): v for k, v in row.items()}
    for key, value in row.items():
        k = str(key).strip().lower()
        if k in _FACE_FLAGS and value is not None:
            flags[TASK_FACE] = _truthy(value)
            continue
        if k in _PERSON_FLAGS and value is not None:
            flags[TASK_PERSON] = _truthy(value)
            continue
        k = _ALIASES.get(k, k)
        if k in _CAMERA_FIELDS and value is not None and k not in data:
            data[k] = value
    ends = {k: row_lc[k] for k in ("x1", "y1", "x2", "y2") if row_lc.get(k) is not None}
    if "roi" not in data and len(ends) == 4:          # line given as four columns x1, y1, x2, y2
        data["roi"] = [ends["x1"], ends["y1"], ends["x2"], ends["y2"]]
    if "tasks" in data:
        data["tasks"] = parse_tasks(data["tasks"])
    elif flags:
        data["tasks"] = [t for t in (TASK_FACE, TASK_PERSON) if flags.get(t)]
    if "enabled" in data:
        data["enabled"] = _truthy(data["enabled"])
    if "camera_id" in data:
        data["camera_id"] = str(data["camera_id"])
    if data.get("tasks") == [] or not data.get("enabled", True):
        return None                                   # no use case selected / switched off
    merged_defaults = dict(defaults or {})
    if "tasks" in data:
        merged_defaults.pop("tasks", None)
    effective_tasks = data.get("tasks", parse_tasks(merged_defaults.get("tasks", [TASK_FACE])))
    if TASK_PERSON not in effective_tasks:
        data["roi"] = None
    return CameraConfig.from_dict(data, merged_defaults)


def rows_to_cameras(rows, defaults: Optional[Dict[str, Any]] = None) -> List[CameraConfig]:
    """Bad rows are logged and skipped, so one wrong camera doesn't stop the others."""
    cams: List[CameraConfig] = []
    seen = set()
    for row in rows:
        row = dict(row)
        try:
            cam = row_to_camera(row, defaults)
        except Exception as exc:
            log.error("Camera row %s skipped: %s", row.get("camera_id", row.get("id", "?")), exc)
            continue
        if cam is None:
            continue
        if cam.camera_id in seen:
            log.error("Camera %s returned twice - keeping the first row", cam.camera_id)
            continue
        seen.add(cam.camera_id)
        cams.append(cam)
    return cams


def fingerprint(cams: List[CameraConfig]) -> tuple:
    return tuple(sorted(repr(c) for c in cams))


def describe_change(old: List[CameraConfig], new: List[CameraConfig]) -> str:
    o, n = {c.camera_id: c for c in old}, {c.camera_id: c for c in new}
    parts = []
    for label, ids in (("added", sorted(set(n) - set(o))), ("removed", sorted(set(o) - set(n))),
                       ("changed", sorted(i for i in set(o) & set(n) if repr(o[i]) != repr(n[i])))):
        if ids:
            parts.append(f"{label}: {', '.join(ids)}")
    return "; ".join(parts) or "no change"


class CameraSource:
    """load() -> enabled cameras;  wait_for_change(timeout, stop) blocks until a NOTIFY or the timeout."""

    def __init__(self, settings: CameraSourceSettings, storage: Optional[StorageSettings] = None) -> None:
        self.settings = settings
        self.storage = storage
        self._db = None
        self._listen_conn = None
        self._listen_failed_at = 0.0

    @property
    def description(self) -> str:
        if self.settings.source == "db":
            return f"database ({self.settings.sql})"
        return str(self.settings.yaml_path)

    def load(self) -> List[CameraConfig]:
        if self.settings.source != "db":
            return [c for c in load_cameras(self.settings.yaml_path) if c.enabled]
        if self._db is None:
            from database.connection import open_database

            self._db = open_database(self.storage)
        return rows_to_cameras(self._db.fetchall(self.settings.sql), self.settings.defaults)

    # ------------------------------------------------------------------ change notifications
    def _listening(self) -> bool:
        if self.settings.source != "db" or not self.settings.notify_channel or self.storage is None:
            return False
        conninfo = self.storage.postgres_conninfo()
        if not conninfo:
            return False
        if self._listen_conn is not None and not self._listen_conn.closed:
            return True
        if time.monotonic() - self._listen_failed_at < 30:
            return False
        try:
            import psycopg

            self._listen_conn = psycopg.connect(conninfo, autocommit=True,
                                                connect_timeout=int(self.storage.pg_connect_timeout_s))
            self._listen_conn.execute(f"LISTEN {self.settings.notify_channel}")
            log.info("Listening for camera changes on channel '%s'", self.settings.notify_channel)
            return True
        except Exception as exc:
            log.warning("LISTEN %s failed (%s) - falling back to polling every %.0fs",
                        self.settings.notify_channel, exc, self.settings.poll_s)
            self._listen_conn, self._listen_failed_at = None, time.monotonic()
            return False

    def wait_for_change(self, timeout: float, stop: threading.Event) -> bool:
        """True if a NOTIFY arrived; False on timeout or stop."""
        deadline = time.monotonic() + timeout
        while not stop.is_set():
            left = deadline - time.monotonic()
            if left <= 0:
                return False
            if not self._listening():
                stop.wait(min(left, 1.0))
                continue
            try:
                for _ in self._listen_conn.notifies(timeout=min(left, 1.0), stop_after=1):
                    return True
            except Exception as exc:
                log.warning("Camera-change listener lost (%s) - reconnecting", exc)
                self._close_listener()
                self._listen_failed_at = time.monotonic()
        return False

    def _close_listener(self) -> None:
        if self._listen_conn is not None:
            try:
                self._listen_conn.close()
            except Exception:
                pass
            self._listen_conn = None

    def close(self) -> None:
        self._close_listener()
        if self._db is not None:
            self._db.close()
            self._db = None
