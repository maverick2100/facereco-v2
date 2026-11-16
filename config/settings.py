"""Application settings: environment (.env) + config/cameras.yaml."""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from constants import (
    ARCFACE_INPUT_MEAN,
    ARCFACE_INPUT_STD,
    CAMERA_TASKS,
    COCO_PERSON_CLASS_ID,
    CONFIG_DIR,
    DEFAULT_ARCFACE_PATH,
    DEFAULT_DB_PATH,
    DEFAULT_SCRFD_PATH,
    DEFAULT_YOLO_PATH,
    PROJECT_ROOT,
    SCRFD_INPUT_MEAN,
    SCRFD_INPUT_STD,
    TASK_FACE,
    TASK_PERSON,
)

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- env helpers
def _env(name: str, default: Any) -> Any:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    raw = raw.strip()
    if isinstance(default, bool):
        return raw.lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


def _env_path(name: str, default: Path) -> Path:
    p = Path(str(_env(name, str(default)))).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p)


# --------------------------------------------------------------------------- dataclasses
@dataclass
class CameraConfig:
    camera_id: str
    name: str
    url: str
    enabled: bool = True
    location: str = ""
    codec: str = "h264"          # h264 | h265 | auto
    decoder: str = "vtdec"       # VideoToolbox hardware decoder (vtdec / vtdec_hw)
    width: int = 1280            # resolution delivered to Python
    height: int = 720
    fps: float = 4.0             # frames/s delivered to the pipeline (GStreamer drops the rest)
    latency_ms: int = 200
    transport: str = "tcp"       # tcp | udp
    priority: float = 1.0        # >1 = scheduled more aggressively
    # Person IN / OUT counting line: [x1, y1, x2, y2] = its two ends. Pixels in delivered resolution,
    # or 0..1 fractions of width/height (stored as pixels). Also accepted as "x1,y1,x2,y2" text / JSON.
    roi: Optional[List[float]] = None
    in_direction: str = ""           # which crossing is IN: down | up | right | left ("" = FR_PERSON_IN_DIRECTION)
    tasks: List[str] = field(default_factory=lambda: [TASK_FACE])  # face | person | both
    gate_id: Optional[int] = None    # written to tbl_person_counts.gate_id (else the device's last known gate)

    @property
    def runs_face(self) -> bool:
        return TASK_FACE in self.tasks

    @property
    def runs_person(self) -> bool:
        return TASK_PERSON in self.tasks

    @classmethod
    def from_dict(cls, data: Dict[str, Any], defaults: Optional[Dict[str, Any]] = None) -> "CameraConfig":
        merged = dict(defaults or {})
        merged.update(data or {})
        known = {f.name for f in fields(cls)}
        unknown = set(merged) - known
        if unknown:
            log.warning("Camera %s: ignoring unknown keys %s", merged.get("camera_id"), sorted(unknown))
        kwargs = {k: v for k, v in merged.items() if k in known}
        if "camera_id" not in kwargs or "url" not in kwargs:
            raise ValueError(f"camera entry needs camera_id and url: {data}")
        kwargs["camera_id"] = str(kwargs["camera_id"])
        kwargs.setdefault("name", kwargs["camera_id"])
        tasks = kwargs.get("tasks", [TASK_FACE])
        if isinstance(tasks, str):
            tasks = [t.strip() for t in tasks.split(",")]
        tasks = [str(t).strip().lower() for t in tasks if str(t).strip()]
        bad = [t for t in tasks if t not in CAMERA_TASKS]
        if bad or not tasks:
            raise ValueError(f"camera {kwargs['camera_id']}: tasks must be a non-empty subset of "
                             f"{list(CAMERA_TASKS)}, got {kwargs.get('tasks')!r}")
        kwargs["tasks"] = list(dict.fromkeys(tasks))
        cam = cls(**kwargs)
        cam.roi = parse_line(cam.camera_id, cam.roi, cam.width, cam.height)
        cam.in_direction = str(cam.in_direction or "").strip().lower()
        if cam.in_direction and cam.in_direction not in ("down", "up", "right", "left"):
            raise ValueError(f"camera {cam.camera_id}: in_direction must be down, up, right or left")
        return cam


def parse_line(camera_id: str, value: Any, width: int, height: int) -> Optional[List[float]]:
    """[x1, y1, x2, y2] (list, JSON text, or "x1,y1,x2,y2") -> pixels. None / empty = no line."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().strip("[]()")
        if not text:
            return None
        value = text.replace(";", ",").split(",")
    if isinstance(value, dict):                   # {"x1":..,"y1":..,"x2":..,"y2":..}
        value = [value.get(k) for k in ("x1", "y1", "x2", "y2")]
    try:
        pts = [float(v) for v in value]
    except (TypeError, ValueError):
        raise ValueError(f"camera {camera_id}: roi must be [x1, y1, x2, y2], got {value!r}") from None
    if not pts:
        return None
    if len(pts) != 4:
        raise ValueError(f"camera {camera_id}: roi must be 4 numbers [x1, y1, x2, y2], got {len(pts)}")
    if min(pts) < 0:
        raise ValueError(f"camera {camera_id}: roi coordinates must be >= 0")
    if all(v <= 1.0 for v in pts):                 # 0..1 fractions -> pixels
        pts = [pts[0] * width, pts[1] * height, pts[2] * width, pts[3] * height]
    if pts[0] == pts[2] and pts[1] == pts[3]:
        raise ValueError(f"camera {camera_id}: roi line's two ends are the same point")
    return pts


@dataclass
class ModelSettings:
    scrfd_path: Path = DEFAULT_SCRFD_PATH
    arcface_path: Path = DEFAULT_ARCFACE_PATH
    compute_units: str = "CPU_AND_NE"   # CPU_AND_NE keeps work on the Neural Engine (no GPU fallback)
    cache_compiled: bool = True
    scrfd_input_mean: float = SCRFD_INPUT_MEAN   # only used when model input is a MultiArray
    scrfd_input_std: float = SCRFD_INPUT_STD
    scrfd_apply_sigmoid: bool = False            # set true if your SCRFD outputs logits
    arcface_input_mean: float = ARCFACE_INPUT_MEAN
    arcface_input_std: float = ARCFACE_INPUT_STD
    warmup_iterations: int = 3
    yolo_path: Path = DEFAULT_YOLO_PATH


@dataclass
class PersonSettings:
    """YOLOv7 person detection + counting (cameras with `tasks: [person]`)."""
    conf_threshold: float = 0.4      # objectness * person-class score
    nms_threshold: float = 0.45
    class_id: int = COCO_PERSON_CLASS_ID
    num_classes: int = 80            # classes in your YOLOv7 model (COCO = 80)
    input_std: float = 255.0         # MultiArray inputs only: pixel / std (YOLO expects 0..1)
    min_box_height: int = 30         # px in delivered resolution; ignore tiny/far detections
    batch_size: int = 4
    max_wait_ms: float = 8.0
    max_fps: float = 0.0             # per-camera cap for person detection (0 = camera fps)
    track_iou: float = 0.3           # tracker: min IoU to continue a track
    track_max_age_s: float = 2.0     # tracker: drop a track not seen for this long
    track_min_hits: int = 2          # tracker: detections needed before a person is counted
    flush_interval_s: float = 60.0   # one count row per camera per interval is appended to the DB
    line_anchor: str = "bottom"      # point of the box that crosses the roi line: bottom (feet) | center
    line_min_frames: int = 2         # frames a person must stay on the new side before IN/OUT counts
    line_min_distance: float = 4.0   # px a person must be past the line (dead band against jitter)
    in_direction: str = "down"       # crossing that is IN: down | up (horizontal line), right | left (vertical)
    init_sql: str = "SELECT * FROM public.fn_initialize_device_personcount()"  # starting totals ("" = start at 0)
    update_sql: str = "CALL sp_upsert_person_count_update(?, ?, ?, ?)"  # (device, in, out, time); "" = off
    # Write IN / OUT straight into this table (one 'In' + one 'Out' row per device per business_date,
    # updated in place). When set it replaces init_sql / update_sql. "" = use the SPs instead.
    count_table: str = "tbl_person_counts"


@dataclass
class DetectionSettings:
    det_threshold: float = 0.5
    nms_threshold: float = 0.4
    min_face_size: int = 40          # px (shorter bbox side) in delivered resolution
    max_faces_per_frame: int = 20
    batch_size: int = 4              # frames per SCRFD batch
    max_wait_ms: float = 8.0         # max time to wait to fill a batch
    max_frame_age_s: float = 1.0     # frames older than this are dropped (never process backlog)


@dataclass
class RecognitionSettings:
    match_threshold: float = 0.45    # cosine similarity
    match_margin: float = 0.05       # best - second best person must exceed this
    batch_size: int = 16             # crops per ArcFace batch
    max_wait_ms: float = 10.0
    queue_size: int = 256
    blur_threshold: float = 40.0     # Laplacian variance of 112x112 crop
    max_yaw_ratio: float = 0.45      # nose offset / eye distance
    min_quality: float = 0.25
    event_cooldown_s: float = 10.0   # same person, same camera
    record_unknown: bool = True
    unknown_cooldown_s: float = 30.0
    unknown_similarity: float = 0.5
    track_iou: float = 0.45          # skip ArcFace for a face that overlaps a recent identified face
    track_reverify_s: float = 2.0    # ...but re-embed it at least this often
    gallery_reload_s: float = 30.0


@dataclass
class IngestionSettings:
    backend: str = "auto"            # auto | gstreamer | opencv
    reconnect_delay_s: float = 2.0
    reconnect_max_delay_s: float = 60.0
    stale_timeout_s: float = 15.0
    queue_depth: int = 1             # frames kept per camera (latest-frame-wins)


@dataclass
class StorageSettings:
    db_path: Path = DEFAULT_DB_PATH
    save_snapshots: bool = False
    snapshot_quality: int = 85
    event_flush_interval_s: float = 1.0
    event_batch_size: int = 100
    event_queue_size: int = 5000
    retention_days: int = 90
    # PostgreSQL: set database_url OR pg_host (+ pg_user / pg_password / pg_database). Empty = SQLite (db_path).
    database_url: str = ""
    pg_host: str = ""
    pg_port: int = 5432
    pg_user: str = ""
    pg_password: str = ""
    pg_database: str = ""
    pg_sslmode: str = ""             # disable | prefer | require | verify-full ...
    pg_pool_min: int = 1
    pg_pool_max: int = 10
    pg_connect_timeout_s: float = 15.0

    @property
    def backend(self) -> str:
        return "postgres" if (self.database_url or self.pg_host) else "sqlite"

    def postgres_conninfo(self) -> str:
        """psycopg connection string, or "" when SQLite is configured.

        Separate FR_PG_* values override the matching part of FR_DATABASE_URL, so a password with
        special characters can be given as FR_PG_PASSWORD without URL-encoding it."""
        if self.backend != "postgres":
            return ""
        from psycopg.conninfo import make_conninfo

        overrides = {k: v for k, v in {
            "host": self.pg_host, "port": str(self.pg_port) if self.pg_host else "",
            "user": self.pg_user, "password": self.pg_password, "dbname": self.pg_database,
            "sslmode": self.pg_sslmode,
        }.items() if v}
        return make_conninfo(self.database_url, **overrides)


@dataclass
class MonitoringSettings:
    health_host: str = "127.0.0.1"
    health_port: int = 8085          # 0 disables the HTTP endpoint
    health_interval_s: float = 30.0
    enable_powermetrics: bool = True  # only works when running as root (sudo)


@dataclass
class CameraSourceSettings:
    """Where the camera list (and each camera's use case) comes from."""
    source: str = "yaml"                          # yaml (config/cameras.yaml) | db (your stored procedure)
    sql: str = "SELECT * FROM public.fn_get_device_configuration()"   # db: any query / SP call, one row per camera
    poll_s: float = 30.0                          # re-check the camera list this often (both sources)
    notify_channel: str = "camera_changed"        # PostgreSQL: LISTEN here for instant reloads ("" = off)
    defaults: Dict[str, Any] = field(default_factory=dict)   # cameras.yaml `defaults:`, applied to DB rows
    yaml_path: Path = CONFIG_DIR / "cameras.yaml"


@dataclass
class ApiSettings:
    """HTTP API for the frontend (python main.py api, or run --api)."""
    host: str = "127.0.0.1"          # 0.0.0.0 to accept requests from other machines
    port: int = 8090
    api_key: str = ""                # when set, every request needs the header X-API-Key: <key>
    cors_origins: List[str] = field(default_factory=list)   # browser origins allowed to call the API
    timezone: str = "Asia/Kolkata"   # default time zone for dashboard days / hours
    auto_setup: bool = True          # create camera_config + the SPs on start if they're missing


@dataclass
class Settings:
    models: ModelSettings = field(default_factory=ModelSettings)
    detection: DetectionSettings = field(default_factory=DetectionSettings)
    person: PersonSettings = field(default_factory=PersonSettings)
    recognition: RecognitionSettings = field(default_factory=RecognitionSettings)
    ingestion: IngestionSettings = field(default_factory=IngestionSettings)
    storage: StorageSettings = field(default_factory=StorageSettings)
    monitoring: MonitoringSettings = field(default_factory=MonitoringSettings)
    cameras: List[CameraConfig] = field(default_factory=list)
    camera_source: CameraSourceSettings = field(default_factory=CameraSourceSettings)
    api: ApiSettings = field(default_factory=ApiSettings)
    log_level: str = "INFO"
    logging_config: Path = CONFIG_DIR / "logging.yaml"

    @property
    def enabled_cameras(self) -> List[CameraConfig]:
        return [c for c in self.cameras if c.enabled]

    @property
    def face_cameras(self) -> List[CameraConfig]:
        return [c for c in self.enabled_cameras if c.runs_face]

    @property
    def person_cameras(self) -> List[CameraConfig]:
        return [c for c in self.enabled_cameras if c.runs_person]


# --------------------------------------------------------------------------- loaders
def load_camera_defaults(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return dict((yaml.safe_load(fh) or {}).get("defaults") or {})


def load_cameras(path: Path) -> List[CameraConfig]:
    if not path.exists():
        log.warning("Camera config %s not found", path)
        return []
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    defaults = raw.get("defaults") or {}
    cams = [CameraConfig.from_dict(c, defaults) for c in (raw.get("cameras") or [])]
    ids = [c.camera_id for c in cams]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate camera_id(s) in {path}: {sorted(dupes)}")
    return cams


def load_settings(env_file: Optional[Path] = None, cameras_file: Optional[Path] = None) -> Settings:
    env_file = env_file or (PROJECT_ROOT / ".env")
    if load_dotenv is not None and env_file.exists():
        load_dotenv(env_file, override=False)

    m = ModelSettings(
        scrfd_path=_env_path("FR_SCRFD_MODEL", DEFAULT_SCRFD_PATH),
        arcface_path=_env_path("FR_ARCFACE_MODEL", DEFAULT_ARCFACE_PATH),
        compute_units=_env("FR_COMPUTE_UNITS", "CPU_AND_NE"),
        cache_compiled=_env("FR_CACHE_COMPILED", True),
        scrfd_input_mean=_env("FR_SCRFD_MEAN", SCRFD_INPUT_MEAN),
        scrfd_input_std=_env("FR_SCRFD_STD", SCRFD_INPUT_STD),
        scrfd_apply_sigmoid=_env("FR_SCRFD_SIGMOID", False),
        arcface_input_mean=_env("FR_ARCFACE_MEAN", ARCFACE_INPUT_MEAN),
        arcface_input_std=_env("FR_ARCFACE_STD", ARCFACE_INPUT_STD),
        warmup_iterations=_env("FR_WARMUP_ITERS", 3),
        yolo_path=_env_path("FR_YOLO_MODEL", DEFAULT_YOLO_PATH),
    )
    p = PersonSettings(
        conf_threshold=_env("FR_PERSON_CONF", 0.4),
        nms_threshold=_env("FR_PERSON_NMS", 0.45),
        class_id=_env("FR_PERSON_CLASS_ID", COCO_PERSON_CLASS_ID),
        num_classes=_env("FR_PERSON_NUM_CLASSES", 80),
        input_std=_env("FR_PERSON_INPUT_STD", 255.0),
        min_box_height=_env("FR_PERSON_MIN_HEIGHT", 30),
        batch_size=_env("FR_PERSON_BATCH", 4),
        max_wait_ms=_env("FR_PERSON_MAX_WAIT_MS", 8.0),
        max_fps=_env("FR_PERSON_MAX_FPS", 0.0),
        track_iou=_env("FR_PERSON_TRACK_IOU", 0.3),
        track_max_age_s=_env("FR_PERSON_TRACK_MAX_AGE_S", 2.0),
        track_min_hits=_env("FR_PERSON_TRACK_MIN_HITS", 2),
        flush_interval_s=_env("FR_PERSON_FLUSH_S", 60.0),
        line_anchor=_env("FR_PERSON_LINE_ANCHOR", _env("FR_PERSON_ZONE_ANCHOR", "bottom")),
        line_min_frames=_env("FR_PERSON_LINE_MIN_FRAMES", _env("FR_PERSON_ZONE_MIN_FRAMES", 2)),
        line_min_distance=_env("FR_PERSON_LINE_MIN_DISTANCE", 4.0),
        in_direction=_env("FR_PERSON_IN_DIRECTION", "down").lower(),
        init_sql=_env("FR_PERSON_INIT_SQL", PersonSettings.init_sql),
        update_sql=_env("FR_PERSON_UPDATE_SQL", PersonSettings.update_sql),
        count_table=_env("FR_PERSON_COUNT_TABLE", PersonSettings.count_table),
    )
    d = DetectionSettings(
        det_threshold=_env("FR_DET_THRESHOLD", 0.5),
        nms_threshold=_env("FR_NMS_THRESHOLD", 0.4),
        min_face_size=_env("FR_MIN_FACE_SIZE", 40),
        max_faces_per_frame=_env("FR_MAX_FACES", 20),
        batch_size=_env("FR_DET_BATCH", 4),
        max_wait_ms=_env("FR_DET_MAX_WAIT_MS", 8.0),
        max_frame_age_s=_env("FR_MAX_FRAME_AGE_S", 1.0),
    )
    r = RecognitionSettings(
        match_threshold=_env("FR_MATCH_THRESHOLD", 0.45),
        match_margin=_env("FR_MATCH_MARGIN", 0.05),
        batch_size=_env("FR_REC_BATCH", 16),
        max_wait_ms=_env("FR_REC_MAX_WAIT_MS", 10.0),
        queue_size=_env("FR_REC_QUEUE", 256),
        blur_threshold=_env("FR_BLUR_THRESHOLD", 40.0),
        max_yaw_ratio=_env("FR_MAX_YAW", 0.45),
        min_quality=_env("FR_MIN_QUALITY", 0.25),
        event_cooldown_s=_env("FR_EVENT_COOLDOWN_S", 10.0),
        record_unknown=_env("FR_RECORD_UNKNOWN", True),
        unknown_cooldown_s=_env("FR_UNKNOWN_COOLDOWN_S", 30.0),
        unknown_similarity=_env("FR_UNKNOWN_SIMILARITY", 0.5),
        track_iou=_env("FR_TRACK_IOU", 0.45),
        track_reverify_s=_env("FR_TRACK_REVERIFY_S", 2.0),
        gallery_reload_s=_env("FR_GALLERY_RELOAD_S", 30.0),
    )
    i = IngestionSettings(
        backend=_env("FR_INGESTION_BACKEND", "auto"),
        reconnect_delay_s=_env("FR_RECONNECT_DELAY_S", 2.0),
        reconnect_max_delay_s=_env("FR_RECONNECT_MAX_DELAY_S", 60.0),
        stale_timeout_s=_env("FR_STALE_TIMEOUT_S", 15.0),
        queue_depth=_env("FR_QUEUE_DEPTH", 1),
    )
    s = StorageSettings(
        db_path=_env_path("FR_DB_PATH", DEFAULT_DB_PATH),
        save_snapshots=_env("FR_SAVE_SNAPSHOTS", False),
        snapshot_quality=_env("FR_SNAPSHOT_QUALITY", 85),
        event_flush_interval_s=_env("FR_EVENT_FLUSH_S", 1.0),
        event_batch_size=_env("FR_EVENT_BATCH", 100),
        event_queue_size=_env("FR_EVENT_QUEUE", 5000),
        retention_days=_env("FR_RETENTION_DAYS", 90),
        database_url=str(_env("FR_DATABASE_URL", "")),
        pg_host=str(_env("FR_PG_HOST", "")),
        pg_port=_env("FR_PG_PORT", 5432),
        pg_user=str(_env("FR_PG_USER", "")),
        pg_password=os.environ.get("FR_PG_PASSWORD", ""),   # not stripped: spaces may be part of it
        pg_database=str(_env("FR_PG_DATABASE", "")),
        pg_sslmode=str(_env("FR_PG_SSLMODE", "")),
        pg_pool_min=_env("FR_PG_POOL_MIN", 1),
        pg_pool_max=_env("FR_PG_POOL_MAX", 10),
        pg_connect_timeout_s=_env("FR_PG_CONNECT_TIMEOUT_S", 15.0),
    )
    mon = MonitoringSettings(
        health_host=_env("FR_HEALTH_HOST", "127.0.0.1"),
        health_port=_env("FR_HEALTH_PORT", 8085),
        health_interval_s=_env("FR_HEALTH_INTERVAL_S", 30.0),
        enable_powermetrics=_env("FR_POWERMETRICS", True),
    )
    cams_path = cameras_file or _env_path("FR_CAMERAS_FILE", CONFIG_DIR / "cameras.yaml")
    settings = Settings(
        models=m,
        detection=d,
        person=p,
        recognition=r,
        ingestion=i,
        storage=s,
        monitoring=mon,
        cameras=load_cameras(cams_path),
        camera_source=CameraSourceSettings(
            source=str(_env("FR_CAMERA_SOURCE", "yaml")).lower(),
            sql=_env("FR_CAMERA_SQL", CameraSourceSettings.sql),
            poll_s=_env("FR_CAMERA_POLL_S", 30.0),
            notify_channel=_env("FR_CAMERA_NOTIFY_CHANNEL", "camera_changed"),
            defaults=load_camera_defaults(cams_path),
            yaml_path=cams_path,
        ),
        api=ApiSettings(
            host=str(_env("FR_API_HOST", "127.0.0.1")),
            port=_env("FR_API_PORT", 8090),
            api_key=os.environ.get("FR_API_KEY", "").strip(),
            cors_origins=[o.strip() for o in str(_env("FR_API_CORS_ORIGINS", "")).split(",") if o.strip()],
            timezone=str(_env("FR_API_TIMEZONE", "Asia/Kolkata")),
            auto_setup=_env("FR_API_AUTO_SETUP", True),
        ),
        log_level=str(_env("FR_LOG_LEVEL", "INFO")).upper(),
        logging_config=_env_path("FR_LOGGING_CONFIG", CONFIG_DIR / "logging.yaml"),
    )
    _validate(settings)
    return settings


def _validate(s: Settings) -> None:
    if s.ingestion.backend not in ("auto", "gstreamer", "opencv"):
        raise ValueError("FR_INGESTION_BACKEND must be auto, gstreamer or opencv")
    if not 0.0 < s.recognition.match_threshold < 1.0:
        raise ValueError("FR_MATCH_THRESHOLD must be in (0, 1)")
    if not 0.0 < s.person.conf_threshold < 1.0:
        raise ValueError("FR_PERSON_CONF must be in (0, 1)")
    if s.person.flush_interval_s <= 0:
        raise ValueError("FR_PERSON_FLUSH_S must be > 0")
    if s.camera_source.source not in ("yaml", "db"):
        raise ValueError("FR_CAMERA_SOURCE must be yaml or db")
    if s.camera_source.notify_channel and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,62}",
                                                            s.camera_source.notify_channel):
        raise ValueError("FR_CAMERA_NOTIFY_CHANNEL must be a plain identifier (letters, digits, _)")
    if not 0 < s.api.port < 65536:
        raise ValueError("FR_API_PORT must be 1-65535")
    if s.camera_source.poll_s <= 0:
        raise ValueError("FR_CAMERA_POLL_S must be > 0")
    if s.person.line_anchor not in ("bottom", "center"):
        raise ValueError("FR_PERSON_LINE_ANCHOR must be bottom or center")
    if s.person.in_direction not in ("down", "up", "right", "left"):
        raise ValueError("FR_PERSON_IN_DIRECTION must be down, up, right or left")
    if s.detection.batch_size < 1 or s.recognition.batch_size < 1 or s.person.batch_size < 1:
        raise ValueError("batch sizes must be >= 1")
    if s.storage.database_url and not s.storage.database_url.startswith(("postgres://", "postgresql://")):
        raise ValueError("FR_DATABASE_URL must start with postgresql:// (leave it empty to use SQLite)")
    for c in s.cameras:
        if c.fps <= 0:
            raise ValueError(f"camera {c.camera_id}: fps must be > 0")
