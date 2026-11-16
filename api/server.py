"""HTTP API for the frontend: cameras + use cases in, dashboard numbers out.

    python main.py api              # API only            -> http://127.0.0.1:8090/docs
    python main.py run --api        # facereco + API in one process

Both ways of driving facereco end in the same table, so you can use either or both:

    frontend --HTTP--> this API --sp_upsert_camera--> camera_config --NOTIFY--> facereco starts the use case
    your backend ----- SELECT sp_upsert_camera(...) ----^

Every camera write goes through sp_upsert_camera / sp_delete_camera, exactly what a backend calling
the SPs directly would run. Dashboard endpoints call the sp_dash_* procedures.
"""
from __future__ import annotations

import hmac
import json
import logging
import urllib.request
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from urllib.parse import quote, unquote, urlsplit, urlunsplit

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, status
from fastapi import Path as PathParam
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from config.settings import Settings
from services.camera_source import row_to_camera

log = logging.getLogger(__name__)

MASK = "***"
URL_SCHEMES = ("rtsp", "rtsps", "http", "https")
CAMERA_ID_PATTERN = r"^[A-Za-z0-9_.-]{1,64}$"
_CAMERA_COLUMNS = ("camera_id, name, location, url, run_face, run_person, count_zone, roi, fps, width, height, "
                   "codec, enabled, updated_at")


# ------------------------------------------------------------------------------------------- URL helpers
def mask_password(url: str) -> str:
    """rtsp://admin:secret@10.0.0.5/s1 -> rtsp://admin:***@10.0.0.5/s1 (username kept so the form can show it)."""
    parts = urlsplit(url)
    if parts.password is None:
        return url
    user = parts.username or ""
    host = parts.netloc.rsplit("@", 1)[1]
    return urlunsplit(parts._replace(netloc=f"{user}:{MASK}@{host}"))


def restore_password(new_url: str, old_url: Optional[str]) -> str:
    """A URL sent back with the masked password (***) keeps the stored password."""
    parts = urlsplit(new_url)
    if parts.password != MASK:
        return new_url
    old_pw = urlsplit(old_url).password if old_url else None
    if old_pw is None:
        raise HTTPException(422, "url: the password is '***' but no password is stored for this camera - "
                                 "send the real password")
    host = parts.netloc.rsplit("@", 1)[1]
    pw = quote(unquote(old_pw), safe="")
    return urlunsplit(parts._replace(netloc=f"{parts.username or ''}:{pw}@{host}"))


# ------------------------------------------------------------------------------------------- request models


class CameraFields(BaseModel):
    """Everything about a camera except its id. Use cases: run_face / run_person."""
    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = Field(None, min_length=1, max_length=200, examples=["Main Gate"])
    url: Optional[str] = Field(None, max_length=2000,
                               examples=["rtsp://admin:pass@192.168.1.101:554/Streaming/Channels/102"])
    run_face: Optional[bool] = Field(None, description="use case: face recognition")
    run_person: Optional[bool] = Field(None, description="use case: person detection + IN/OUT counting")
    roi: Optional[List[float]] = Field(
        None, description="IN/OUT count line [x1, y1, x2, y2] (its two ends): pixels or 0..1 fractions of the frame",
        examples=[[0.0, 0.6, 1.0, 0.6]])
    location: Optional[str] = Field(None, max_length=200)
    enabled: Optional[bool] = Field(None, description="false = camera stopped (kept in the list)")
    fps: Optional[float] = Field(None, gt=0, le=30, description="frames/s analysed (null = default)")
    width: Optional[int] = Field(None, gt=0, le=7680)
    height: Optional[int] = Field(None, gt=0, le=4320)
    codec: Optional[Literal["h264", "h265", "auto"]] = None

    @field_validator("url")
    @classmethod
    def _url(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.strip()
        parts = urlsplit(v)
        if parts.scheme.lower() not in URL_SCHEMES or not parts.hostname:
            raise ValueError(f"must be a camera stream URL ({', '.join(s + '://' for s in URL_SCHEMES)})")
        return v

    @field_validator("roi")
    @classmethod
    def _roi(cls, v):
        if v is None:
            return v
        if len(v) != 4 or min(v) < 0:
            raise ValueError("must be a line [x1, y1, x2, y2] with coordinates >= 0")
        if v[0] == v[2] and v[1] == v[3]:
            raise ValueError("the line's two ends are the same point")
        return v


class CameraBody(CameraFields):
    """PUT /cameras/{id}: the full camera. Fields left out get their defaults."""
    name: str = Field(..., min_length=1, max_length=200, examples=["Main Gate"])
    url: str = Field(..., max_length=2000, examples=["rtsp://admin:pass@192.168.1.101:554/Streaming/Channels/102"])
    run_face: bool = False
    run_person: bool = False
    location: str = ""
    enabled: bool = True


class CameraCreate(CameraBody):
    """POST /cameras: like PUT, with the id in the body."""
    camera_id: str = Field(..., pattern=CAMERA_ID_PATTERN, examples=["gate1"])


# ------------------------------------------------------------------------------------------- app
def create_app(settings: Settings, db=None) -> FastAPI:
    """`db` may be passed in (tests); otherwise the PostgreSQL in .env is opened on start-up."""
    api_cfg = settings.api
    state: Dict[str, Any] = {"db": db}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        own = state["db"] is None
        if own:
            from database.connection import open_database

            state["db"] = open_database(settings.storage)
        d = state["db"]
        if getattr(d, "backend", "sqlite") != "postgres":
            raise RuntimeError("The API needs PostgreSQL: set FR_DATABASE_URL or FR_PG_HOST in .env")
        if api_cfg.auto_setup:
            from database.sql_setup import ensure_sql_objects

            ensure_sql_objects(d)
        if settings.camera_source.source != "db":
            log.warning("FR_CAMERA_SOURCE=%s: facereco reads cameras.yaml, so cameras saved through the API "
                        "won't start anything. Set FR_CAMERA_SOURCE=db.", settings.camera_source.source)
        if not api_cfg.api_key and api_cfg.host not in ("127.0.0.1", "localhost", "::1"):
            log.warning("API listening on %s without FR_API_KEY - anyone on the network can change cameras",
                        api_cfg.host)
        log.info("API ready on http://%s:%d (docs: /docs)", api_cfg.host, api_cfg.port)
        try:
            yield
        finally:
            if own:
                d.close()

    app = FastAPI(
        title="facereco API",
        version="1.0",
        lifespan=lifespan,
        description="Cameras and their use cases (face recognition / person counting) in, dashboard "
                    "numbers out. Saving a camera starts its use cases within ~1 s.",
    )
    if api_cfg.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=api_cfg.cors_origins, allow_methods=["*"],
                           allow_headers=["*"])

    def database():
        return state["db"]

    def check_key(request: Request) -> None:
        if not api_cfg.api_key:
            return
        given = request.headers.get("x-api-key", "")
        if not hmac.compare_digest(given.encode(), api_cfg.api_key.encode()):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing or wrong X-API-Key header")

    try:
        import psycopg
        from psycopg_pool import PoolTimeout
    except ImportError:  # pragma: no cover - psycopg is in requirements.txt
        psycopg = None

    if psycopg is not None:
        @app.exception_handler(psycopg.errors.DataError)
        async def _bad_value(_request: Request, exc: Exception):
            # e.g. an unknown time zone: the database rejected a value the client sent
            return JSONResponse({"detail": str(exc).splitlines()[0]}, status_code=400)

        async def _db_down(_request: Request, exc: Exception):
            log.error("Database unavailable: %s", exc)
            return JSONResponse({"detail": "database unavailable"}, status_code=503)

        app.add_exception_handler(psycopg.OperationalError, _db_down)
        app.add_exception_handler(PoolTimeout, _db_down)

    # ------------------------------------------------------------------ helpers
    def _get_row(d, camera_id: str) -> Optional[dict]:
        row = d.fetchone(f"SELECT {_CAMERA_COLUMNS} FROM camera_config WHERE camera_id = ?", (camera_id,))
        return dict(row) if row else None

    def _public(row: dict, live: Optional[dict] = None) -> dict:
        out = dict(row)
        out["url"] = mask_password(out["url"])
        if not out.get("run_person"):
            out["roi"] = None
        out["use_cases"] = [n for n, on in (("face", out["run_face"]), ("person", out["run_person"])) if on]
        running = out["enabled"] and bool(out["use_cases"])
        out["state"] = "stopped" if not running else (live or {}).get("status", "starting")
        if live is not None:
            out["live"] = {k: live.get(k) for k in ("status", "fps", "seconds_since_frame", "reconnects",
                                                     "last_error")}
        return out

    def _facereco(path: str) -> Optional[Any]:
        """GET from facereco's health server; None when facereco isn't running."""
        mon = settings.monitoring
        if not mon.health_port:
            return None
        host = "127.0.0.1" if mon.health_host in ("0.0.0.0", "") else mon.health_host
        try:
            with urllib.request.urlopen(f"http://{host}:{mon.health_port}{path}", timeout=1.5) as r:
                return json.loads(r.read())
        except Exception:
            return None

    def _live_by_camera() -> Optional[Dict[str, dict]]:
        data = _facereco("/cameras")
        if data is None:
            return None
        return {c["camera_id"]: c for c in data.get("cameras", [])}

    def _save(d, camera_id: str, f: Dict[str, Any]) -> None:
        """Validate the merged camera the same way facereco will, then sp_upsert_camera."""
        if not f["run_person"]:
            f["roi"] = None
        check = {**f, "camera_id": camera_id, "enabled": True,
                 "tasks": [t for t, on in (("face", f["run_face"]), ("person", f["run_person"])) if on] or ["face"]}
        try:
            row_to_camera(check, settings.camera_source.defaults)
        except (ValueError, TypeError) as exc:
            raise HTTPException(422, str(exc))
        d.execute(
            "SELECT sp_upsert_camera(p_camera_id => ?::text, p_name => ?::text, p_url => ?::text, "
            "p_run_face => ?::boolean, p_run_person => ?::boolean, p_count_zone => ?::jsonb, "
            "p_location => ?::text, p_enabled => ?::boolean, p_fps => ?::real, p_width => ?::integer, "
            "p_height => ?::integer, p_roi => ?::jsonb, p_codec => ?::text)",
            (camera_id, f["name"], f["url"], f["run_face"], f["run_person"],
             None, f.get("location") or "",                       # count_zone: replaced by the roi line
             f["enabled"], f.get("fps"), f.get("width"), f.get("height"),
             json.dumps(f["roi"]) if f.get("roi") else None, f.get("codec")))

    def _put(d, camera_id: str, body: CameraBody, response: Response) -> dict:
        old = _get_row(d, camera_id)
        fields = body.model_dump()
        fields["url"] = restore_password(fields["url"], old["url"] if old else None)
        _save(d, camera_id, fields)
        response.status_code = status.HTTP_200_OK if old else status.HTTP_201_CREATED
        return _public(_get_row(d, camera_id))

    def _patch(d, camera_id: str, changes: Dict[str, Any]) -> dict:
        old = _get_row(d, camera_id)
        if old is None:
            raise HTTPException(404, f"camera {camera_id} not found")
        fields = {k: old[k] for k in CameraFields.model_fields}
        if "url" in changes and changes["url"] is not None:
            changes["url"] = restore_password(changes["url"], old["url"])
        for k, v in changes.items():
            if v is None and k in ("name", "url", "run_face", "run_person", "enabled"):
                raise HTTPException(422, f"{k} can't be null")
            fields[k] = v
        _save(d, camera_id, fields)
        return _public(_get_row(d, camera_id))

    def _cameras_param(cameras: Optional[str]) -> Optional[List[str]]:
        ids = [c.strip() for c in (cameras or "").split(",") if c.strip()]
        return ids or None

    # ------------------------------------------------------------------ status
    @app.get("/health", tags=["status"], summary="Is the API up (no key needed)")
    def health():
        return {"ok": True}

    @app.get("/status", tags=["status"], dependencies=[Depends(check_key)],
             summary="API, database and facereco status")
    def get_status(d=Depends(database)):
        try:
            d.fetchone("SELECT 1 AS ok")
            db_ok = True
        except Exception:
            db_ok = False
        fr = _facereco("/health")
        return {"api": "ok", "database": "ok" if db_ok else "unavailable",
                "facereco": fr if fr is not None else "idle (no running cameras) or not started",
                "camera_source": settings.camera_source.source}

    # ------------------------------------------------------------------ cameras
    @app.get("/cameras", tags=["cameras"], dependencies=[Depends(check_key)],
             summary="All cameras with use cases and live state (passwords masked)")
    def list_cameras(d=Depends(database)):
        live = _live_by_camera()
        rows = d.fetchall(f"SELECT {_CAMERA_COLUMNS} FROM camera_config ORDER BY camera_id")
        return {"facereco_running": live is not None,
                "cameras": [_public(dict(r), (live or {}).get(r["camera_id"]) if live is not None else None)
                            for r in rows]}

    @app.get("/cameras/{camera_id}", tags=["cameras"], dependencies=[Depends(check_key)])
    def get_camera(camera_id: str, d=Depends(database)):
        row = _get_row(d, camera_id)
        if row is None:
            raise HTTPException(404, f"camera {camera_id} not found")
        live = _live_by_camera()
        return _public(row, live.get(camera_id) if live is not None else None)

    @app.post("/cameras", tags=["cameras"], dependencies=[Depends(check_key)], status_code=201,
              summary="Add (or replace) a camera; its use cases start automatically")
    def create_camera(body: CameraCreate, response: Response, d=Depends(database)):
        return _put(d, body.camera_id, CameraBody(**body.model_dump(exclude={"camera_id"})), response)

    @app.put("/cameras/{camera_id}", tags=["cameras"], dependencies=[Depends(check_key)],
             summary="Add or replace a camera (201 = created, 200 = updated)")
    def put_camera(body: CameraBody, response: Response,
                   camera_id: str = PathParam(..., pattern=CAMERA_ID_PATTERN), d=Depends(database)):
        return _put(d, camera_id, body, response)

    @app.patch("/cameras/{camera_id}", tags=["cameras"], dependencies=[Depends(check_key)],
               summary="Change only the fields sent, e.g. {\"run_face\": true}")
    def patch_camera(camera_id: str, body: CameraFields, d=Depends(database)):
        return _patch(d, camera_id, body.model_dump(exclude_unset=True))

    @app.post("/cameras/{camera_id}/start", tags=["cameras"], dependencies=[Depends(check_key)],
              summary="Start the camera (enabled = true)")
    def start_camera(camera_id: str, d=Depends(database)):
        return _patch(d, camera_id, {"enabled": True})

    @app.post("/cameras/{camera_id}/stop", tags=["cameras"], dependencies=[Depends(check_key)],
              summary="Stop the camera (enabled = false; settings are kept)")
    def stop_camera(camera_id: str, d=Depends(database)):
        return _patch(d, camera_id, {"enabled": False})

    @app.delete("/cameras/{camera_id}", tags=["cameras"], dependencies=[Depends(check_key)], status_code=204)
    def delete_camera(camera_id: str, d=Depends(database)):
        if _get_row(d, camera_id) is None:
            raise HTTPException(404, f"camera {camera_id} not found")
        d.execute("SELECT sp_delete_camera(?::text)", (camera_id,))
        return Response(status_code=204)

    # ------------------------------------------------------------------ dashboard (person counting)
    tz_q = Query(None, description=f"IANA time zone (default {api_cfg.timezone})")
    cams_q = Query(None, description="comma-separated camera ids (default: all)")

    @app.get("/dashboard/summary", tags=["dashboard"], dependencies=[Depends(check_key)],
             summary="Tiles for one day: IN, OUT, inside now, unique people, peak, live cameras")
    def dash_summary(day: Optional[date] = None, cameras: Optional[str] = cams_q, tz: Optional[str] = tz_q,
                     d=Depends(database)):
        row = d.fetchone("SELECT * FROM sp_dash_summary(?::date, ?::text[], ?::text)",
                         (day, _cameras_param(cameras), tz or api_cfg.timezone))
        return dict(row) if row else {}

    @app.get("/dashboard/hourly", tags=["dashboard"], dependencies=[Depends(check_key)],
             summary="IN / OUT per hour for one day (24 rows)")
    def dash_hourly(day: Optional[date] = None, cameras: Optional[str] = cams_q, tz: Optional[str] = tz_q,
                    d=Depends(database)):
        return d.fetchall("SELECT * FROM sp_dash_hourly_in_out(?::date, ?::text[], ?::text)",
                          (day, _cameras_param(cameras), tz or api_cfg.timezone))

    @app.get("/dashboard/daily", tags=["dashboard"], dependencies=[Depends(check_key)],
             summary="One row per day: IN, OUT, unique people, peak (default: last 7 days)")
    def dash_daily(date_from: Optional[date] = Query(None, alias="from"),
                   date_to: Optional[date] = Query(None, alias="to"),
                   cameras: Optional[str] = cams_q, tz: Optional[str] = tz_q, d=Depends(database)):
        if date_from and date_to and (date_to - date_from).days > 366:
            raise HTTPException(422, "range is limited to 366 days")
        if date_from and date_to and date_from > date_to:
            raise HTTPException(422, "'from' is after 'to'")
        return d.fetchall("SELECT * FROM sp_dash_daily_totals(?::date, ?::date, ?::text[], ?::text)",
                          (date_from, date_to, _cameras_param(cameras), tz or api_cfg.timezone))

    @app.get("/dashboard/cameras", tags=["dashboard"], dependencies=[Depends(check_key)],
             summary="Per-camera numbers for one day")
    def dash_cameras(day: Optional[date] = None, tz: Optional[str] = tz_q, d=Depends(database)):
        return d.fetchall("SELECT * FROM sp_dash_camera_counts(?::date, ?::text)", (day, tz or api_cfg.timezone))

    # ------------------------------------------------------------------ face recognition events
    @app.get("/events", tags=["dashboard"], dependencies=[Depends(check_key)],
             summary="Recognised faces, newest first")
    def events(since: Optional[datetime] = None, until: Optional[datetime] = None,
               camera: Optional[str] = None, known_only: bool = False,
               limit: int = Query(100, ge=1, le=1000), d=Depends(database)):
        def epoch(t: Optional[datetime]) -> Optional[float]:
            if t is None:
                return None
            return (t if t.tzinfo else t.replace(tzinfo=timezone.utc)).timestamp()

        return d.fetchall(
            "SELECT e.id, e.camera_id, e.person_id, p.name AS person_name, e.label, e.similarity, "
            "       to_timestamp(e.ts) AS time "
            "FROM events e LEFT JOIN persons p ON p.id = e.person_id "
            "WHERE (?::float8 IS NULL OR e.ts >= ?::float8) AND (?::float8 IS NULL OR e.ts < ?::float8) "
            "  AND (?::text IS NULL OR e.camera_id = ?::text) AND (NOT ?::boolean OR e.person_id IS NOT NULL) "
            "ORDER BY e.ts DESC LIMIT ?::integer",
            (epoch(since), epoch(since), epoch(until), epoch(until), camera, camera, known_only, limit))

    return app


# ------------------------------------------------------------------------------------------- serving
def make_server(settings: Settings, host: Optional[str] = None, port: Optional[int] = None):
    import uvicorn

    if host:
        settings.api.host = host
    if port:
        settings.api.port = port
    config = uvicorn.Config(create_app(settings), host=settings.api.host, port=settings.api.port,
                            log_config=None, access_log=False)
    return uvicorn.Server(config)


def serve(settings: Settings, host: Optional[str] = None, port: Optional[int] = None) -> None:
    """Blocking: python main.py api."""
    make_server(settings, host, port).run()


def serve_in_thread(settings: Settings):
    """python main.py run --api: the API next to the pipelines. Returns the server (call .should_exit = True)."""
    import threading

    server = make_server(settings)                     # uvicorn only installs signal handlers in the main
    threading.Thread(target=server.run, name="api-http", daemon=True).start()   # thread: Ctrl+C stays with run
    return server
