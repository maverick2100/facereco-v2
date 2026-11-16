"""One GStreamer pipeline per camera.

    rtspsrc -> depay -> parse -> vtdec (VideoToolbox, hardware)
            -> videorate (drop to N fps) -> videoscale -> videoconvert(BGR)
            -> appsink (max-buffers=1, drop=true)

Frames are decoded by the M4 media engine; only N fps are colour-converted
and handed to Python, which keeps CPU usage low.
"""
from __future__ import annotations

import logging
import re
import threading
from fractions import Fraction
from typing import Any, Callable, Optional
from urllib.parse import quote

import numpy as np

from config.settings import CameraConfig, IngestionSettings
from ingestion.appsink_handler import AppSinkHandler
from ingestion.camera_health import STARTING, STOPPED, CameraHealth

log = logging.getLogger(__name__)

SINK_NAME = "sink"


def _split_credentials(url: str):
    """Split 'scheme://user:pass@host/path' -> (scheme, creds|None, host_and_path).

    Done by hand because passwords often contain '#', '/', '?' or '@', which
    confuse urllib's parser.
    """
    scheme, sep, rest = url.partition("://")
    if not sep or "@" not in rest:
        return None, None, url
    creds, _, host = rest.rpartition("@")
    return scheme, creds, host


def _quote_credentials(url: str) -> str:
    """Percent-encode user/password so special characters survive parsing."""
    scheme, creds, host = _split_credentials(url)
    if creds is None:
        return url
    user, _, pwd = creds.partition(":")
    creds = quote(user, safe="%") + (":" + quote(pwd, safe="%") if pwd else "")
    return f"{scheme}://{creds}@{host}"


def mask_url(url: str) -> str:
    scheme, creds, host = _split_credentials(url)
    return url if creds is None else f"{scheme}://***@{host}"


_REASON = re.compile(r"reason ([a-z-]+)")


def explain_error(message: str, debug: Optional[str], codec: str = "h264") -> str:
    """GStreamer's message plus the reason buried in its debug text, and what to do about it.

    "Internal data stream error." alone says nothing; the debug string says e.g.
    "streaming stopped, reason not-linked (-1)", which on an RTSP camera almost always means
    the stream's codec isn't the one the pipeline was built for.
    """
    m = _REASON.search(debug or "")
    if not m:
        return message
    reason = m.group(1)
    out = f"{message} (reason: {reason})"
    if reason in ("not-linked", "not-negotiated"):
        other = "h265" if codec.lower() == "h264" else "h264"
        out += (f" - the camera's video codec probably isn't {codec}: set codec: {other} or codec: auto "
                f"for this camera")
    return out


def build_pipeline_description(cam: CameraConfig) -> str:
    url = cam.url.strip()
    fr = Fraction(cam.fps).limit_denominator(100)
    sync = "false"
    # Live sources: a leaky queue drops stale frames instead of building latency. Files must NOT leak:
    # with sync=true the sink paces playback by back-pressure, and a leaky queue throws those buffers
    # away instead, which stalls the stream after ~2 frames.
    queue = "queue max-size-buffers=2 leaky=downstream"

    if url.startswith("test://"):
        pattern = url[len("test://"):] or "ball"
        head = (
            f"videotestsrc is-live=true pattern={pattern} ! "
            f"video/x-raw,width={cam.width},height={cam.height},framerate=25/1"
        )
    elif url.startswith("file://"):
        head = f'uridecodebin uri="{url}"'
        sync = "true"  # play files in real time
        queue = "queue max-size-buffers=2"
    else:
        src = (
            f'rtspsrc location="{_quote_credentials(url)}" protocols={cam.transport} '
            f"latency={cam.latency_ms} drop-on-latency=true "
            f"timeout=5000000 tcp-timeout=5000000"
        )
        codec = cam.codec.lower()
        if codec == "h264":
            head = f"{src} ! rtph264depay ! h264parse ! {cam.decoder}"
        elif codec in ("h265", "hevc"):
            head = f"{src} ! rtph265depay ! h265parse ! {cam.decoder}"
        else:  # auto: let decodebin pick (vtdec has primary rank on macOS)
            head = f"{src} ! decodebin"

    tail = (
        f"{queue} ! "
        f"videorate drop-only=true ! video/x-raw,framerate={fr.numerator}/{fr.denominator} ! "
        f"videoscale ! video/x-raw,width={cam.width},height={cam.height} ! "
        "videoconvert n-threads=1 ! video/x-raw,format=BGR ! "
        f"appsink name={SINK_NAME} emit-signals=true max-buffers=1 drop=true sync={sync}"
    )
    return f"{head} ! {tail}"


class GstCameraPipeline:
    def __init__(self, camera: CameraConfig, on_frame: Callable[[np.ndarray], None], health: CameraHealth,
                 settings: IngestionSettings, Gst: Any) -> None:
        self.camera = camera
        self.health = health
        self.settings = settings
        self._Gst = Gst
        self._on_frame_cb = on_frame
        self._pipeline: Optional[Any] = None
        self._bus: Optional[Any] = None
        self._lock = threading.Lock()
        self._restarting = False
        self._stop = threading.Event()
        self._backoff = settings.reconnect_delay_s
        self._got_frame = False

    @property
    def restarting(self) -> bool:
        return self._restarting

    # ------------------------------------------------------------------ frames
    def _on_frame(self, img: np.ndarray) -> None:
        if not self._got_frame:
            self._got_frame = True
            self._backoff = self.settings.reconnect_delay_s
            log.info("[%s] streaming %dx%d", self.camera.camera_id, img.shape[1], img.shape[0])
        self._on_frame_cb(img)

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        try:
            self._build_and_play()
        except Exception as exc:
            log.error("[%s] failed to start: %s", self.camera.camera_id, exc)
            self.health.on_error(str(exc))
            self.schedule_restart("start failed")

    def _build_and_play(self) -> None:
        Gst = self._Gst
        desc = build_pipeline_description(self.camera)
        log.debug("[%s] pipeline: %s", self.camera.camera_id, desc.replace(self.camera.url, mask_url(self.camera.url)))
        pipeline = Gst.parse_launch(desc)
        pipeline.set_name(f"cam-{self.camera.camera_id}")
        sink = pipeline.get_by_name(SINK_NAME)
        AppSinkHandler(self.camera.camera_id, self._on_frame, Gst).attach(sink)
        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)
        self._got_frame = False
        self.health.mark_restarted()
        self.health.set_status(STARTING)
        with self._lock:
            self._pipeline, self._bus = pipeline, bus
        if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("pipeline refused to go to PLAYING")

    def _on_bus_message(self, _bus: Any, msg: Any) -> None:
        Gst = self._Gst
        t = msg.type
        if t == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            detail = explain_error(err.message, dbg, self.camera.codec)
            log.warning("[%s] GStreamer error: %s", self.camera.camera_id, detail)
            log.debug("[%s] debug: %s", self.camera.camera_id, dbg)
            self.health.on_error(detail)
            self.schedule_restart(detail)
        elif t == Gst.MessageType.EOS:
            log.info("[%s] end of stream", self.camera.camera_id)
            self.schedule_restart("eos")
        elif t == Gst.MessageType.WARNING:
            w, _ = msg.parse_warning()
            log.debug("[%s] warning: %s", self.camera.camera_id, w.message)

    def schedule_restart(self, reason: str) -> None:
        with self._lock:
            if self._restarting or self._stop.is_set():
                return
            self._restarting = True
        # Tear-down of rtspsrc can block on the network: never do it on the GLib main loop.
        threading.Thread(target=self._restart_worker, args=(reason,), name=f"restart-{self.camera.camera_id}",
                         daemon=True).start()

    def _restart_worker(self, reason: str) -> None:
        try:
            while not self._stop.is_set():
                delay = self._backoff
                self._backoff = min(self._backoff * 2, self.settings.reconnect_max_delay_s)
                log.info("[%s] reconnecting in %.1fs (%s)", self.camera.camera_id, delay, reason)
                self._teardown()
                self.health.on_reconnect(reason)
                if self._stop.wait(delay):
                    return
                try:
                    self._build_and_play()
                    return
                except Exception as exc:
                    reason = str(exc)
                    self.health.on_error(reason)
        finally:
            with self._lock:
                self._restarting = False

    def _teardown(self) -> None:
        with self._lock:
            pipeline, bus = self._pipeline, self._bus
            self._pipeline = self._bus = None
        if bus is not None:
            try:
                bus.remove_signal_watch()
            except Exception:
                pass
        if pipeline is not None:
            pipeline.set_state(self._Gst.State.NULL)

    def stop(self) -> None:
        self._stop.set()
        self._teardown()
        self.health.set_status(STOPPED)
