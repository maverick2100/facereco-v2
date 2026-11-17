"""Image URLs stored in the database, and the files behind them.

    URL  = <FR_OFFENDER_IMAGE_BASE_URL>/images/<folder>/<epoch ms>_<suffix>.jpg
    file = <dir from .env>/<epoch ms>_<suffix>.jpg      (only when that dir is set)

folder / dir:
    offender  FR_OFFENDER_IMAGE_DIR   enrolment thumbnails      (tbl_offenders.imagepath)
    detected  FR_DETECTED_IMAGE_DIR   frames + faces from search (tbl_offender_search_logs)
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from utils.image_utils import save_jpeg

OFFENDER = ("offender", "FR_OFFENDER_IMAGE_DIR")
DETECTED = ("detected", "FR_DETECTED_IMAGE_DIR")

_ts_lock = threading.Lock()
_last_ts = 0


def unique_ms() -> int:
    """Epoch milliseconds, strictly increasing within this process (two images never share a name)."""
    global _last_ts
    with _ts_lock:
        _last_ts = max(int(time.time() * 1000), _last_ts + 1)
        return _last_ts


def base_url() -> str:
    """FR_OFFENDER_IMAGE_BASE_URL as a URL prefix. A bare IP / host (192.168.1.50, 192.168.1.50:8080)
    gets https:// in front; write http://... explicitly to use plain http."""
    raw = os.getenv("FR_OFFENDER_IMAGE_BASE_URL", "").strip().rstrip("/")
    if raw and "://" not in raw:
        raw = "https://" + raw
    return raw


def image_url(kind: tuple, ts_ms: int, suffix: str) -> Optional[str]:
    """Public URL for an image, or None when FR_OFFENDER_IMAGE_BASE_URL is not set."""
    base = base_url()
    return f"{base}/images/{kind[0]}/{ts_ms}_{suffix}.jpg" if base else None


def image_dir(kind: tuple) -> Optional[Path]:
    raw = os.getenv(kind[1], "").strip()
    return Path(raw).expanduser() if raw else None


def store(kind: tuple, ts_ms: int, suffix: str, img: Optional[np.ndarray], quality: int = 85) -> Optional[str]:
    """Save img as <dir>/<ts>_<suffix>.jpg (if the dir is configured) and return its URL."""
    folder = image_dir(kind)
    if folder is not None and img is not None:
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"{ts_ms}_{suffix}.jpg"
        if not save_jpeg(target, img, quality):
            raise RuntimeError(f"could not write image {target}")
    return image_url(kind, ts_ms, suffix)
