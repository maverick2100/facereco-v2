"""Project-wide constants (paths, model geometry, labels)."""
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_CACHE_DIR = MODELS_DIR / ".compiled"  # cached .mlmodelc (skips re-compiling on start)
CONFIG_DIR = PROJECT_ROOT / "config"
LOGS_DIR = PROJECT_ROOT / "logs"
DATA_DIR = PROJECT_ROOT / "data"
ENROLLMENT_DIR = DATA_DIR / "enrollment"
EXPORTS_DIR = DATA_DIR / "exports"
SNAPSHOTS_DIR = EXPORTS_DIR / "snapshots"

DEFAULT_SCRFD_PATH = MODELS_DIR / "scrfd.mlpackage"
DEFAULT_ARCFACE_PATH = MODELS_DIR / "arcface.mlpackage"
DEFAULT_YOLO_PATH = MODELS_DIR / "yolov7.mlpackage"
DEFAULT_DB_PATH = DATA_DIR / "face_recognition.db"

# --------------------------------------------------------------------------- camera tasks
TASK_FACE = "face"        # SCRFD + ArcFace recognition
TASK_PERSON = "person"    # YOLOv7 person detection + counting
CAMERA_TASKS = (TASK_FACE, TASK_PERSON)

# --------------------------------------------------------------------------- SCRFD
SCRFD_DEFAULT_INPUT = (640, 640)  # (width, height) used if the model spec has no fixed size
SCRFD_INPUT_MEAN = 127.5
SCRFD_INPUT_STD = 128.0

# --------------------------------------------------------------------------- ArcFace
ARCFACE_INPUT_SIZE = 112
ARCFACE_EMBEDDING_DIM = 512
ARCFACE_INPUT_MEAN = 127.5
ARCFACE_INPUT_STD = 127.5

# Reference 5-point landmarks (left eye, right eye, nose, left mouth, right mouth)
# for a 112x112 ArcFace crop (InsightFace standard template).
ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

# --------------------------------------------------------------------------- YOLOv7 (person)
YOLO_DEFAULT_INPUT = (640, 640)
YOLO_PAD_VALUE = 114
COCO_PERSON_CLASS_ID = 0
# Default YOLOv7 anchors (pixels) per stride - only used for raw (un-decoded) head outputs.
YOLOV7_ANCHORS = {
    8: [(12, 16), (19, 36), (40, 28)],
    16: [(36, 75), (76, 55), (72, 146)],
    32: [(142, 110), (192, 243), (459, 401)],
}

# --------------------------------------------------------------------------- misc
UNKNOWN_PERSON_ID = -1
UNKNOWN_LABEL = "unknown"
EVENT_LOGGER_NAME = "events"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
