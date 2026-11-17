import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import CameraConfig, DetectionSettings, RecognitionSettings  # noqa: E402
from constants import UNKNOWN_PERSON_ID  # noqa: E402
from fakes import FakeArcFaceModel, FakeSCRFDModel  # noqa: E402
from inference.arcface_engine import ArcFaceEngine  # noqa: E402
from inference.scrfd_engine import SCRFDEngine  # noqa: E402
from ingestion.frame_receiver import FrameReceiver  # noqa: E402
from monitoring.performance import PerformanceTracker  # noqa: E402
from recognition.duplicate_filter import DuplicateFilter, TrackCache  # noqa: E402
from recognition.embedding_manager import EmbeddingGallery  # noqa: E402
from recognition.face_matcher import FaceMatcher  # noqa: E402
from scheduler.frame_scheduler import FrameScheduler  # noqa: E402
from services.recognition_service import RecognitionService  # noqa: E402


def _unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


def test_matcher_picks_best_person_and_rejects_unknown():
    g = EmbeddingGallery(dim=4)
    g.replace([2, 1, 2], np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0.9, 0.1, 0]], np.float32))
    m = FaceMatcher(g, threshold=0.5, margin=0.05)
    res = m.match_batch(np.stack([_unit([1, 0.1, 0, 0]), _unit([0, 1, 0, 0]), _unit([0, 0, 0, 1])]))
    assert [r.person_id for r in res] == [1, 2, UNKNOWN_PERSON_ID]
    assert res[1].similarity > 0.99 and not res[2].is_known


def test_matcher_margin_rejects_ambiguous():
    g = EmbeddingGallery(dim=2)
    g.replace([1, 2], np.array([[1, 0.05], [1, -0.05]], np.float32))
    assert not FaceMatcher(g, threshold=0.5, margin=0.05).match(_unit([1, 0])).is_known


def test_empty_gallery_is_unknown():
    assert FaceMatcher(EmbeddingGallery()).match(_unit(np.ones(512))).person_id == UNKNOWN_PERSON_ID


def test_gallery_add_remove():
    g = EmbeddingGallery(dim=2)
    g.add(5, [[1, 0]])
    g.add(3, [[0, 1], [0, 2]])
    assert g.num_persons == 2 and g.size == 3
    g.remove_person(3)
    assert g.num_persons == 1 and g.snapshot().unique_ids.tolist() == [5]


def test_duplicate_filter_cooldown():
    f = DuplicateFilter(cooldown_s=10, unknown_cooldown_s=10, unknown_similarity=0.8)
    assert f.should_emit("c1", 1, now=0)
    assert not f.should_emit("c1", 1, now=5)
    assert f.should_emit("c2", 1, now=5)  # other camera
    assert not f.should_emit("c1", 1, now=14)  # still present (last seen 5)
    assert f.should_emit("c1", 1, now=30)
    e = _unit([1, 0, 0])
    assert f.should_emit("c1", UNKNOWN_PERSON_ID, e, now=0)
    assert not f.should_emit("c1", UNKNOWN_PERSON_ID, _unit([1, 0.1, 0]), now=1)
    assert f.should_emit("c1", UNKNOWN_PERSON_ID, _unit([0, 1, 0]), now=2)


def test_track_cache_reuse_and_reverify():
    t = TrackCache(iou_threshold=0.5, reverify_s=2.0, ttl_s=1.5)
    assert t.lookup("c1", [0, 0, 100, 100], now=0) is None
    t.update("c1", [0, 0, 100, 100], 7, 0.8, now=0)
    assert t.lookup("c1", [5, 5, 105, 105], now=0.5) == (7, 0.8)
    assert t.lookup("c2", [5, 5, 105, 105], now=0.5) is None
    assert t.lookup("c1", [500, 500, 600, 600], now=0.5) is None
    assert t.lookup("c1", [5, 5, 105, 105], now=2.5) is None  # needs re-verification


class _FakeInference:
    def __init__(self):
        self.scrfd = SCRFDEngine(FakeSCRFDModel())
        self.arcface = ArcFaceEngine(FakeArcFaceModel())

    def detect(self, images):
        return self.scrfd.detect_batch(images)

    def embed(self, crops):
        return self.arcface.embed_batch(crops)


class _FakeOffenders:
    """Stands in for OffenderRepository: one enrolled vector in tbl_offenders, logs kept in memory."""

    def __init__(self, vec):
        self.vec = _unit(vec)
        self.searches = 0
        self.logs = []

    def search(self, embedding):
        self.searches += 1
        return {"offender_id": 7, "offender_name": "Dave", "similarity": float(np.dot(self.vec, _unit(embedding)))}

    def log_search(self, **row):
        self.logs.append(row)
        return len(self.logs)


def test_end_to_end_pipeline_with_fake_models(monkeypatch, tmp_path):
    monkeypatch.setenv("FR_OFFENDER_IMAGE_BASE_URL", "https://faces.example.com")
    monkeypatch.setenv("FR_DETECTED_IMAGE_DIR", str(tmp_path / "detected"))
    inf = _FakeInference()
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 255, (720, 1280, 3)).astype(np.uint8)
    dets, kps = inf.detect([frame])[0]
    from processing.face_alignment import norm_crop

    offenders = _FakeOffenders(inf.embed([norm_crop(frame, kps[0])])[0])
    perf = PerformanceTracker()
    sched = FrameScheduler(max_frame_age_s=5)
    cams = [CameraConfig(camera_id="27", name="gate", url="test://", gate_id=11)]
    recv = FrameReceiver("27", sched.register("27"))
    det_cfg = DetectionSettings(min_face_size=20)
    rec_cfg = RecognitionSettings(blur_threshold=1.0, min_quality=0.0, max_yaw_ratio=1.0)
    svc = RecognitionService(inf, sched, offenders, DuplicateFilter(60), TrackCache(0.45, 2.0),
                             cams, det_cfg, rec_cfg, perf)
    svc.start()
    for _ in range(4):
        recv(frame)
        time.sleep(0.1)
    time.sleep(0.3)
    svc.stop()

    assert offenders.searches >= 1
    assert perf.count("recognized") == 1 and len(offenders.logs) == 1  # de-duplicated: one log row
    row = offenders.logs[0]
    assert (row["device_id"], row["gate_id"], row["offender_id"]) == (27, 11, 7)
    assert row["match_percentage"] > 99 and row["time_taken_ms"] > 0 and row["frameno"] >= 0
    assert len(row["embedding"]) == 512
    assert row["detected_imagepath"].startswith("https://faces.example.com/images/detected/")
    assert row["detected_imagepath"].endswith("_frame.jpg") and row["detected_face"].endswith("_face.jpg")
    assert len(list((tmp_path / "detected").glob("*.jpg"))) == 2         # frame + face saved
