import numpy as np
import pytest

from app.assessment.alert_manager import EVENT_ZONE_ENTRY
from app.assessment.threat_scorer import RiskLevel
from app.core.config import Settings
from app.vision.pipeline import ImageAnalyzer, build_pipeline, build_scorer, build_zone_provider
from app.vision.preprocessing import FramePreprocessor
from tests.conftest import FakeDetector, blank_frame, person_walk


@pytest.fixture
def cfg(tmp_path):
    return Settings(_env_file=None, zones_file=tmp_path / "none.json", alert_cooldown_s=1.0,
                    max_frame_width=0)


def test_walk_into_zone_produces_context_scores_and_alerts(cfg):
    detector = FakeDetector()
    sink: list = []
    pipeline = build_pipeline(cfg, detector, build_zone_provider(cfg), session_id=7, sinks=[sink.append])
    assert detector.resets == 1  # tracker state is reset when a pipeline is built

    first = pipeline.process_frame(blank_frame(), 0, 0.0)
    assert first.annotated.shape == (240, 320, 3) and len(first.objects) == 1
    assert first.objects[0].primary_zone.status.value == "outside"
    assert first.objects[0].assessment.level is RiskLevel.LOW

    for fid in range(1, 50):
        result = pipeline.process_frame(blank_frame(), fid, fid / 10.0)
    obj = result.objects[0]
    assert obj.primary_zone.status.value == "inside" and obj.dwell_s > 0
    assert obj.motion.direction == "right" and obj.motion.speed_px_s == pytest.approx(50.0, rel=0.05)
    assert obj.assessment.level.rank >= RiskLevel.MEDIUM.rank   # crossing the zone is not benign
    assert 0 <= obj.assessment.score <= 100

    types = [(e.event_type, e.level) for e in sink]
    assert (EVENT_ZONE_ENTRY, RiskLevel.MEDIUM) in types or any(t == EVENT_ZONE_ENTRY for t, _ in types)
    assert any(lv is RiskLevel.HIGH for _, lv in types)
    assert len([e for e in sink if e.event_type == EVENT_ZONE_ENTRY]) == 1   # one entry, not per frame
    assert len(sink) <= 6                                                    # no per-frame spam
    assert all(e.session_id == 7 for e in sink)


def test_loitering_in_zone_raises_the_score_as_persistence_grows(cfg):
    still = lambda fid: [person_walk(46, conf=0.95)]  # stationary at x=250, inside the zone
    sink: list = []
    pipeline = build_pipeline(cfg, FakeDetector(still), build_zone_provider(cfg), sinks=[sink.append])
    scores = {}
    for fid in range(0, 160):
        result = pipeline.process_frame(blank_frame(), fid, fid / 10.0)
        scores[fid] = result.objects[0].assessment
    assert scores[5].score < scores[60].score < scores[159].score
    assert scores[159].factor("persistence").value == 1.0
    assert scores[159].factor("movement").value == 0.0
    assert scores[159].level is RiskLevel.HIGH
    # MEDIUM arrives together with the zone entry (one merged event), then HIGH once dwell builds up.
    assert [(e.event_type, e.level) for e in sink] == [
        (EVENT_ZONE_ENTRY, RiskLevel.MEDIUM), ("threat_level", RiskLevel.HIGH)]


def test_zone_coordinates_scale_with_frame_size(cfg):
    provider = build_zone_provider(cfg)
    small, big = provider.for_size(320, 240), provider.for_size(1280, 720)
    assert big.zones[0].polygon[0] == (640.0, 180.0)
    assert small.zones[0].polygon[0] == (160.0, 60.0)
    assert provider.for_size(320, 240) is small  # cached


def test_resizing_keeps_zone_alignment(cfg):
    cfg = cfg.model_copy(update={"max_frame_width": 160})
    pipeline = build_pipeline(cfg, FakeDetector(lambda fid: []), build_zone_provider(cfg))
    result = pipeline.process_frame(blank_frame(), 0, 0.0)
    assert result.size == (160, 120) and result.annotated.shape == (120, 160, 3)


def test_image_analysis_marks_temporal_factors_unknown(cfg):
    detector = FakeDetector(lambda fid: [person_walk(30)])
    analyzer = ImageAnalyzer(detector, build_zone_provider(cfg), build_scorer(cfg), FramePreprocessor())
    result = analyzer.analyze(blank_frame())
    (obj,) = result.objects
    a = obj.assessment
    assert not a.factor("movement").known and not a.factor("persistence").known
    assert a.factor("proximity").known
    assert obj.track_id is None and result.max_level is a.level
    assert result.annotated.shape == (240, 320, 3)


def test_invalid_frame_is_rejected(cfg):
    from app.core.exceptions import InvalidFrameError
    pipeline = build_pipeline(cfg, FakeDetector(), build_zone_provider(cfg))
    with pytest.raises(InvalidFrameError):
        pipeline.process_frame(np.zeros((0, 0, 3), np.uint8), 0, 0.0)
