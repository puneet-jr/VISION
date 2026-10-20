from datetime import datetime

import pytest

from app.assessment.alert_manager import (
    EVENT_THREAT_LEVEL, EVENT_ZONE_ENTRY, AlertManager,
)
from app.assessment.threat_scorer import RiskLevel, ThreatAssessment
from app.vision.types import Detection, ObjectState
from app.vision.zone_analyzer import ZoneContext, ZoneStatus

WALL = datetime(2025, 1, 1, 12, 0, 0)
SCORES = {RiskLevel.LOW: 10.0, RiskLevel.MEDIUM: 45.0, RiskLevel.HIGH: 80.0}


def obj(level: RiskLevel, track_id: int = 1, zone: bool = True) -> ObjectState:
    det = Detection(0, 0.0, 0, "person", 0.9, (0, 0, 10, 10), track_id)
    ctx = ZoneContext("z1", "Restricted A", True, ZoneStatus.INSIDE, 0.0)
    return ObjectState(det, zone_contexts=(ctx,) if zone else (), primary_zone=ctx if zone else None,
                       dwell_s=1.0, assessment=ThreatAssessment(SCORES[level], level, (), "because"))


def manager(**kw) -> tuple[AlertManager, list]:
    sink: list = []
    kw.setdefault("cooldown_s", 10.0)
    return AlertManager(sinks=[sink.append], session_id=1, **kw), sink


def feed(m, level, t, entered=(), track_id=1):
    return m.process(obj(level, track_id), entered, t, WALL)


def test_event_carries_required_fields():
    m, sink = manager()
    (event,) = feed(m, RiskLevel.HIGH, 0.0)
    assert event.event_type == EVENT_THREAT_LEVEL and event.level is RiskLevel.HIGH
    assert event.track_id == 1 and event.class_name == "person" and event.score == 80.0
    assert event.zone_name == "Restricted A" and event.explanation == "because" and event.reason
    assert event.timestamp == WALL and sink == [event]
    assert event.to_dict()["timestamp"].endswith("Z") and event.to_dict()["priority"] == 3


def test_unchanged_condition_does_not_repeat_every_frame():
    m, sink = manager()
    for i in range(100):
        feed(m, RiskLevel.HIGH, i * 0.1)
    assert len(sink) == 1


def test_below_min_level_is_silent_but_low_can_be_enabled():
    m, sink = manager()
    feed(m, RiskLevel.LOW, 0.0)
    assert sink == []
    m2, sink2 = manager(min_level=RiskLevel.LOW)
    feed(m2, RiskLevel.LOW, 0.0)
    assert len(sink2) == 1 and sink2[0].level is RiskLevel.LOW


def test_escalation_bypasses_cooldown():
    m, sink = manager()
    feed(m, RiskLevel.MEDIUM, 0.0)
    feed(m, RiskLevel.HIGH, 0.5)
    assert [e.level for e in sink] == [RiskLevel.MEDIUM, RiskLevel.HIGH]
    assert "MEDIUM -> HIGH" in sink[1].reason


def test_deescalation_waits_for_cooldown():
    m, sink = manager(cooldown_s=10.0)
    feed(m, RiskLevel.HIGH, 0.0)
    feed(m, RiskLevel.MEDIUM, 3.0)
    assert len(sink) == 1
    feed(m, RiskLevel.MEDIUM, 11.0)
    assert [e.level for e in sink] == [RiskLevel.HIGH, RiskLevel.MEDIUM]


def test_flicker_around_a_boundary_is_suppressed():
    m, sink = manager(cooldown_s=10.0)
    for i in range(40):
        feed(m, RiskLevel.HIGH if i % 2 == 0 else RiskLevel.MEDIUM, i * 0.1)
    assert len(sink) == 1


def test_reescalation_after_dropping_below_alert_band_is_reported():
    m, sink = manager(cooldown_s=5.0)
    feed(m, RiskLevel.HIGH, 0.0)
    feed(m, RiskLevel.LOW, 6.0)   # cooled down and below min level: state forgotten
    feed(m, RiskLevel.HIGH, 7.0)
    assert len(sink) == 2


def test_zone_entry_event_and_cooldown():
    m, sink = manager(cooldown_s=10.0)
    events = feed(m, RiskLevel.LOW, 0.0, entered=["z1"])
    assert [e.event_type for e in events] == [EVENT_ZONE_ENTRY]
    assert "entered zone" in events[0].reason and events[0].zone_id == "z1"
    assert feed(m, RiskLevel.LOW, 2.0, entered=["z1"]) == []        # re-entry inside cooldown
    assert len(feed(m, RiskLevel.LOW, 12.0, entered=["z1"])) == 1   # after cooldown


def test_zone_entry_and_level_change_on_the_same_frame_are_one_event():
    m, sink = manager()
    (event,) = feed(m, RiskLevel.HIGH, 0.0, entered=["z1"])
    assert event.event_type == EVENT_ZONE_ENTRY and event.level is RiskLevel.HIGH
    assert "entered zone" in event.reason and "threat level HIGH" in event.reason
    assert feed(m, RiskLevel.HIGH, 0.1) == []   # the level was recorded, so no repeat follows
    assert len(sink) == 1


def test_events_are_ordered_by_priority():
    m, _ = manager()
    events = [e for t, lv in ((0.0, RiskLevel.MEDIUM), (0.1, RiskLevel.HIGH)) for e in feed(m, lv, t)]
    assert sorted(events, key=lambda e: -e.priority)[0].level is RiskLevel.HIGH


def test_tracks_are_independent():
    m, sink = manager()
    feed(m, RiskLevel.HIGH, 0.0, track_id=1)
    feed(m, RiskLevel.HIGH, 0.1, track_id=2)
    assert len(sink) == 2


def test_failing_sink_does_not_break_processing():
    seen: list = []

    def boom(_event):
        raise RuntimeError("db down")

    m = AlertManager(sinks=[boom, seen.append], session_id=1)
    feed(m, RiskLevel.HIGH, 0.0)
    assert len(seen) == 1


def test_object_without_track_or_assessment_is_ignored():
    m, sink = manager()
    det = Detection(0, 0.0, 0, "person", 0.9, (0, 0, 1, 1), None)
    assert m.process(ObjectState(det), [], 0.0) == [] and sink == []


def test_prune_forgets_stale_tracks():
    m, sink = manager()
    feed(m, RiskLevel.HIGH, 0.0)
    m.prune(100.0, stale_after_s=5.0)
    feed(m, RiskLevel.HIGH, 100.0)
    assert len(sink) == 2
