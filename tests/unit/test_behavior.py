import pytest

from app.assessment.behavior import (
    APPROACHING, CROWDING, INTRUSION, LOITERING, RUNNING, UNATTENDED, BehaviorAnalyzer, BehaviorConfig,
)
from app.assessment.threat_scorer import RiskAdjustment, ThreatInputs, ThreatScorer
from app.vision.motion_analyzer import MotionState
from app.vision.types import Detection, ObjectState
from app.vision.zone_analyzer import ZoneContext, ZoneStatus

DIAG = 1000.0


def state(tid, cls="person", at=(100.0, 100.0), speed=0.0, status=ZoneStatus.OUTSIDE, dwell=0.0,
          approach="unknown", obs=10):
    det = Detection(0, 0.0, 0, cls, 0.9, (at[0] - 10, at[1] - 10, at[0] + 10, at[1] + 10), tid)
    ctx = ZoneContext("z1", "Vault", True, status, 0.0 if status is ZoneStatus.INSIDE else 50.0)
    motion = MotionState(obs, speed, 0.0, None, "stationary", approach)
    return ObjectState(det, motion, (ctx,), ctx, dwell)


def codes(tags, tid):
    return {t.code for t in tags.get(tid, ())}


def test_intrusion_and_loitering_need_stillness_and_time():
    b = BehaviorAnalyzer(BehaviorConfig(loiter_s=6.0))
    walking = b.analyze([state(1, speed=60, status=ZoneStatus.INSIDE, dwell=10)], 10.0, DIAG)
    assert codes(walking, 1) == {INTRUSION}
    briefly = b.analyze([state(2, speed=0, status=ZoneStatus.INSIDE, dwell=2)], 10.0, DIAG)
    assert codes(briefly, 2) == {INTRUSION}
    staying = b.analyze([state(3, speed=0, status=ZoneStatus.INSIDE, dwell=9)], 10.0, DIAG)
    assert codes(staying, 3) == {INTRUSION, LOITERING}


def test_single_observation_is_not_treated_as_stillness():
    b = BehaviorAnalyzer()
    tags = b.analyze([state(1, speed=0, status=ZoneStatus.INSIDE, dwell=30, obs=1)], 0.0, DIAG)
    assert LOITERING not in codes(tags, 1)


def test_running_only_for_people_above_threshold():
    b = BehaviorAnalyzer(BehaviorConfig(run_speed_diag_per_s=0.18))
    tags = b.analyze([state(1, speed=250), state(2, speed=50), state(3, cls="car", speed=400)], 0.0, DIAG)
    assert RUNNING in codes(tags, 1) and RUNNING not in codes(tags, 2) and RUNNING not in codes(tags, 3)


def test_approaching_zone_requires_near_status():
    b = BehaviorAnalyzer()
    near = b.analyze([state(1, speed=40, status=ZoneStatus.NEAR, approach="approaching")], 0.0, DIAG)
    far = b.analyze([state(2, speed=40, status=ZoneStatus.OUTSIDE, approach="approaching")], 0.0, DIAG)
    assert APPROACHING in codes(near, 1) and APPROACHING not in codes(far, 2)


def test_unattended_item_only_after_time_and_when_nobody_is_near():
    b = BehaviorAnalyzer(BehaviorConfig(unattended_s=8.0, unattended_dist_diag=0.12))
    bag = lambda: state(7, cls="backpack", at=(500.0, 500.0))
    for t in range(0, 12):  # stationary 11 seconds
        tags = b.analyze([bag()], float(t), DIAG)
    assert UNATTENDED in codes(tags, 7)

    b2 = BehaviorAnalyzer(BehaviorConfig(unattended_s=8.0))
    owner = state(1, at=(540.0, 500.0), speed=5)  # 40px away: owns the bag
    for t in range(0, 12):
        tags = b2.analyze([bag(), owner], float(t), DIAG)
    assert UNATTENDED not in codes(tags, 7)

    b3 = BehaviorAnalyzer(BehaviorConfig(unattended_s=8.0))
    for t in range(0, 5):  # too short
        tags = b3.analyze([bag()], float(t), DIAG)
    assert UNATTENDED not in codes(tags, 7)


def test_moving_item_resets_the_unattended_timer():
    b = BehaviorAnalyzer(BehaviorConfig(unattended_s=5.0))
    for t in range(0, 4):
        b.analyze([state(7, cls="suitcase", speed=0)], float(t), DIAG)
    b.analyze([state(7, cls="suitcase", speed=200)], 4.0, DIAG)  # picked up and carried
    tags = b.analyze([state(7, cls="suitcase", speed=0)], 6.0, DIAG)
    assert UNATTENDED not in codes(tags, 7)


def test_crowding_tags_everyone_inside_the_zone():
    b = BehaviorAnalyzer(BehaviorConfig(crowd_count=3))
    people = [state(i, status=ZoneStatus.INSIDE, speed=30, dwell=2) for i in (1, 2, 3)]
    tags = b.analyze(people + [state(4, status=ZoneStatus.OUTSIDE)], 0.0, DIAG)
    assert all(CROWDING in codes(tags, i) for i in (1, 2, 3)) and CROWDING not in codes(tags, 4)
    two = b.analyze(people[:2], 1.0, DIAG)
    assert CROWDING not in codes(two, 1)


def test_risk_adjustment_only_raises_and_explains():
    tags = BehaviorAnalyzer().analyze(
        [state(1, cls="backpack")] , 0.0, DIAG)  # nothing yet
    assert BehaviorAnalyzer.risk_adjustment(0.5, tags[1]) is None
    b = BehaviorAnalyzer(BehaviorConfig(unattended_s=1.0))
    for t in range(4):
        t_tags = b.analyze([state(1, cls="backpack")], float(t), DIAG)
    adj = BehaviorAnalyzer.risk_adjustment(0.5, t_tags[1])
    assert adj is not None and adj.value == 0.85 and "unattended" in adj.reason
    assert BehaviorAnalyzer.risk_adjustment(0.95, t_tags[1]) is None  # never lowers a higher class risk


def test_scorer_uses_context_adjusted_risk():
    scorer = ThreatScorer(object_risk={"backpack": 0.5})
    base = scorer.score(ThreatInputs("backpack", 0.9, DIAG))
    raised = scorer.score(ThreatInputs("backpack", 0.9, DIAG, risk_adjustment=RiskAdjustment(0.85, "unattended item")))
    assert raised.score == pytest.approx(base.score + 0.30 * (0.85 - 0.5) * 100)
    detail = raised.factor("object_risk").detail
    assert "raised from 0.50" in detail and "unattended item" in detail
    lowered = scorer.score(ThreatInputs("backpack", 0.9, DIAG, risk_adjustment=RiskAdjustment(0.2, "x")))
    assert lowered.score == base.score   # adjustments can only raise the class risk
