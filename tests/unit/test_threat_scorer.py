import math

import pytest

from app.assessment.threat_scorer import (
    RiskLevel, ThreatInputs, ThreatScorer, ThreatWeights, classify_score,
)
from app.core.exceptions import ConfigError, InvalidThreatInputError

DIAG = 400.0


def scorer(**kw) -> ThreatScorer:
    return ThreatScorer(object_risk={"person": 0.4, "knife": 0.9}, **kw)


def full(**over) -> ThreatInputs:
    base = dict(class_name="person", confidence=0.9, frame_diagonal_px=DIAG, speed_px_s=60.0,
                approaching=True, zone_distance_px=0.0, dwell_s=10.0)
    base.update(over)
    return ThreatInputs(**base)


@pytest.mark.parametrize("score,level", [
    (0, RiskLevel.LOW), (30, RiskLevel.LOW), (30.5, RiskLevel.MEDIUM), (31, RiskLevel.MEDIUM),
    (60, RiskLevel.MEDIUM), (60.5, RiskLevel.HIGH), (61, RiskLevel.HIGH), (100, RiskLevel.HIGH),
])
def test_classification_boundaries(score, level):
    assert classify_score(score) is level


@pytest.mark.parametrize("bad", [-0.1, 100.1, float("nan")])
def test_classification_rejects_out_of_range(bad):
    with pytest.raises(InvalidThreatInputError):
        classify_score(bad)


def test_weighted_score_matches_formula():
    # 0.30*0.4 + 0.15*0.9 + 0.20*1 + 0.20*1 + 0.15*1 = 0.805
    result = scorer().score(full())
    assert result.score == pytest.approx(80.5)
    assert result.level is RiskLevel.HIGH
    assert sum(f.contribution for f in result.factors) == pytest.approx(result.score, abs=0.05)
    assert result.factor("object_risk").contribution == pytest.approx(12.0)


def test_person_alone_is_not_high():
    result = scorer().score(full(speed_px_s=0.0, approaching=False, zone_distance_px=1000.0, dwell_s=0.0))
    assert result.level is RiskLevel.LOW
    assert result.score < 30


def test_score_always_within_bounds_at_extremes():
    low = scorer().score(ThreatInputs("x", 0.0, DIAG, 0.0, False, 1e9, 0.0))
    high = scorer().score(ThreatInputs("knife", 1.0, DIAG, 1e9, True, 0.0, 1e9))
    assert 0 <= low.score <= 100 and 0 <= high.score <= 100
    assert high.score == pytest.approx(0.3 * 90 + 15 + 20 + 20 + 15)


def test_unknown_factors_are_flagged_and_contribute_zero():
    result = scorer().score(ThreatInputs("person", 0.8, DIAG, zone_distance_px=0.0))
    assert not result.factor("movement").known and result.factor("movement").contribution == 0
    assert not result.factor("persistence").known
    assert "[unknown]" in result.explanation


def test_proximity_decays_linearly_to_zero():
    s = scorer(zone_near_fraction=0.25)  # near = 100px
    assert s.score(full(zone_distance_px=50.0)).factor("proximity").value == pytest.approx(0.5)
    assert s.score(full(zone_distance_px=100.0)).factor("proximity").value == 0.0
    assert s.score(full(zone_distance_px=500.0)).factor("proximity").value == 0.0


def test_unlisted_class_uses_default_risk():
    assert scorer(default_risk=0.2).score(full(class_name="zebra")).factor("object_risk").value == 0.2


@pytest.mark.parametrize("field,value", [
    ("confidence", 1.5), ("confidence", -0.1), ("confidence", float("nan")),
    ("speed_px_s", -1.0), ("dwell_s", -1.0), ("class_name", "  "), ("frame_diagonal_px", 0.0),
    ("zone_distance_px", float("inf")),
])
def test_invalid_inputs_rejected(field, value):
    with pytest.raises(InvalidThreatInputError):
        scorer().score(full(**{field: value}))


def test_weights_must_sum_to_one_and_be_non_negative():
    ThreatWeights(0.2, 0.2, 0.2, 0.2, 0.2)
    with pytest.raises(ConfigError):
        ThreatWeights(0.3, 0.3, 0.3, 0.3, 0.3)
    with pytest.raises(ConfigError):
        ThreatWeights(1.5, -0.5, 0.0, 0.0, 0.0)
    with pytest.raises(ConfigError):
        ThreatWeights(math.nan, 0.25, 0.25, 0.25, 0.25)


def test_invalid_object_risk_rejected():
    with pytest.raises(ConfigError):
        ThreatScorer(object_risk={"person": 1.2})


def test_custom_weights_change_the_result():
    only_confidence = ThreatWeights(0.0, 1.0, 0.0, 0.0, 0.0)
    assert ThreatScorer(weights=only_confidence).score(full(confidence=0.5)).score == pytest.approx(50.0)
