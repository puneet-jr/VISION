"""Configurable, explainable weighted threat scoring.

    Threat Score = w1*ObjectRisk + w2*Confidence + w3*Movement
                   + w4*Proximity + w5*Persistence        (each factor in [0, 1])

The weighted sum is scaled to 0-100. Weights, object risks and normalisation
references are engineering defaults that can be changed through configuration;
they are NOT scientifically validated probabilities of a threat. Detector
confidence is used as-is (it is not a threat probability).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Optional

from app.core.exceptions import ConfigError, InvalidThreatInputError


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

    @property
    def rank(self) -> int:
        return {"LOW": 0, "MEDIUM": 1, "HIGH": 2}[self.value]


LOW_MAX = 30.0
MEDIUM_MAX = 60.0


def classify_score(score: float) -> RiskLevel:
    """LOW: 0-30, MEDIUM: 31-60, HIGH: 61-100 (fractions fall to the next band)."""
    if score < 0 or score > 100 or math.isnan(score):
        raise InvalidThreatInputError(f"score {score} outside 0-100")
    if score <= LOW_MAX:
        return RiskLevel.LOW
    if score <= MEDIUM_MAX:
        return RiskLevel.MEDIUM
    return RiskLevel.HIGH


@dataclass(frozen=True)
class ThreatWeights:
    object_risk: float = 0.30
    confidence: float = 0.15
    movement: float = 0.20
    proximity: float = 0.20
    persistence: float = 0.15

    def __post_init__(self) -> None:
        values = self.as_tuple()
        if any((not isinstance(v, (int, float))) or math.isnan(v) or v < 0 for v in values):
            raise ConfigError("threat weights must be non-negative numbers")
        if abs(sum(values) - 1.0) > 1e-6:
            raise ConfigError(f"threat weights must sum to 1.0 (got {sum(values):.6f})")

    def as_tuple(self) -> tuple[float, float, float, float, float]:
        return (self.object_risk, self.confidence, self.movement, self.proximity, self.persistence)

    @classmethod
    def from_tuple(cls, values: tuple[float, ...]) -> "ThreatWeights":
        return cls(*values)


@dataclass(frozen=True)
class RiskAdjustment:
    """Context-adjusted object risk (e.g. an unattended bag). Only ever raises the class risk."""

    value: float
    reason: str


@dataclass(frozen=True)
class ThreatInputs:
    """Raw observations for one object. ``None`` means "unknown / not observable"."""

    class_name: str
    confidence: float
    frame_diagonal_px: float
    speed_px_s: Optional[float] = None
    approaching: Optional[bool] = None
    zone_distance_px: Optional[float] = None  # to nearest restricted zone; <= 0 means inside
    dwell_s: Optional[float] = None
    risk_adjustment: Optional[RiskAdjustment] = None


@dataclass(frozen=True)
class FactorScore:
    name: str
    value: float  # normalised, 0..1
    weight: float
    contribution: float  # points out of 100
    known: bool
    detail: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": round(self.value, 4),
            "weight": self.weight,
            "contribution": round(self.contribution, 2),
            "known": self.known,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ThreatAssessment:
    score: float
    level: RiskLevel
    factors: tuple[FactorScore, ...] = field(default_factory=tuple)
    explanation: str = ""

    def factor(self, name: str) -> FactorScore:
        for f in self.factors:
            if f.name == name:
                return f
        raise KeyError(name)

    def factors_dict(self) -> dict[str, dict]:
        return {f.name: f.to_dict() for f in self.factors}


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _finite(name: str, value: float) -> float:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidThreatInputError(f"{name} must be a number, got {value!r}")
    if math.isnan(value) or math.isinf(value):
        raise InvalidThreatInputError(f"{name} must be finite, got {value!r}")
    return float(value)


class ThreatScorer:
    """Deterministic rule-based scorer."""

    def __init__(
        self,
        weights: ThreatWeights | None = None,
        object_risk: Mapping[str, float] | None = None,
        default_risk: float = 0.20,
        speed_ref_diag_per_s: float = 0.15,
        zone_near_fraction: float = 0.15,
        dwell_ref_s: float = 10.0,
    ) -> None:
        self.weights = weights or ThreatWeights()
        self.object_risk = {k.lower(): float(v) for k, v in (object_risk or {}).items()}
        for name, risk in {**self.object_risk, "<default>": default_risk}.items():
            if not 0.0 <= risk <= 1.0:
                raise ConfigError(f"object risk for '{name}' must be within [0, 1]")
        if min(speed_ref_diag_per_s, zone_near_fraction, dwell_ref_s) <= 0:
            raise ConfigError("normalisation references must be positive")
        self.default_risk = float(default_risk)
        self.speed_ref = speed_ref_diag_per_s
        self.near_fraction = zone_near_fraction
        self.dwell_ref_s = dwell_ref_s

    # -- individual factor normalisation -------------------------------------------------
    def class_risk(self, class_name: str) -> float:
        """Configured (or default) risk of a class, before any context adjustment."""
        return self.object_risk.get(class_name.strip().lower(), self.default_risk)

    def _object_risk(self, class_name: str, adjustment: Optional[RiskAdjustment] = None
                     ) -> tuple[float, bool, str]:
        key = class_name.strip().lower()
        base = self.class_risk(key)
        if adjustment is not None and adjustment.value > base:
            return min(1.0, adjustment.value), True, f"raised from {base:.2f} by context: {adjustment.reason}"
        if key in self.object_risk:
            return base, True, f"configured risk for '{key}'"
        return base, True, f"default risk (no rule for '{key}')"

    def _movement(self, speed: Optional[float], approaching: Optional[bool], diag: float
                  ) -> tuple[float, bool, str]:
        if speed is None:
            return 0.0, False, "unknown (no temporal data); contributes 0"
        speed_n = _clip01(speed / (self.speed_ref * diag))
        if approaching is None:
            return speed_n, True, f"speed {speed:.1f}px/s; approach unknown"
        value = 0.6 * speed_n + 0.4 * (1.0 if approaching else 0.0)
        return value, True, f"speed {speed:.1f}px/s; {'approaching' if approaching else 'not approaching'} zone"

    def _proximity(self, distance: Optional[float], diag: float) -> tuple[float, bool, str]:
        if distance is None:
            return 0.0, True, "no restricted zone configured"
        if distance <= 0:
            return 1.0, True, "inside restricted zone"
        near = self.near_fraction * diag
        return _clip01(1.0 - distance / near), True, f"{distance:.0f}px from restricted zone"

    def _persistence(self, dwell: Optional[float]) -> tuple[float, bool, str]:
        if dwell is None:
            return 0.0, False, "unknown (no temporal data); contributes 0"
        return _clip01(dwell / self.dwell_ref_s), True, f"{dwell:.1f}s in monitored zone"

    # -- scoring -------------------------------------------------------------------------
    def _validate(self, i: ThreatInputs) -> None:
        if not isinstance(i.class_name, str) or not i.class_name.strip():
            raise InvalidThreatInputError("class_name must be a non-empty string")
        conf = _finite("confidence", i.confidence)
        if not 0.0 <= conf <= 1.0:
            raise InvalidThreatInputError(f"confidence {conf} outside [0, 1]")
        if _finite("frame_diagonal_px", i.frame_diagonal_px) <= 0:
            raise InvalidThreatInputError("frame_diagonal_px must be positive")
        if i.speed_px_s is not None and _finite("speed_px_s", i.speed_px_s) < 0:
            raise InvalidThreatInputError("speed_px_s must be >= 0")
        if i.dwell_s is not None and _finite("dwell_s", i.dwell_s) < 0:
            raise InvalidThreatInputError("dwell_s must be >= 0")
        if i.zone_distance_px is not None:
            _finite("zone_distance_px", i.zone_distance_px)

    def score(self, inputs: ThreatInputs) -> ThreatAssessment:
        self._validate(inputs)
        diag = float(inputs.frame_diagonal_px)
        w = self.weights
        parts = [
            ("object_risk", w.object_risk, *self._object_risk(inputs.class_name, inputs.risk_adjustment)),
            ("confidence", w.confidence, float(inputs.confidence), True, "detector confidence"),
            ("movement", w.movement, *self._movement(inputs.speed_px_s, inputs.approaching, diag)),
            ("proximity", w.proximity, *self._proximity(inputs.zone_distance_px, diag)),
            ("persistence", w.persistence, *self._persistence(inputs.dwell_s)),
        ]
        factors = tuple(
            FactorScore(name, value, weight, weight * value * 100.0, known, detail)
            for name, weight, value, known, detail in parts
        )
        total = round(min(100.0, max(0.0, sum(f.contribution for f in factors))), 2)
        level = classify_score(total)
        explanation = self._explain(total, level, factors)
        return ThreatAssessment(total, level, factors, explanation)

    @staticmethod
    def _explain(total: float, level: RiskLevel, factors: tuple[FactorScore, ...]) -> str:
        lines = [f"Threat score {total:.1f}/100 ({level.value})."]
        for f in factors:
            tag = "" if f.known else " [unknown]"
            lines.append(
                f"{f.name}: {f.value:.2f} x w{f.weight:.2f} = {f.contribution:.1f} pts{tag} - {f.detail}"
            )
        return " ".join(lines)
