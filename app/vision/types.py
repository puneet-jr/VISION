"""Shared value objects passed between pipeline stages."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from app.assessment.threat_scorer import ThreatAssessment

if TYPE_CHECKING:  # avoids an import cycle (behavior imports these types)
    from app.assessment.behavior import BehaviorTag
from app.vision.motion_analyzer import MotionState, UNKNOWN_MOTION
from app.vision.zone_analyzer import ZoneContext

BBox = tuple[float, float, float, float]  # x1, y1, x2, y2 in processing-frame pixels


@dataclass(frozen=True)
class Detection:
    frame_id: int
    timestamp: float  # seconds on the source timeline
    class_id: int
    class_name: str
    confidence: float
    bbox: BBox
    track_id: Optional[int] = None  # set by the tracker; identifies an object within a session only

    @property
    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "timestamp": round(self.timestamp, 3),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": [round(v, 1) for v in self.bbox],
            "centroid": [round(v, 1) for v in self.centroid],
            "track_id": self.track_id,
        }


@dataclass(frozen=True)
class ObjectState:
    """Everything known about one tracked object on the current frame."""

    detection: Detection
    motion: MotionState = UNKNOWN_MOTION
    zone_contexts: tuple[ZoneContext, ...] = field(default_factory=tuple)
    primary_zone: Optional[ZoneContext] = None
    dwell_s: Optional[float] = None
    assessment: Optional[ThreatAssessment] = None
    behaviors: tuple["BehaviorTag", ...] = field(default_factory=tuple)

    @property
    def track_id(self) -> Optional[int]:
        return self.detection.track_id

    @property
    def class_name(self) -> str:
        return self.detection.class_name

    def movement_context(self) -> dict:
        ctx = self.motion.to_dict()
        ctx["dwell_s"] = None if self.dwell_s is None else round(self.dwell_s, 2)
        ctx["behaviors"] = [b.code for b in self.behaviors]
        return ctx

    def to_dict(self) -> dict:
        d = self.detection
        zone = self.primary_zone
        a = self.assessment
        return {
            "track_id": d.track_id,
            "class_id": d.class_id,
            "class_name": d.class_name,
            "confidence": round(d.confidence, 4),
            "bbox": [round(v, 1) for v in d.bbox],
            "centroid": [round(v, 1) for v in d.centroid],
            "speed_px_s": round(self.motion.speed_px_s, 2),
            "direction": self.motion.direction,
            "approach": self.motion.approach,
            "zone_id": zone.zone_id if zone else None,
            "zone_name": zone.zone_name if zone else None,
            "zone_status": zone.status.value if zone else None,
            "zone_distance_px": round(zone.distance_px, 1) if zone else None,
            "dwell_s": None if self.dwell_s is None else round(self.dwell_s, 2),
            "score": a.score if a else None,
            "level": a.level.value if a else None,
            "factors": a.factors_dict() if a else {},
            "explanation": a.explanation if a else "",
            "behaviors": [b.to_dict() for b in self.behaviors],
        }
