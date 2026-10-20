"""Turns per-frame threat assessments into a small number of meaningful alert events.

Rules (per tracked object):
* A zone-entry event is emitted once per entry, with a per-(track, zone) cooldown.
* A threat-level event is emitted when the classification differs from the last emitted
  one. Escalations are emitted immediately; de-escalations wait for the cooldown. This
  suppresses per-frame repeats and flicker around a band boundary.
* Levels below ``min_level`` do not produce threat-level events.
* A zone entry and a level change on the same frame are reported as one event.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Iterable, Optional

from app.assessment.threat_scorer import RiskLevel
from app.vision.types import ObjectState

logger = logging.getLogger(__name__)

EVENT_THREAT_LEVEL = "threat_level"
EVENT_ZONE_ENTRY = "zone_entry"

EventSink = Callable[["AlertEvent"], None]


def utc_now() -> datetime:
    """Naive UTC timestamp (what the database stores)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


@dataclass
class AlertEvent:
    session_id: Optional[int]
    timestamp: datetime
    source_time_s: float
    event_type: str
    track_id: int
    class_name: str
    confidence: float
    score: float
    level: RiskLevel
    zone_id: Optional[str]
    zone_name: Optional[str]
    movement: dict
    factors: dict
    explanation: str
    reason: str
    bbox: list[float] = field(default_factory=list)
    centroid: list[float] = field(default_factory=list)
    event_id: Optional[int] = None

    @property
    def priority(self) -> int:
        """Display priority: HIGH first."""
        return {"HIGH": 3, "MEDIUM": 2, "LOW": 1}[self.level.value]

    def to_dict(self) -> dict:
        return {
            "id": self.event_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
            "event_type": self.event_type,
            "track_id": self.track_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "score": self.score,
            "level": self.level.value,
            "priority": self.priority,
            "zone_id": self.zone_id,
            "zone_name": self.zone_name,
            "movement": self.movement,
            "factors": self.factors,
            "explanation": self.explanation,
            "reason": self.reason,
        }


@dataclass
class _TrackAlertState:
    last_level: Optional[RiskLevel] = None
    last_emit_t: Optional[float] = None
    last_seen_t: float = 0.0


class AlertManager:
    def __init__(
        self,
        cooldown_s: float = 10.0,
        min_level: RiskLevel = RiskLevel.MEDIUM,
        sinks: Iterable[EventSink] = (),
        session_id: Optional[int] = None,
    ) -> None:
        self.cooldown_s = cooldown_s
        self.min_level = min_level
        self.session_id = session_id
        self._sinks: list[EventSink] = list(sinks)
        self._tracks: dict[int, _TrackAlertState] = {}
        self._zone_last_emit: dict[tuple[int, str], float] = {}

    def add_sink(self, sink: EventSink) -> None:
        self._sinks.append(sink)

    def process(self, obj: ObjectState, entered_zone_ids: Iterable[str], t: float,
                wall_time: Optional[datetime] = None) -> list[AlertEvent]:
        """Evaluate one object on one frame. ``t`` is the source-timeline time in seconds."""
        if obj.track_id is None or obj.assessment is None:
            return []
        state = self._tracks.setdefault(obj.track_id, _TrackAlertState())
        state.last_seen_t = t
        wall = wall_time or utc_now()
        events: list[AlertEvent] = []

        for zone_id in entered_zone_ids:
            key = (obj.track_id, zone_id)
            last = self._zone_last_emit.get(key)
            if last is not None and t - last < self.cooldown_s:
                continue
            self._zone_last_emit[key] = t
            zone = next((c for c in obj.zone_contexts if c.zone_id == zone_id), None)
            name = zone.zone_name if zone else zone_id
            events.append(self._make_event(
                obj, wall, t, EVENT_ZONE_ENTRY, zone_id, name,
                f"{obj.class_name} #{obj.track_id} entered zone '{name}'"))

        level_event, change = self._level_event(obj, state, t, wall)
        if level_event is not None:
            if events:
                # Same frame as a zone entry: report once, carrying the level change in the reason.
                events[0].reason += f"; {change}"
            else:
                events.append(level_event)

        events.sort(key=lambda e: -e.priority)
        for event in events:
            self._dispatch(event)
        return events

    def _level_event(self, obj: ObjectState, state: _TrackAlertState, t: float, wall: datetime
                     ) -> tuple[Optional[AlertEvent], str]:
        """Return (event, change description); (None, "") when nothing should be emitted."""
        assert obj.assessment is not None
        level = obj.assessment.level
        cooled = state.last_emit_t is None or t - state.last_emit_t >= self.cooldown_s

        if level.rank < self.min_level.rank:
            # Fell below the alerting band: forget the last emitted level once cooled down,
            # so a later re-escalation is reported again.
            if cooled:
                state.last_level = None
            return None, ""
        if state.last_level == level:
            return None, ""
        escalation = state.last_level is None or level.rank > state.last_level.rank
        if not (escalation or cooled):
            return None, ""

        previous = state.last_level
        state.last_level, state.last_emit_t = level, t
        zone = obj.primary_zone
        change = (f"threat level {previous.value + ' -> ' if previous else ''}{level.value} "
                  f"(score {obj.assessment.score:.1f})")
        event = self._make_event(obj, wall, t, EVENT_THREAT_LEVEL,
                                 zone.zone_id if zone else None, zone.zone_name if zone else None,
                                 f"{obj.class_name} #{obj.track_id} {change}")
        return event, change

    def _make_event(self, obj: ObjectState, wall: datetime, t: float, event_type: str,
                    zone_id: Optional[str], zone_name: Optional[str], reason: str) -> AlertEvent:
        a = obj.assessment
        assert a is not None and obj.track_id is not None
        d = obj.detection
        return AlertEvent(
            session_id=self.session_id, timestamp=wall, source_time_s=t, event_type=event_type,
            track_id=obj.track_id, class_name=d.class_name, confidence=d.confidence,
            score=a.score, level=a.level, zone_id=zone_id, zone_name=zone_name,
            movement=obj.movement_context(), factors=a.factors_dict(), explanation=a.explanation,
            reason=reason, bbox=[round(v, 1) for v in d.bbox],
            centroid=[round(v, 1) for v in d.centroid],
        )

    def _dispatch(self, event: AlertEvent) -> None:
        for sink in self._sinks:
            try:
                sink(event)
            except Exception:  # a failing sink must never stop the pipeline
                logger.exception("alert sink %r failed", sink)

    def prune(self, t: float, stale_after_s: float) -> None:
        stale = {tid for tid, s in self._tracks.items() if t - s.last_seen_t > stale_after_s}
        for tid in stale:
            del self._tracks[tid]
        for key in [k for k in self._zone_last_emit if k[0] in stale]:
            del self._zone_last_emit[key]
