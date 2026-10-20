"""Behaviour recognition on top of tracking, motion and zone context.

Detection alone says *what* is visible. These rules say *what it is doing* - using the track's
history and the other objects in the scene - and feed the threat score as a context-adjusted
object risk (so the five-factor formula and its weights stay unchanged and explainable).

Tags are deterministic engineering rules with configurable thresholds. They are not learned
and not validated against labelled data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from app.assessment.threat_scorer import RiskAdjustment
from app.vision.types import ObjectState
from app.vision.zone_analyzer import ZoneStatus

INTRUSION = "intrusion"
LOITERING = "loitering"
RUNNING = "running"
APPROACHING = "approaching_zone"
UNATTENDED = "unattended_object"
CROWDING = "crowding"

CARRIED_CLASSES = frozenset({"backpack", "suitcase", "handbag"})
PERSON = "person"
UNATTENDED_RISK = 0.85
CROWD_RISK_BONUS = 0.15


@dataclass(frozen=True)
class BehaviorTag:
    code: str
    label: str
    detail: str

    def to_dict(self) -> dict:
        return {"code": self.code, "label": self.label, "detail": self.detail}


@dataclass
class BehaviorConfig:
    loiter_s: float = 6.0
    loiter_speed_diag_per_s: float = 0.02   # at/below this a tracked object counts as "staying put"
    run_speed_diag_per_s: float = 0.18
    unattended_s: float = 8.0
    unattended_dist_diag: float = 0.12      # a person closer than this "owns" the item
    crowd_count: int = 3


class BehaviorAnalyzer:
    """Stateful per session: remembers how long each track has been stationary."""

    def __init__(self, config: BehaviorConfig | None = None) -> None:
        self.cfg = config or BehaviorConfig()
        self._still_since: dict[int, float] = {}
        self._last_seen: dict[int, float] = {}

    def analyze(self, states: Sequence[ObjectState], t: float, diag: float
                ) -> dict[int, tuple[BehaviorTag, ...]]:
        cfg = self.cfg
        people = [s for s in states if s.class_name == PERSON]
        tags: dict[int, list[BehaviorTag]] = {}

        # People currently inside each restricted zone (for crowding).
        in_zone: dict[str, list[ObjectState]] = {}
        for s in people:
            for c in s.zone_contexts:
                if c.restricted and c.status is ZoneStatus.INSIDE:
                    in_zone.setdefault(c.zone_id, []).append(s)

        for s in states:
            tid = s.track_id
            if tid is None:
                continue
            out = tags.setdefault(tid, [])
            speed_frac = s.motion.speed_px_s / diag if diag > 0 and s.motion.observations >= 2 else None
            still = speed_frac is not None and speed_frac <= cfg.loiter_speed_diag_per_s
            self._track_stillness(tid, t, still)
            zone = s.primary_zone
            inside_restricted = any(c.restricted and c.status is ZoneStatus.INSIDE for c in s.zone_contexts)

            if inside_restricted and zone is not None:
                name = next((c.zone_name for c in s.zone_contexts
                             if c.restricted and c.status is ZoneStatus.INSIDE), zone.zone_name)
                out.append(BehaviorTag(INTRUSION, "Inside restricted zone", f"in '{name}'"))
                if s.dwell_s is not None and s.dwell_s >= cfg.loiter_s and still:
                    out.append(BehaviorTag(LOITERING, "Loitering",
                                           f"stationary in '{name}' for {s.dwell_s:.0f}s"))
            elif s.motion.approach == "approaching" and zone is not None and zone.status is ZoneStatus.NEAR:
                out.append(BehaviorTag(APPROACHING, "Approaching zone", f"heading toward '{zone.zone_name}'"))

            if s.class_name == PERSON and speed_frac is not None and speed_frac >= cfg.run_speed_diag_per_s:
                out.append(BehaviorTag(RUNNING, "Moving fast", f"{s.motion.speed_px_s:.0f}px/s"))

            if s.class_name in CARRIED_CLASSES:
                held = self._still_for(tid, t)
                if held >= cfg.unattended_s and not self._person_nearby(s, people, diag):
                    out.append(BehaviorTag(UNATTENDED, "Unattended item",
                                           f"{s.class_name} stationary {held:.0f}s with no person nearby"))

        for zone_people in in_zone.values():
            if len(zone_people) >= cfg.crowd_count:
                for s in zone_people:
                    if s.track_id is not None:
                        tags[s.track_id].append(BehaviorTag(
                            CROWDING, "Crowding", f"{len(zone_people)} people in the same restricted zone"))

        self._prune(t)
        return {tid: tuple(v) for tid, v in tags.items()}

    @staticmethod
    def risk_adjustment(base_risk: float, tags: Sequence[BehaviorTag]) -> RiskAdjustment | None:
        """Context-adjusted object risk, or None when nothing raises it above the class default."""
        codes = {t.code: t for t in tags}
        if UNATTENDED in codes and UNATTENDED_RISK > base_risk:
            return RiskAdjustment(UNATTENDED_RISK, f"unattended item ({codes[UNATTENDED].detail})")
        if CROWDING in codes:
            raised = min(1.0, base_risk + CROWD_RISK_BONUS)
            if raised > base_risk:
                return RiskAdjustment(raised, f"crowding ({codes[CROWDING].detail})")
        return None

    # -- state helpers ------------------------------------------------------------------
    def _track_stillness(self, tid: int, t: float, still: bool) -> None:
        self._last_seen[tid] = t
        if still:
            self._still_since.setdefault(tid, t)
        else:
            self._still_since.pop(tid, None)

    def _still_for(self, tid: int, t: float) -> float:
        since = self._still_since.get(tid)
        return 0.0 if since is None else max(0.0, t - since)

    def _person_nearby(self, item: ObjectState, people: Sequence[ObjectState], diag: float) -> bool:
        ix, iy = item.detection.centroid
        limit = self.cfg.unattended_dist_diag * diag
        return any(math.hypot(ix - p.detection.centroid[0], iy - p.detection.centroid[1]) <= limit
                   for p in people)

    def _prune(self, t: float, stale_after_s: float = 10.0) -> None:
        for tid in [k for k, seen in self._last_seen.items() if t - seen > stale_after_s]:
            self._last_seen.pop(tid, None)
            self._still_since.pop(tid, None)
