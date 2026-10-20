"""End-to-end pipeline: pre-process -> detect -> track -> context -> behaviour -> score -> alert."""
from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from app.assessment.alert_manager import AlertEvent, AlertManager, EventSink
from app.assessment.behavior import BehaviorAnalyzer, BehaviorConfig
from app.assessment.threat_scorer import RiskLevel, ThreatInputs, ThreatScorer, ThreatWeights
from app.core.config import Settings
from app.vision import annotator
from app.vision.detector import ObjectDetector
from app.vision.motion_analyzer import MotionAnalyzer
from app.vision.preprocessing import FramePreprocessor
from app.vision.tracker import ObjectTracker, UltralyticsTracker
from app.vision.types import Detection, ObjectState
from app.vision.zone_analyzer import (
    Zone, ZoneAnalyzer, ZoneContext, ZonePresenceTracker, ZoneStatus, load_zones,
)

logger = logging.getLogger(__name__)

_PRUNE_EVERY_FRAMES = 60


@dataclass
class FrameResult:
    frame_id: int
    timestamp: float
    size: tuple[int, int]
    objects: list[ObjectState]
    events: list[AlertEvent]
    annotated: np.ndarray            # VisionShield view
    plain: Optional[np.ndarray] = None  # "typical detector" view (boxes + labels only)


@dataclass
class ImageResult:
    size: tuple[int, int]
    objects: list[ObjectState]
    annotated: np.ndarray
    zones: list[Zone] = field(default_factory=list)

    @property
    def max_level(self) -> Optional[RiskLevel]:
        levels = [o.assessment.level for o in self.objects if o.assessment]
        return max(levels, key=lambda lv: lv.rank) if levels else None


class ZoneProvider:
    """Holds the zones (reference-frame coordinates) and scales them to the processing size.

    Zones can be replaced while a session runs (live zone editing); the pipeline asks for the
    scaled analyzer on every frame, so a change applies on the next frame.
    """

    def __init__(self, zones: Iterable[Zone], ref_size: tuple[float, float],
                 near_fraction: float) -> None:
        self._lock = threading.Lock()
        self._near_fraction = near_fraction
        self._cache: dict[tuple[int, int], ZoneAnalyzer] = {}
        self._base = ZoneAnalyzer(zones, near_distance_px=0.0)
        self._ref = ref_size

    @property
    def zones(self) -> tuple[Zone, ...]:
        """Zones in reference-frame coordinates."""
        return self._base.zones

    @property
    def reference_size(self) -> tuple[float, float]:
        return self._ref

    def set_zones(self, zones: Iterable[Zone], ref_size: Optional[tuple[float, float]] = None) -> None:
        with self._lock:
            self._base = ZoneAnalyzer(zones, near_distance_px=0.0)
            if ref_size is not None:
                self._ref = ref_size
            self._cache = {}

    def for_size(self, width: int, height: int) -> ZoneAnalyzer:
        with self._lock:
            key = (width, height)
            if key not in self._cache:
                near = self._near_fraction * math.hypot(width, height)
                self._cache[key] = self._base.scaled(width / self._ref[0], height / self._ref[1], near)
            return self._cache[key]


def _contextualise(zones: ZoneAnalyzer, centroid: tuple[float, float]
                   ) -> tuple[tuple[ZoneContext, ...], Optional[ZoneContext], Optional[ZoneContext]]:
    """Return (all contexts, nearest restricted zone for scoring, zone to display)."""
    contexts = tuple(zones.analyze(centroid))
    nearest_restricted = ZoneAnalyzer.nearest_restricted(contexts)
    inside = [c for c in contexts if c.status is ZoneStatus.INSIDE]
    display = inside[0] if inside else (min(contexts, key=lambda c: c.distance_px) if contexts else None)
    return contexts, nearest_restricted, display


def _restricted_distance(state: ObjectState) -> Optional[float]:
    nearest = ZoneAnalyzer.nearest_restricted(state.zone_contexts)
    return nearest.distance_px if nearest else None


def _score(scorer: ThreatScorer, state: ObjectState, diag: float, temporal: bool):
    det = state.detection
    adjustment = BehaviorAnalyzer.risk_adjustment(scorer.class_risk(det.class_name), state.behaviors)
    return scorer.score(ThreatInputs(
        class_name=det.class_name,
        confidence=min(1.0, max(0.0, det.confidence)),
        frame_diagonal_px=diag,
        # With a single observation (or a still image) speed is unknown rather than zero.
        speed_px_s=state.motion.speed_px_s if temporal and state.motion.observations >= 2 else None,
        approaching=state.motion.approaching if temporal else None,
        zone_distance_px=_restricted_distance(state),
        dwell_s=state.dwell_s if temporal else None,
        risk_adjustment=adjustment,
    ))


class VisionPipeline:
    """Per-session video pipeline. Not thread-safe: one capture loop drives it."""

    def __init__(
        self,
        tracker: ObjectTracker,
        zone_provider: ZoneProvider,
        scorer: ThreatScorer,
        alerts: AlertManager,
        preprocessor: FramePreprocessor,
        motion: MotionAnalyzer,
        presence: ZonePresenceTracker,
        behavior: Optional[BehaviorAnalyzer] = None,
        stale_after_s: float = 5.0,
        render_plain: bool = True,
    ) -> None:
        self.tracker = tracker
        self.zone_provider = zone_provider
        self.scorer = scorer
        self.alerts = alerts
        self.preprocessor = preprocessor
        self.motion = motion
        self.presence = presence
        self.behavior = behavior or BehaviorAnalyzer()
        self.stale_after_s = stale_after_s
        self.render_plain = render_plain
        self._frames = 0

    def process_frame(self, frame: np.ndarray, frame_id: int, timestamp: float) -> FrameResult:
        prepared = self.preprocessor.prepare(frame)
        image = prepared.image
        width, height = prepared.size
        zones = self.zone_provider.for_size(width, height)
        diag = math.hypot(width, height)

        # 1. observe: tracking + motion + zone context for every tracked object
        detections = self.tracker.update(image, frame_id, timestamp)
        observed: list[tuple[ObjectState, list[str]]] = [
            self._observe(det, zones, timestamp) for det in detections]

        # 2. behaviours need the whole scene (unattended items, crowding)
        tags = self.behavior.analyze([s for s, _ in observed], timestamp, diag)

        # 3. score with context-adjusted risk, then alert
        objects: list[ObjectState] = []
        events: list[AlertEvent] = []
        for state, entered in observed:
            state = replace(state, behaviors=tags.get(state.track_id, ()))
            state = replace(state, assessment=_score(self.scorer, state, diag, temporal=True))
            objects.append(state)
            events.extend(self.alerts.process(state, entered, timestamp))

        self._frames += 1
        if self._frames % _PRUNE_EVERY_FRAMES == 0:
            self.motion.prune(timestamp, self.stale_after_s)
            self.presence.prune(timestamp, self.stale_after_s)
            self.alerts.prune(timestamp, self.stale_after_s)

        plain = None
        if self.render_plain:
            plain = image.copy()
            annotator.draw_plain(plain, objects)
        annotator.draw_zones(image, zones.zones)
        annotator.draw_objects(image, objects)
        annotator.draw_banner(image, f"frame {frame_id}  t={timestamp:6.1f}s  objects={len(objects)}")
        events.sort(key=lambda e: -e.priority)
        return FrameResult(frame_id, timestamp, (width, height), objects, events, image, plain)

    def _observe(self, det: Detection, zones: ZoneAnalyzer, t: float) -> tuple[ObjectState, list[str]]:
        tid = det.track_id
        assert tid is not None
        contexts, restricted, display = _contextualise(zones, det.centroid)
        inside_ids = [c.zone_id for c in contexts if c.status is ZoneStatus.INSIDE]
        entered = self.presence.update(tid, t, inside_ids)
        motion = self.motion.update(tid, t, det.centroid, restricted.distance_px if restricted else None)
        dwell = self.presence.max_dwell_s(tid, t)
        return ObjectState(det, motion, contexts, display, dwell), entered


class ImageAnalyzer:
    """Single-image analysis. No temporal data exists, so movement and persistence are
    reported as unknown (they contribute 0 to the score and are flagged as such)."""

    def __init__(self, detector: ObjectDetector, zone_provider: ZoneProvider,
                 scorer: ThreatScorer, preprocessor: FramePreprocessor) -> None:
        self.detector = detector
        self.zone_provider = zone_provider
        self.scorer = scorer
        self.preprocessor = preprocessor

    def analyze(self, frame: np.ndarray) -> ImageResult:
        prepared = self.preprocessor.prepare(frame)
        image = prepared.image
        width, height = prepared.size
        zones = self.zone_provider.for_size(width, height)
        diag = math.hypot(width, height)
        objects: list[ObjectState] = []
        for det in self.detector.detect(image, 0, 0.0):
            contexts, _restricted, display = _contextualise(zones, det.centroid)
            state = ObjectState(det, zone_contexts=contexts, primary_zone=display)
            objects.append(replace(state, assessment=_score(self.scorer, state, diag, temporal=False)))
        annotator.draw_zones(image, zones.zones)
        annotator.draw_objects(image, objects, trajectories=False)
        return ImageResult((width, height), objects, image, list(zones.zones))


# -- factories ---------------------------------------------------------------------------
def build_scorer(settings: Settings) -> ThreatScorer:
    return ThreatScorer(
        weights=ThreatWeights.from_tuple(settings.weights_tuple()),
        object_risk=settings.object_risk,
        default_risk=settings.default_object_risk,
        speed_ref_diag_per_s=settings.speed_ref_diag_per_s,
        zone_near_fraction=settings.zone_near_fraction,
        dwell_ref_s=settings.dwell_ref_s,
    )


def build_zone_provider(settings: Settings, zones_path: Optional[Path] = None) -> ZoneProvider:
    path = zones_path if zones_path is not None else settings.resolve(settings.zones_file)
    zones, ref_size = load_zones(path)
    return ZoneProvider(zones, ref_size, settings.zone_near_fraction)


def build_behavior(settings: Settings) -> BehaviorAnalyzer:
    return BehaviorAnalyzer(BehaviorConfig(
        loiter_s=settings.behavior_loiter_s,
        unattended_s=settings.behavior_unattended_s,
        crowd_count=settings.behavior_crowd_count,
        run_speed_diag_per_s=settings.behavior_run_speed_diag_per_s,
    ))


def build_pipeline(
    settings: Settings,
    detector: ObjectDetector,
    zone_provider: ZoneProvider,
    session_id: Optional[int] = None,
    sinks: Iterable[EventSink] = (),
    tracker: Optional[ObjectTracker] = None,
    render_plain: bool = True,
) -> VisionPipeline:
    tracker = tracker or UltralyticsTracker(detector, settings.tracker_config)
    alerts = AlertManager(
        cooldown_s=settings.alert_cooldown_s,
        min_level=RiskLevel(settings.alert_min_level),
        sinks=sinks,
        session_id=session_id,
    )
    return VisionPipeline(
        tracker=tracker,
        zone_provider=zone_provider,
        scorer=build_scorer(settings),
        alerts=alerts,
        preprocessor=FramePreprocessor(settings.max_frame_width),
        motion=MotionAnalyzer(
            history_len=settings.trajectory_len,
            min_observations=settings.approach_min_observations,
            min_motion_px=3.0,
        ),
        presence=ZonePresenceTracker(settings.zone_exit_grace_s),
        behavior=build_behavior(settings),
        stale_after_s=settings.track_stale_s,
        render_plain=render_plain,
    )
