"""Per-track movement analysis from consecutive centroid observations.

Speeds are in image space (pixels per second of the source timeline). They are NOT
real-world speeds: converting to metres per second requires camera calibration.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

Point = tuple[float, float]

_COMPASS = ["right", "up-right", "up", "up-left", "left", "down-left", "down", "down-right"]


@dataclass(frozen=True)
class MotionState:
    observations: int
    speed_px_s: float
    displacement_px: float
    direction_deg: Optional[float]  # 0 = right, 90 = up (image y axis inverted)
    direction: str  # compass label, "stationary" or "unknown"
    approach: str  # "approaching" | "receding" | "steady" | "unknown"
    trajectory: tuple[Point, ...] = field(default_factory=tuple)

    @property
    def approaching(self) -> Optional[bool]:
        """True/False once enough evidence exists, None while unknown."""
        if self.approach == "unknown":
            return None
        return self.approach == "approaching"

    def to_dict(self) -> dict:
        return {
            "observations": self.observations,
            "speed_px_s": round(self.speed_px_s, 2),
            "displacement_px": round(self.displacement_px, 2),
            "direction_deg": None if self.direction_deg is None else round(self.direction_deg, 1),
            "direction": self.direction,
            "approach": self.approach,
        }


UNKNOWN_MOTION = MotionState(0, 0.0, 0.0, None, "unknown", "unknown")


def direction_label(angle_deg: float) -> str:
    return _COMPASS[int(((angle_deg % 360) + 22.5) // 45) % 8]


@dataclass
class _Obs:
    t: float
    x: float
    y: float
    zone_distance: Optional[float]


class MotionAnalyzer:
    """Keeps a bounded history per track and derives speed, direction and zone approach."""

    def __init__(
        self,
        history_len: int = 30,
        speed_window: int = 10,
        min_observations: int = 5,
        min_motion_px: float = 3.0,
    ) -> None:
        if history_len < 2 or speed_window < 2 or min_observations < 2:
            raise ValueError("history_len, speed_window and min_observations must be >= 2")
        self.history_len = history_len
        self.speed_window = speed_window
        self.min_observations = min_observations
        self.min_motion_px = min_motion_px
        self._history: dict[int, Deque[_Obs]] = {}

    def update(self, track_id: int, t: float, centroid: Point,
               zone_distance: Optional[float] = None) -> MotionState:
        hist = self._history.setdefault(track_id, deque(maxlen=self.history_len))
        obs = _Obs(t, float(centroid[0]), float(centroid[1]), zone_distance)
        if hist and t <= hist[-1].t:
            # Repeated or out-of-order timestamp: replace the latest sample, never add dt <= 0.
            if t == hist[-1].t:
                hist[-1] = obs
            return self._state(hist)
        hist.append(obs)
        return self._state(hist)

    def state(self, track_id: int) -> MotionState:
        hist = self._history.get(track_id)
        return self._state(hist) if hist else UNKNOWN_MOTION

    def _state(self, hist: Deque[_Obs]) -> MotionState:
        n = len(hist)
        trajectory = tuple((o.x, o.y) for o in hist)
        if n < 2:
            return MotionState(n, 0.0, 0.0, None, "unknown", "unknown", trajectory)

        window = list(hist)[-self.speed_window:]
        first, last = window[0], window[-1]
        dt = last.t - first.t
        dx, dy = last.x - first.x, last.y - first.y
        displacement = math.hypot(dx, dy)
        speed = displacement / dt if dt > 0 else 0.0
        if displacement < self.min_motion_px:
            direction, angle = "stationary", None
        else:
            angle = math.degrees(math.atan2(-dy, dx)) % 360
            direction = direction_label(angle)
        return MotionState(n, speed, displacement, angle, direction, self._approach(hist), trajectory)

    def _approach(self, hist: Deque[_Obs]) -> str:
        samples = [o.zone_distance for o in hist if o.zone_distance is not None]
        if len(samples) < self.min_observations:
            return "unknown"
        recent = samples[-self.speed_window:]
        delta = recent[-1] - recent[0]
        if delta < -self.min_motion_px:
            return "approaching"
        if delta > self.min_motion_px:
            return "receding"
        return "steady"

    def prune(self, now: float, stale_after_s: float) -> None:
        for tid in [t for t, h in self._history.items() if not h or now - h[-1].t > stale_after_s]:
            del self._history[tid]

    def __len__(self) -> int:
        return len(self._history)
