"""Spatial zone geometry and per-track zone presence (entry / dwell time).

Zones are polygons in image coordinates, declared against a reference frame size
and scaled to the actual processing resolution.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional, Sequence

from app.core.exceptions import ConfigError

Point = tuple[float, float]

DEFAULT_ZONES_DATA: dict = {
    "reference_size": [1280, 720],
    "zones": [
        {
            "id": "zone-1",
            "name": "Restricted Area A",
            "restricted": True,
            "polygon": [[640, 180], [1200, 180], [1200, 680], [640, 680]],
        }
    ],
}


class ZoneStatus(str, Enum):
    INSIDE = "inside"
    NEAR = "near"
    OUTSIDE = "outside"


@dataclass(frozen=True)
class Zone:
    zone_id: str
    name: str
    polygon: tuple[Point, ...]
    restricted: bool = True

    def __post_init__(self) -> None:
        if len(self.polygon) < 3:
            raise ConfigError(f"zone '{self.zone_id}' needs at least 3 polygon points")

    def to_dict(self) -> dict:
        return {
            "id": self.zone_id,
            "name": self.name,
            "restricted": self.restricted,
            "polygon": [list(p) for p in self.polygon],
        }


@dataclass(frozen=True)
class ZoneContext:
    zone_id: str
    zone_name: str
    restricted: bool
    status: ZoneStatus
    distance_px: float  # 0 when inside, otherwise distance to the polygon boundary


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Ray casting; boundary points count as inside."""
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if _on_segment(point, (x1, y1), (x2, y2)):
            return True
        if (y1 > y) != (y2 > y):
            x_cross = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < x_cross:
                inside = not inside
    return inside


def _on_segment(p: Point, a: Point, b: Point, eps: float = 1e-9) -> bool:
    return _point_segment_distance(p, a, b) <= eps


def _point_segment_distance(p: Point, a: Point, b: Point) -> float:
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return math.hypot(p[0] - ax, p[1] - ay)
    t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / length_sq))
    return math.hypot(p[0] - (ax + t * dx), p[1] - (ay + t * dy))


def distance_to_polygon(point: Point, polygon: Sequence[Point]) -> float:
    """0 if inside/on boundary, else the distance to the nearest edge."""
    if point_in_polygon(point, polygon):
        return 0.0
    n = len(polygon)
    return min(_point_segment_distance(point, polygon[i], polygon[(i + 1) % n]) for i in range(n))


class ZoneAnalyzer:
    """Stateless geometric analysis of a point against configured zones."""

    def __init__(self, zones: Iterable[Zone], near_distance_px: float) -> None:
        self.zones: tuple[Zone, ...] = tuple(zones)
        if near_distance_px < 0:
            raise ConfigError("near_distance_px must be >= 0")
        self.near_distance_px = near_distance_px

    def scaled(self, sx: float, sy: float, near_distance_px: Optional[float] = None) -> "ZoneAnalyzer":
        zones = [
            Zone(z.zone_id, z.name, tuple((x * sx, y * sy) for x, y in z.polygon), z.restricted)
            for z in self.zones
        ]
        return ZoneAnalyzer(zones, self.near_distance_px if near_distance_px is None else near_distance_px)

    def analyze(self, point: Point) -> list[ZoneContext]:
        out: list[ZoneContext] = []
        for z in self.zones:
            dist = distance_to_polygon(point, z.polygon)
            if dist == 0.0:
                status = ZoneStatus.INSIDE
            elif dist <= self.near_distance_px:
                status = ZoneStatus.NEAR
            else:
                status = ZoneStatus.OUTSIDE
            out.append(ZoneContext(z.zone_id, z.name, z.restricted, status, dist))
        return out

    @staticmethod
    def nearest_restricted(contexts: Sequence[ZoneContext]) -> Optional[ZoneContext]:
        restricted = [c for c in contexts if c.restricted]
        return min(restricted, key=lambda c: c.distance_px) if restricted else None


def load_zones(path: Optional[Path]) -> tuple[list[Zone], tuple[float, float]]:
    """Load zones from JSON; fall back to the built-in demo zone if the file is absent."""
    if path is not None and path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(f"cannot read zones file {path.name}: {exc}") from exc
    else:
        data = DEFAULT_ZONES_DATA
    return parse_zones(data)


def parse_zones(data: dict) -> tuple[list[Zone], tuple[float, float]]:
    try:
        ref = data.get("reference_size", [1280, 720])
        ref_size = (float(ref[0]), float(ref[1]))
        if ref_size[0] <= 0 or ref_size[1] <= 0:
            raise ValueError("reference_size must be positive")
        zones = [
            Zone(
                zone_id=str(z["id"]),
                name=str(z.get("name", z["id"])),
                polygon=tuple((float(x), float(y)) for x, y in z["polygon"]),
                restricted=bool(z.get("restricted", True)),
            )
            for z in data.get("zones", [])
        ]
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ConfigError(f"invalid zones definition: {exc}") from exc
    ids = [z.zone_id for z in zones]
    if len(ids) != len(set(ids)):
        raise ConfigError("zone ids must be unique")
    return zones, ref_size


@dataclass
class _Presence:
    entered_at: float
    last_inside_at: float


class ZonePresenceTracker:
    """Tracks zone entries and dwell time per (track, zone).

    A short absence (``exit_grace_s``) - border jitter or temporary track loss - does not
    reset dwell time or trigger a second entry.
    """

    def __init__(self, exit_grace_s: float = 2.0) -> None:
        self.exit_grace_s = exit_grace_s
        self._presence: dict[tuple[int, str], _Presence] = {}
        self._last_seen: dict[int, float] = {}

    def update(self, track_id: int, t: float, inside_zone_ids: Iterable[str]) -> list[str]:
        """Record an observation; return ids of zones newly entered at this observation."""
        self._last_seen[track_id] = t
        inside = set(inside_zone_ids)
        entered: list[str] = []
        for zone_id in inside:
            state = self._presence.get((track_id, zone_id))
            if state is None:
                self._presence[(track_id, zone_id)] = _Presence(t, t)
                entered.append(zone_id)
            else:
                state.last_inside_at = max(state.last_inside_at, t)
        for key in [k for k in self._presence if k[0] == track_id and k[1] not in inside]:
            if t - self._presence[key].last_inside_at > self.exit_grace_s:
                del self._presence[key]
        return sorted(entered)

    def dwell_s(self, track_id: int, zone_id: str, t: float) -> float:
        state = self._presence.get((track_id, zone_id))
        if state is None or t - state.last_inside_at > self.exit_grace_s:
            return 0.0
        return max(0.0, t - state.entered_at)

    def max_dwell_s(self, track_id: int, t: float) -> float:
        return max(
            (self.dwell_s(tid, zid, t) for (tid, zid) in self._presence if tid == track_id),
            default=0.0,
        )

    def prune(self, t: float, stale_after_s: float) -> None:
        """Drop state for tracks not observed for ``stale_after_s``."""
        stale = {tid for tid, seen in self._last_seen.items() if t - seen > stale_after_s}
        for tid in stale:
            del self._last_seen[tid]
        for key in [k for k in self._presence if k[0] in stale]:
            del self._presence[key]
