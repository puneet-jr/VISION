from __future__ import annotations

import math
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class TrackedObjectOut(BaseModel):
    track_id: Optional[int] = None
    class_id: int
    class_name: str
    confidence: float
    bbox: list[float]
    centroid: list[float]
    speed_px_s: float = 0.0
    direction: str = "unknown"
    approach: str = "unknown"
    zone_id: Optional[str] = None
    zone_name: Optional[str] = None
    zone_status: Optional[str] = None
    zone_distance_px: Optional[float] = None
    dwell_s: Optional[float] = None
    score: Optional[float] = None
    level: Optional[str] = None
    factors: dict = {}
    explanation: str = ""
    behaviors: list[dict] = []


class ObjectsOut(BaseModel):
    session_id: Optional[int] = None
    active: bool
    objects: list[TrackedObjectOut]


class ImageAnalysisOut(BaseModel):
    width: int
    height: int
    object_count: int
    max_level: Optional[str] = None
    note: str
    objects: list[TrackedObjectOut]
    annotated_image: str  # data URI (JPEG)


class ZoneOut(BaseModel):
    id: str
    name: str
    restricted: bool
    polygon: list[list[float]]


class ZonesOut(BaseModel):
    reference_size: list[float]
    zones: list[ZoneOut]


class ZoneIn(BaseModel):
    """A zone as drawn in the dashboard: polygon in reference-frame coordinates."""

    id: str = Field(min_length=1, max_length=64, pattern=r"^[\w\-. ]+$")
    name: str = Field(min_length=1, max_length=80)
    restricted: bool = True
    polygon: list[list[float]] = Field(min_length=3, max_length=60)

    @field_validator("polygon")
    @classmethod
    def _points(cls, v: list[list[float]]) -> list[list[float]]:
        for pt in v:
            if len(pt) != 2 or not all(math.isfinite(c) for c in pt):
                raise ValueError("each polygon point must be two finite numbers [x, y]")
        return v


class ZonesIn(BaseModel):
    reference_size: list[float] = Field(default=[1280.0, 720.0], min_length=2, max_length=2)
    zones: list[ZoneIn] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def _within_reference(self) -> "ZonesIn":
        w, h = self.reference_size
        if not (0 < w <= 10000 and 0 < h <= 10000):
            raise ValueError("reference_size must be within (0, 10000]")
        ids = [z.id for z in self.zones]
        if len(ids) != len(set(ids)):
            raise ValueError("zone ids must be unique")
        for z in self.zones:
            if any(not (-0.01 * w <= x <= 1.01 * w and -0.01 * h <= y <= 1.01 * h) for x, y in z.polygon):
                raise ValueError(f"zone '{z.id}' has points outside the reference frame")
        return self
