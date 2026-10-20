from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


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
