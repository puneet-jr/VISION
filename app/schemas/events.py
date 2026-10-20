from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict

from app.schemas.common import UtcDatetime


class ObservationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    track_id: int
    class_name: str
    bbox: list[float]
    centroid_x: float
    centroid_y: float
    speed_px_s: float
    dwell_s: Optional[float] = None


class EventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    session_id: int
    timestamp: UtcDatetime
    event_type: str
    track_id: int
    class_name: str
    confidence: float
    score: float
    level: str
    zone_id: Optional[str] = None
    zone_name: Optional[str] = None
    movement: dict
    factors: dict
    explanation: str
    reason: str
    observation: Optional[ObservationOut] = None


class EventListOut(BaseModel):
    items: list[EventOut]
    total: int
    page: int
    page_size: int


class TimelineBucket(BaseModel):
    bucket: str
    LOW: int = 0
    MEDIUM: int = 0
    HIGH: int = 0


class AnalyticsSummaryOut(BaseModel):
    total: int
    by_level: dict[str, int]
    by_class: dict[str, int]
    by_event_type: dict[str, int]
    timeline: list[TimelineBucket]
