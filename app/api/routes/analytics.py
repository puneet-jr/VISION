from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_event_repo
from app.db.repository import EventRepository
from app.schemas.common import to_naive_utc
from app.schemas.events import AnalyticsSummaryOut

router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])


@router.get("/summary", response_model=AnalyticsSummaryOut)
def summary(start: Optional[datetime] = None, end: Optional[datetime] = None,
            repo: EventRepository = Depends(get_event_repo)) -> AnalyticsSummaryOut:
    """Event counts by classification, class and hour within an optional time range."""
    start_utc, end_utc = to_naive_utc(start), to_naive_utc(end)
    if start_utc and end_utc and start_utc > end_utc:
        raise HTTPException(status_code=422, detail="'start' must not be after 'end'")
    return AnalyticsSummaryOut(**repo.summary(start_utc, end_utc))
