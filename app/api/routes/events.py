from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.dependencies import get_event_repo
from app.db.repository import EventRepository
from app.schemas.common import to_naive_utc
from app.schemas.events import EventListOut, EventOut

router = APIRouter(prefix="/api/v1", tags=["events"])


@router.get("/events", response_model=EventListOut)
def list_events(
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=200),
    level: Optional[Literal["LOW", "MEDIUM", "HIGH"]] = None,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    session_id: Optional[int] = Query(None, ge=1),
    repo: EventRepository = Depends(get_event_repo),
) -> EventListOut:
    start_utc, end_utc = to_naive_utc(start), to_naive_utc(end)
    if start_utc and end_utc and start_utc > end_utc:
        raise HTTPException(status_code=422, detail="'start' must not be after 'end'")
    rows, total = repo.list(page, page_size, level, start_utc, end_utc, session_id)
    return EventListOut(items=[EventOut.model_validate(r) for r in rows], total=total,
                        page=page, page_size=page_size)


@router.get("/events/{event_id}", response_model=EventOut)
def get_event(event_id: int, repo: EventRepository = Depends(get_event_repo)) -> EventOut:
    row = repo.get(event_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"event {event_id} not found")
    return EventOut.model_validate(row)
