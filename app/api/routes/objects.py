from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_service
from app.schemas.objects import ObjectsOut, TrackedObjectOut
from app.services.surveillance_service import SurveillanceService

router = APIRouter(prefix="/api/v1", tags=["objects"])


@router.get("/objects", response_model=ObjectsOut)
def current_objects(service: SurveillanceService = Depends(get_service)) -> ObjectsOut:
    """Summaries of objects tracked on the latest processed frame."""
    return ObjectsOut(
        session_id=service.session_id,
        active=service.is_running,
        objects=[TrackedObjectOut(**o) for o in service.current_objects()],
    )
