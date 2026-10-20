from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_service
from app.schemas.objects import ZoneOut, ZonesOut
from app.services.surveillance_service import SurveillanceService

router = APIRouter(prefix="/api/v1", tags=["zones"])


@router.get("/zones", response_model=ZonesOut)
def list_zones(service: SurveillanceService = Depends(get_service)) -> ZonesOut:
    """Configured zones, in coordinates of the reference frame size."""
    provider = service.zone_provider
    return ZonesOut(
        reference_size=list(provider.reference_size),
        zones=[ZoneOut(**z.to_dict()) for z in provider.zones],
    )
