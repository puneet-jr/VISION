from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_service
from app.schemas.objects import ZoneOut, ZonesIn, ZonesOut
from app.services.surveillance_service import SurveillanceService
from app.vision.zone_analyzer import Zone

router = APIRouter(prefix="/api/v1", tags=["zones"])


def _out(service: SurveillanceService) -> ZonesOut:
    provider = service.zone_provider
    return ZonesOut(
        reference_size=list(provider.reference_size),
        zones=[ZoneOut(**z.to_dict()) for z in provider.zones],
    )


@router.get("/zones", response_model=ZonesOut)
def list_zones(service: SurveillanceService = Depends(get_service)) -> ZonesOut:
    """Configured zones, in coordinates of the reference frame size."""
    return _out(service)


@router.put("/zones", response_model=ZonesOut)
def replace_zones(body: ZonesIn, service: SurveillanceService = Depends(get_service)) -> ZonesOut:
    """Replace all zones. Applies to a running session on the next frame and is saved to disk."""
    zones = [Zone(z.id, z.name, tuple((float(x), float(y)) for x, y in z.polygon), z.restricted)
             for z in body.zones]
    service.update_zones(zones, (body.reference_size[0], body.reference_size[1]))
    return _out(service)


@router.post("/zones/reset", response_model=ZonesOut)
def reset_zones(service: SurveillanceService = Depends(get_service)) -> ZonesOut:
    """Restore the built-in demo zone."""
    service.reset_zones()
    return _out(service)
