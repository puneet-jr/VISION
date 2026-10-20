"""Annotated video delivery: an MJPEG stream and a single-frame snapshot.

Frames are produced once by the capture loop; every client only reads the latest JPEG,
so extra viewers never start extra captures. Frames never travel over the event WebSocket.
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from app.api.dependencies import get_service, get_settings_dep
from app.core.config import Settings
from app.services.surveillance_service import SurveillanceService
from app.vision import annotator

router = APIRouter(prefix="/api/v1", tags=["video"])

BOUNDARY = "frame"


@router.get("/frame")
def latest_frame(service: SurveillanceService = Depends(get_service)) -> Response:
    jpeg, _ = service.latest_frame()
    if jpeg is None:
        raise HTTPException(status_code=404, detail="no frame has been processed yet")
    return Response(content=jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.get("/stream")
async def stream(request: Request, service: SurveillanceService = Depends(get_service),
                 settings: Settings = Depends(get_settings_dep)) -> StreamingResponse:
    async def generate():
        last = -1
        while not await request.is_disconnected():
            jpeg, seq = service.latest_frame()
            if jpeg is None:
                jpeg = annotator.encode_jpeg(annotator.placeholder_frame("No active source"),
                                             settings.jpeg_quality)
                seq = 0
            if seq != last:
                last = seq
                yield (f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                       f"Content-Length: {len(jpeg)}\r\n\r\n").encode() + jpeg + b"\r\n"
            elif not service.is_running:
                break  # final frame delivered and nothing more will come
            await asyncio.sleep(0.04)

    return StreamingResponse(generate(), media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
                             headers={"Cache-Control": "no-store"})
