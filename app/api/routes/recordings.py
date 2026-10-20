"""Saved session recordings: list, replay (MJPEG), download, delete.

Recordings are addressed by session id only; the file name is looked up server-side, so a
request can never name an arbitrary path.
"""
from __future__ import annotations

import asyncio

import cv2
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from app.api.dependencies import get_service, get_session_repo
from app.db.repository import SessionRepository
from app.schemas.sessions import RecordingOut
from app.services.surveillance_service import SurveillanceService

router = APIRouter(prefix="/api/v1/recordings", tags=["recordings"])


def _path_or_404(service: SurveillanceService, session_id: int):
    path = service.recording_path(session_id)
    if path is None:
        raise HTTPException(404, f"no finished recording for session {session_id}")
    return path


@router.get("", response_model=list[RecordingOut])
def list_recordings(repo: SessionRepository = Depends(get_session_repo),
                    service: SurveillanceService = Depends(get_service)) -> list[RecordingOut]:
    out = []
    for row in repo.with_recordings():
        path = service.recording_path(row.id)
        if path is not None:
            out.append(RecordingOut(session_id=row.id, source_label=row.source_label,
                                    started_at=row.started_at, seconds=row.recording_seconds,
                                    size_bytes=path.stat().st_size))
    return out


@router.get("/{session_id}/download")
def download_recording(session_id: int, service: SurveillanceService = Depends(get_service)) -> FileResponse:
    path = _path_or_404(service, session_id)
    return FileResponse(path, media_type="video/x-msvideo", filename=f"visionshield_session_{session_id}.avi")


@router.get("/{session_id}/replay")
async def replay_recording(session_id: int, request: Request,
                           service: SurveillanceService = Depends(get_service)) -> StreamingResponse:
    """Plays the recording back as an MJPEG stream (viewable in an <img> tag) at its real speed."""
    path = _path_or_404(service, session_id)

    async def generate():
        cap = cv2.VideoCapture(str(path))
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
            delay = 1.0 / max(1.0, min(fps, 60.0))
            while not await request.is_disconnected():
                ok, frame = await run_in_threadpool(cap.read)
                if not ok:
                    break
                ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if not ok:
                    continue
                data = buf.tobytes()
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
                await asyncio.sleep(delay)
        finally:
            cap.release()

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame",
                             headers={"Cache-Control": "no-store"})


@router.delete("/{session_id}", status_code=204)
def delete_recording(session_id: int, service: SurveillanceService = Depends(get_service),
                     repo: SessionRepository = Depends(get_session_repo)) -> None:
    path = _path_or_404(service, session_id)
    path.unlink(missing_ok=True)
    repo.clear_recording(session_id)
