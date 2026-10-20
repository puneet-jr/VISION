"""Controlled uploads: still images for one-off analysis, videos for later sessions."""
from __future__ import annotations

import base64
import uuid
from pathlib import Path

import cv2
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.dependencies import get_service, get_settings_dep
from app.core.config import Settings
from app.schemas.objects import ImageAnalysisOut, TrackedObjectOut
from app.services.surveillance_service import SurveillanceService
from app.vision import annotator
from app.vision.image_source import ALLOWED_IMAGE_EXTENSIONS

router = APIRouter(prefix="/api/v1", tags=["uploads"])

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
CHUNK = 1024 * 1024
IMAGE_NOTE = ("Single image: movement, approach direction and persistence need temporal data "
              "and are reported as unknown (they contribute 0 to the score).")


def _extension(file: UploadFile) -> str:
    return Path(file.filename or "").suffix.lower()


@router.post("/images/analyze", response_model=ImageAnalysisOut)
def analyze_image(file: UploadFile = File(...), service: SurveillanceService = Depends(get_service),
                  settings: Settings = Depends(get_settings_dep)) -> ImageAnalysisOut:
    if _extension(file) not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(415, f"unsupported image type; allowed: {sorted(ALLOWED_IMAGE_EXTENSIONS)}")
    limit = settings.max_image_upload_mb * 1024 * 1024
    data = file.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"image larger than {settings.max_image_upload_mb} MB")

    result = service.analyze_image(data)  # InvalidFrameError -> 422 via the exception handlers
    jpeg = annotator.encode_jpeg(result.annotated, settings.jpeg_quality) or b""
    level = result.max_level
    return ImageAnalysisOut(
        width=result.size[0], height=result.size[1], object_count=len(result.objects),
        max_level=level.value if level else None, note=IMAGE_NOTE,
        objects=[TrackedObjectOut(**o.to_dict()) for o in result.objects],
        annotated_image="data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii"),
    )


@router.post("/videos/upload", status_code=201)
def upload_video(file: UploadFile = File(...), settings: Settings = Depends(get_settings_dep)) -> dict:
    """Store a video under a generated name in the video directory; select it when starting a session."""
    ext = _extension(file)
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(415, f"unsupported video type; allowed: {sorted(ALLOWED_VIDEO_EXTENSIONS)}")
    folder = settings.video_path
    folder.mkdir(parents=True, exist_ok=True)
    name = f"upload_{uuid.uuid4().hex[:12]}{ext}"
    target = folder / name
    limit = settings.max_video_upload_mb * 1024 * 1024
    written = 0
    try:
        with open(target, "wb") as out:
            while chunk := file.file.read(CHUNK):
                written += len(chunk)
                if written > limit:
                    raise HTTPException(413, f"video larger than {settings.max_video_upload_mb} MB")
                out.write(chunk)
        cap = cv2.VideoCapture(str(target))
        readable = cap.isOpened() and cap.read()[0]
        cap.release()
        if not readable:
            raise HTTPException(422, "file is not a readable video")
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    return {"video_name": name, "size_bytes": written}
