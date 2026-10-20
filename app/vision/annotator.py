"""Draws zones, boxes, labels and trajectories onto frames."""
from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

from app.vision.types import ObjectState
from app.vision.zone_analyzer import Zone

# BGR colours
LEVEL_COLORS = {"LOW": (80, 175, 76), "MEDIUM": (0, 165, 255), "HIGH": (40, 40, 230)}
UNSCORED_COLOR = (200, 200, 200)
ZONE_COLOR = (60, 60, 220)
MONITORED_ZONE_COLOR = (200, 150, 40)


def draw_zones(image: np.ndarray, zones: Sequence[Zone]) -> None:
    if not zones:
        return
    overlay = image.copy()
    for z in zones:
        pts = np.array(z.polygon, dtype=np.int32)
        color = ZONE_COLOR if z.restricted else MONITORED_ZONE_COLOR
        cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, 0.18, image, 0.82, 0, dst=image)
    for z in zones:
        pts = np.array(z.polygon, dtype=np.int32)
        color = ZONE_COLOR if z.restricted else MONITORED_ZONE_COLOR
        cv2.polylines(image, [pts], True, color, 2, cv2.LINE_AA)
        cv2.putText(image, z.name, (int(pts[0][0]) + 4, int(pts[0][1]) + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def draw_objects(image: np.ndarray, objects: Sequence[ObjectState],
                 trajectories: bool = True) -> None:
    for obj in objects:
        d = obj.detection
        level = obj.assessment.level.value if obj.assessment else None
        color = LEVEL_COLORS.get(level, UNSCORED_COLOR) if level else UNSCORED_COLOR
        x1, y1, x2, y2 = (int(v) for v in d.bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        if trajectories and len(obj.motion.trajectory) > 1:
            pts = np.array(obj.motion.trajectory, dtype=np.int32)
            cv2.polylines(image, [pts], False, color, 1, cv2.LINE_AA)

        idtxt = f"#{d.track_id} " if d.track_id is not None else ""
        label = f"{idtxt}{d.class_name} {d.confidence:.2f}"
        if obj.assessment:
            label += f" | {obj.assessment.score:.0f} {obj.assessment.level.value}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(0, y1 - th - base - 4)
        cv2.rectangle(image, (x1, top), (x1 + tw + 6, top + th + base + 4), color, -1)
        cv2.putText(image, label, (x1 + 3, top + th + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)


def draw_banner(image: np.ndarray, text: str) -> None:
    cv2.rectangle(image, (0, 0), (image.shape[1], 24), (30, 30, 30), -1)
    cv2.putText(image, text, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)


def placeholder_frame(message: str, size: tuple[int, int] = (640, 360)) -> np.ndarray:
    """A neutral frame shown when no source is active or the source is unavailable."""
    w, h = size
    image = np.full((h, w, 3), 28, dtype=np.uint8)
    (tw, _), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(image, message, (max(10, (w - tw) // 2), h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (200, 200, 200), 2, cv2.LINE_AA)
    return image


def encode_jpeg(image: np.ndarray, quality: int = 80) -> Optional[bytes]:
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes() if ok else None
