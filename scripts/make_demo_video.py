"""Create a short demo clip by panning across a still photo (no camera needed).

Objects in the photo drift across the frame, which gives ByteTrack stable ids, a trajectory
and a speed so the whole pipeline can be exercised repeatably. It is a smoke-test aid, not a
realistic surveillance scenario.

    python scripts/make_demo_video.py --image path/to/street_photo.jpg --out data/videos/demo_pan.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="photo containing people/vehicles")
    parser.add_argument("--out", type=Path, default=Path("data/videos/demo_pan.mp4"))
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    args = parser.parse_args()

    photo = cv2.imread(str(args.image))
    if photo is None:
        print(f"error: cannot read {args.image}")
        return 2
    # Scale so the photo is at least 1.6x the output width, leaving room to pan.
    scale = max(args.width * 1.6 / photo.shape[1], args.height / photo.shape[0])
    photo = cv2.resize(photo, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = photo.shape[:2]
    frames = int(args.seconds * args.fps)
    max_x, max_y = w - args.width, max(0, h - args.height)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                             (args.width, args.height))
    if not writer.isOpened():
        print("error: cannot open video writer")
        return 2
    for i in range(frames):
        t = i / max(1, frames - 1)
        x = int(max_x * t)
        y = int(max_y * 0.5)
        writer.write(np.ascontiguousarray(photo[y:y + args.height, x:x + args.width]))
    writer.release()
    print(f"wrote {args.out} ({frames} frames @ {args.fps} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
