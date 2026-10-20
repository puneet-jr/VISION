"""Offline dataset evaluation CLI.

Examples (from the project root, with the virtualenv active):
    python scripts/evaluate_dataset.py mot17 --root data/datasets --max-frames 200
    python scripts/evaluate_dataset.py visdrone --root data/datasets --max-images 100
    python scripts/evaluate_dataset.py ucf-crime --root data/datasets/UCF-Crime --max-videos 20

Datasets are never downloaded automatically; see docs/DATASETS.md for the expected layouts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import BASE_DIR, get_settings  # noqa: E402
from app.core.exceptions import VisionShieldError  # noqa: E402
from app.core.logging import configure_logging  # noqa: E402
from app.evaluation import runners  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="VisionShield AI offline evaluation")
    sub = parser.add_subparsers(dest="dataset", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--root", type=Path, required=True, help="dataset root folder")
        p.add_argument("--out", type=Path, default=BASE_DIR / "outputs" / "evaluation")

    mot = sub.add_parser("mot17", help="tracking metrics (needs motmetrics)")
    common(mot)
    mot.add_argument("--split", default="train")
    mot.add_argument("--sequences", nargs="*", help="substring filter, e.g. 02 04")
    mot.add_argument("--max-frames", type=int, default=0, help="0 = whole sequence")

    vis = sub.add_parser("visdrone", help="detection precision/recall on VisDrone2019-DET")
    common(vis)
    vis.add_argument("--split", default="val")
    vis.add_argument("--max-images", type=int, default=0)

    ucf = sub.add_parser("ucf-crime", help="video-level descriptive analysis on UCF-Crime")
    common(ucf)
    ucf.add_argument("--split-file", default="Anomaly_Test.txt")
    ucf.add_argument("--max-videos", type=int, default=0)
    ucf.add_argument("--frame-stride", type=int, default=5)

    args = parser.parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    try:
        if args.dataset == "mot17":
            result = runners.evaluate_mot17(settings, args.root, args.out, args.split,
                                            args.sequences, args.max_frames)
        elif args.dataset == "visdrone":
            result = runners.evaluate_visdrone_det(settings, args.root, args.split, args.max_images)
        else:
            result = runners.evaluate_ucf_crime(settings, args.root, args.out, args.split_file,
                                                args.max_videos, args.frame_stride)
    except VisionShieldError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
