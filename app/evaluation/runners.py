"""Offline evaluation runners. These need local datasets and a detector; they are not run by
the application or the automated tests, and no results are shipped with the repository.
"""
from __future__ import annotations

import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2

from app.core.config import Settings
from app.core.exceptions import ConfigError
from app.evaluation import datasets
from app.evaluation.metrics import MatchCounts, iou_distance_matrix, match_boxes, roc_auc
from app.vision.detector import ObjectDetector, YoloDetector
from app.vision.pipeline import build_pipeline, build_zone_provider
from app.vision.tracker import UltralyticsTracker
from app.vision.video_source import SourceKind, SourceSpec, VideoSource

logger = logging.getLogger(__name__)


def evaluate_mot17(settings: Settings, root: Path, out_dir: Path, split: str = "train",
                   sequences: Optional[list[str]] = None, max_frames: int = 0,
                   detector: Optional[ObjectDetector] = None) -> dict:
    """Run detector+ByteTrack on MOT17 sequences and score with py-motmetrics.

    Reports tracking metrics for the *pretrained COCO* detector with default settings; it is
    not tuned on MOT17. Only 'person' tracks are evaluated.
    """
    try:
        import motmetrics as mm
    except ImportError as exc:
        raise ConfigError("motmetrics is required for MOT17 evaluation (pip install motmetrics)") from exc
    if split != "train":
        raise ConfigError("the MOT17 test split has no public ground truth; use --split train")

    detector = detector or YoloDetector(settings)
    seqs = datasets.find_mot17_sequences(root, split)
    if sequences:
        seqs = [s for s in seqs if any(key in s.name for key in sequences)]
    out_dir.mkdir(parents=True, exist_ok=True)

    accumulators, names = [], []
    for seq in seqs:
        tracker = UltralyticsTracker(detector, settings.tracker_config)
        gt = datasets.parse_mot_gt(seq.gt_file)
        acc = mm.MOTAccumulator(auto_id=True)
        rows: list[str] = []
        images = list(datasets.list_images(seq.image_dir))
        if max_frames:
            images = images[:max_frames]
        for idx, path in enumerate(images, start=1):
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            dets = [d for d in tracker.update(frame, idx, idx / seq.frame_rate) if d.class_name == "person"]
            for d in dets:
                x1, y1, x2, y2 = d.bbox
                rows.append(f"{idx},{d.track_id},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{d.confidence:.4f},-1,-1,-1")
            gts = gt.get(idx, [])
            acc.update([g.track_id for g in gts], [d.track_id for d in dets],
                       iou_distance_matrix([g.xyxy for g in gts], [d.bbox for d in dets]))
        (out_dir / f"{seq.name}.txt").write_text("\n".join(rows))
        accumulators.append(acc)
        names.append(seq.name)
        logger.info("MOT17 %s: %d frames evaluated", seq.name, len(images))

    mh = mm.metrics.create()
    summary = mh.compute_many(accumulators, metrics=mm.metrics.motchallenge_metrics,
                              names=names, generate_overall=True)
    text = mm.io.render_summary(summary, formatters=mh.formatters,
                                namemap=mm.io.motchallenge_metric_names)
    (out_dir / "mot17_summary.txt").write_text(text)
    return {"dataset": "MOT17", "split": split, "sequences": names, "max_frames": max_frames or "all",
            "summary_file": str(out_dir / "mot17_summary.txt"),
            "metrics": summary.round(4).to_dict(orient="index")}


def evaluate_visdrone_det(settings: Settings, root: Path, split: str = "val", max_images: int = 0,
                          detector: Optional[ObjectDetector] = None) -> dict:
    """Detection precision/recall/F1 at IoU 0.5 against VisDrone2019-DET after COCO class mapping.

    This is a thresholded P/R/F1 at the configured confidence, not COCO-style mAP.
    """
    images_dir, ann_dir = datasets.find_visdrone_det(root, split)
    detector = detector or YoloDetector(settings)
    per_class: dict[str, MatchCounts] = defaultdict(MatchCounts)
    overall = MatchCounts()
    evaluated = 0
    for path in datasets.list_images(images_dir):
        ann = datasets.annotation_for(path, ann_dir)
        frame = cv2.imread(str(path))
        if ann is None or frame is None:
            continue
        scored, ignored = datasets.parse_visdrone_annotations(ann)
        dets = detector.detect(frame, evaluated, 0.0)
        for label in set(datasets.VISDRONE_TO_COCO.values()):
            preds = [(d.bbox, d.confidence) for d in dets if d.class_name == label]
            gts = [g.xyxy for g in scored if g.label == label]
            if not preds and not gts:
                continue
            c = match_boxes(preds, gts, [g.xyxy for g in ignored])
            per_class[label].add(c)
            overall.add(c)
        evaluated += 1
        if max_images and evaluated >= max_images:
            break
    return {"dataset": "VisDrone2019-DET", "split": split, "images_evaluated": evaluated,
            "iou_threshold": 0.5, "overall": overall.to_dict(),
            "per_class": {k: v.to_dict() for k, v in sorted(per_class.items())}}


def evaluate_ucf_crime(settings: Settings, root: Path, out_dir: Path, split_file: str = "Anomaly_Test.txt",
                       max_videos: int = 0, frame_stride: int = 5,
                       detector: Optional[ObjectDetector] = None) -> dict:
    """Run the full pipeline per video and record the maximum threat score and alert counts.

    UCF-Crime has video-level labels only. The threat score is context-specific (zones) and
    is NOT trained for anomaly detection, so the reported AUC is a descriptive diagnostic of
    "max score vs. anomalous/normal", not evidence of anomaly-detection performance.
    """
    items = datasets.read_ucf_split(root, split_file)
    if max_videos:
        items = items[:max_videos]
    detector = detector or YoloDetector(settings)
    zones = build_zone_provider(settings)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for video, label in items:
        if not video.is_file():
            logger.warning("missing video %s", video.name)
            continue
        pipeline = build_pipeline(settings, detector, zones)
        source = VideoSource(SourceSpec(SourceKind.VIDEO, path=video))
        max_score, high_events, events = 0.0, 0, 0
        try:
            source.open()
            while (packet := source.read()) is not None:
                if packet.frame_id % max(1, frame_stride):
                    continue
                result = pipeline.process_frame(packet.image, packet.frame_id, packet.timestamp)
                for o in result.objects:
                    if o.assessment:
                        max_score = max(max_score, o.assessment.score)
                events += len(result.events)
                high_events += sum(1 for e in result.events if e.level.value == "HIGH")
        finally:
            source.close()
        rows.append({"video": video.name, "label": label, "max_score": round(max_score, 2),
                     "events": events, "high_events": high_events})
    csv_path = out_dir / "ucf_crime_videos.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["video", "label", "max_score", "events", "high_events"])
        writer.writeheader()
        writer.writerows(rows)
    auc = roc_auc([r["max_score"] for r in rows], [r["label"] for r in rows])
    return {"dataset": "UCF-Crime", "split_file": split_file, "videos_evaluated": len(rows),
            "frame_stride": frame_stride, "per_video_csv": str(csv_path),
            "descriptive_auc_max_score": None if auc is None else round(auc, 4),
            "caveat": "diagnostic only; the threat score is not trained for anomaly detection"}
