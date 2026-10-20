"""Dataset layouts and annotation parsers for optional offline evaluation.

Nothing here downloads data. Point ``--root`` at a folder you prepared yourself (see
docs/DATASETS.md). Only datasets whose annotations support a metric are evaluated with it.
"""
from __future__ import annotations

import configparser
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from app.core.exceptions import ConfigError

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

# VisDrone-DET category id -> COCO class name used by the pretrained detector.
VISDRONE_TO_COCO = {
    1: "person",   # pedestrian
    2: "person",   # people
    3: "bicycle",
    4: "car",
    5: "car",      # van
    6: "truck",
    9: "bus",
    10: "motorcycle",
}


@dataclass(frozen=True)
class GTBox:
    x: float
    y: float
    w: float
    h: float
    label: str = "person"
    track_id: int = -1

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)


@dataclass(frozen=True)
class MOTSequence:
    name: str
    path: Path
    frame_rate: float
    width: int
    height: int
    length: int

    @property
    def image_dir(self) -> Path:
        return self.path / "img1"

    @property
    def gt_file(self) -> Path:
        return self.path / "gt" / "gt.txt"


# -- MOT17 -----------------------------------------------------------------------------
def find_mot17_sequences(root: Path, split: str = "train", detector_suffix: str = "FRCNN"
                         ) -> list[MOTSequence]:
    """Expected: <root>/MOT17/<split>/MOT17-02-FRCNN/{img1,gt/gt.txt,seqinfo.ini}.

    The three detector variants (DPM/FRCNN/SDP) share identical images and ground truth, so
    one is enough. The public *test* split has no ground truth and cannot be scored locally.
    """
    if split not in {"train", "test"}:
        raise ConfigError("split must be 'train' or 'test'")
    base = root / "MOT17" / split
    if not base.is_dir():
        raise ConfigError(f"MOT17 split folder not found: {base}")
    sequences = []
    for seq_dir in sorted(p for p in base.iterdir() if p.is_dir() and p.name.endswith(detector_suffix)):
        info = seq_dir / "seqinfo.ini"
        if not info.is_file():
            continue
        cp = configparser.ConfigParser()
        cp.read(info)
        s = cp["Sequence"]
        sequences.append(MOTSequence(seq_dir.name, seq_dir, float(s["frameRate"]),
                                     int(s["imWidth"]), int(s["imHeight"]), int(s["seqLength"])))
    if not sequences:
        raise ConfigError(f"no MOT17 '*-{detector_suffix}' sequences with seqinfo.ini under {base}")
    return sequences


def parse_mot_gt(path: Path) -> dict[int, list[GTBox]]:
    """MOTChallenge gt.txt: frame,id,x,y,w,h,flag,class,visibility. Keeps scored pedestrians."""
    frames: dict[int, list[GTBox]] = defaultdict(list)
    for line in path.read_text().splitlines():
        parts = line.strip().split(",")
        if len(parts) < 8:
            continue
        frame, tid = int(parts[0]), int(parts[1])
        x, y, w, h = (float(v) for v in parts[2:6])
        flag, cls = int(float(parts[6])), int(float(parts[7]))
        if flag == 0 or cls != 1:  # ignored entries / non-pedestrian classes
            continue
        frames[frame].append(GTBox(x, y, w, h, "person", tid))
    return frames


# -- VisDrone --------------------------------------------------------------------------
def find_visdrone_det(root: Path, split: str = "val") -> tuple[Path, Path]:
    """Expected: <root>/VisDrone2019-DET-<split>/{images,annotations}."""
    base = root / f"VisDrone2019-DET-{split}"
    images, annotations = base / "images", base / "annotations"
    if not images.is_dir() or not annotations.is_dir():
        raise ConfigError(f"VisDrone-DET folders not found under {base}")
    return images, annotations


def parse_visdrone_annotations(path: Path) -> tuple[list[GTBox], list[GTBox]]:
    """Line: x,y,w,h,score,category,truncation,occlusion. Returns (scored, ignored regions)."""
    scored: list[GTBox] = []
    ignored: list[GTBox] = []
    for line in path.read_text().splitlines():
        parts = line.strip().split(",")
        if len(parts) < 6:
            continue
        x, y, w, h = (float(v) for v in parts[:4])
        category = int(parts[5])
        if category in VISDRONE_TO_COCO:
            scored.append(GTBox(x, y, w, h, VISDRONE_TO_COCO[category]))
        else:  # 0 = ignored region, 7/8 = tricycles, 11 = others: no COCO equivalent
            ignored.append(GTBox(x, y, w, h, "ignore"))
    return scored, ignored


# -- UCF-Crime -------------------------------------------------------------------------
def read_ucf_split(root: Path, split_file: str = "Anomaly_Test.txt") -> list[tuple[Path, int]]:
    """Return (video path, label) with label 1 = anomalous, 0 = normal.

    Expected: <root>/Anomaly_Detection_splits/<split_file> listing relative video paths
    (e.g. ``Abuse/Abuse028_x264.mp4``) and the videos beneath <root>. The official split
    keeps train and test videos disjoint; this function never mixes them.
    """
    split_path = root / "Anomaly_Detection_splits" / split_file
    if not split_path.is_file():
        raise ConfigError(f"UCF-Crime split file not found: {split_path}")
    items: list[tuple[Path, int]] = []
    for line in split_path.read_text().splitlines():
        rel = line.strip().replace("\\", "/")
        if not rel:
            continue
        video = (root / rel).resolve()
        if root.resolve() not in video.parents:
            continue  # ignore path escapes in the split file
        label = 0 if "normal" in rel.lower() else 1
        items.append((video, label))
    return items


def list_images(folder: Path) -> Iterator[Path]:
    return (p for p in sorted(folder.iterdir()) if p.suffix.lower() in IMAGE_EXTS)


def annotation_for(image: Path, annotations: Path) -> Optional[Path]:
    candidate = annotations / (image.stem + ".txt")
    return candidate if candidate.is_file() else None
