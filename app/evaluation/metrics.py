"""Dependency-free evaluation helpers (IoU matching, precision/recall, ROC-AUC)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

Box = tuple[float, float, float, float]  # x1, y1, x2, y2


def iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def iou_distance_matrix(gts: Sequence[Box], hyps: Sequence[Box], max_distance: float = 0.5
                        ) -> list[list[float]]:
    """1 - IoU for every (ground truth, hypothesis) pair; pairs beyond ``max_distance`` are NaN
    (MOTChallenge convention: IoU < 0.5 cannot be a match). Built here instead of with
    ``motmetrics.distances.iou_matrix``, which breaks on NumPy 2."""
    nan = float("nan")
    matrix = []
    for g in gts:
        row = []
        for h in hyps:
            d = 1.0 - iou(g, h)
            row.append(d if d <= max_distance else nan)
        matrix.append(row)
    return matrix


@dataclass
class MatchCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other: "MatchCounts") -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def to_dict(self) -> dict:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn, "precision": round(self.precision, 4),
                "recall": round(self.recall, 4), "f1": round(self.f1, 4)}


def match_boxes(preds: Sequence[tuple[Box, float]], gts: Sequence[Box],
                ignore: Sequence[Box] = (), iou_thr: float = 0.5) -> MatchCounts:
    """Greedy matching by descending confidence. Predictions that fall mostly inside an
    ignore region are neither true nor false positives."""
    order = sorted(range(len(preds)), key=lambda i: -preds[i][1])
    taken: set[int] = set()
    counts = MatchCounts()
    for i in order:
        box = preds[i][0]
        best, best_iou = -1, iou_thr
        for j, g in enumerate(gts):
            if j in taken:
                continue
            v = iou(box, g)
            if v >= best_iou:
                best, best_iou = j, v
        if best >= 0:
            taken.add(best)
            counts.tp += 1
        elif any(_overlap_fraction(box, r) > 0.5 for r in ignore):
            continue
        else:
            counts.fp += 1
    counts.fn = len(gts) - len(taken)
    return counts


def _overlap_fraction(box: Box, region: Box) -> float:
    ix1, iy1 = max(box[0], region[0]), max(box[1], region[1])
    ix2, iy2 = min(box[2], region[2]), min(box[3], region[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = (box[2] - box[0]) * (box[3] - box[1])
    return inter / area if area > 0 else 0.0


def roc_auc(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    """Rank-based AUC (ties get average rank). None unless both classes are present."""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos) * len(neg))
