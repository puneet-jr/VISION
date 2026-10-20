"""Runs the offline evaluation runners on tiny synthetic datasets with a scripted detector.

This verifies the plumbing (parsing, matching, output files). It says nothing about real model
performance on MOT17 / VisDrone / UCF-Crime.
"""
import cv2
import numpy as np
import pytest

from app.core.config import Settings
from app.evaluation import runners
from app.vision.types import Detection
from tests.conftest import FakeDetector


@pytest.fixture
def cfg(tmp_path):
    return Settings(_env_file=None, zones_file=tmp_path / "none.json")


def box_detector(x=100.0, y=100.0, w=40.0, h=80.0, cls="person", tid=7):
    return FakeDetector(lambda fid: [Detection(fid, 0.0, 0, cls, 0.9, (x, y, x + w, y + h), tid)])


def write_image(path, w=320, h=240):
    cv2.imwrite(str(path), np.zeros((h, w, 3), np.uint8))


def test_mot17_runner_scores_consistent_tracking(cfg, tmp_path):
    seq = tmp_path / "MOT17" / "train" / "MOT17-02-FRCNN"
    (seq / "img1").mkdir(parents=True)
    (seq / "gt").mkdir()
    (seq / "seqinfo.ini").write_text("[Sequence]\nname=MOT17-02-FRCNN\nframeRate=30\nseqLength=4\nimWidth=320\nimHeight=240\n")
    gt_lines = []
    for f in range(1, 5):
        write_image(seq / "img1" / f"{f:06d}.jpg")
        gt_lines.append(f"{f},1,100,100,40,80,1,1,1.0")
    (seq / "gt" / "gt.txt").write_text("\n".join(gt_lines))

    result = runners.evaluate_mot17(cfg, tmp_path, tmp_path / "out", detector=box_detector())
    assert result["sequences"] == ["MOT17-02-FRCNN"]
    m = result["metrics"]["MOT17-02-FRCNN"]
    assert m["num_switches"] == 0 and m["mota"] == pytest.approx(1.0) and m["idf1"] == pytest.approx(1.0)
    assert (tmp_path / "out" / "MOT17-02-FRCNN.txt").read_text().startswith("1,7,100.00,100.00,40.00,80.00")
    assert (tmp_path / "out" / "mot17_summary.txt").is_file()


def test_mot17_test_split_is_refused(cfg, tmp_path):
    from app.core.exceptions import ConfigError
    with pytest.raises(ConfigError):
        runners.evaluate_mot17(cfg, tmp_path, tmp_path / "out", split="test")


def test_visdrone_runner_counts_matches_and_ignored_regions(cfg, tmp_path):
    base = tmp_path / "VisDrone2019-DET-val"
    (base / "images").mkdir(parents=True)
    (base / "annotations").mkdir()
    write_image(base / "images" / "a.jpg")
    # one pedestrian GT that the detector finds, one car GT it misses
    (base / "annotations" / "a.txt").write_text("100,100,40,80,1,1,0,0\n10,10,30,20,1,4,0,0\n")
    result = runners.evaluate_visdrone_det(cfg, tmp_path, detector=box_detector())
    assert result["images_evaluated"] == 1
    assert result["per_class"]["person"]["tp"] == 1 and result["per_class"]["car"]["fn"] == 1
    assert result["overall"]["precision"] == 1.0 and result["overall"]["recall"] == 0.5


def test_ucf_runner_writes_csv_and_descriptive_auc(cfg, tmp_path):
    split_dir = tmp_path / "Anomaly_Detection_splits"
    split_dir.mkdir()
    (tmp_path / "Abuse").mkdir()
    (tmp_path / "Normal").mkdir()
    for rel in ("Abuse/a.avi", "Normal/n.avi"):
        writer = cv2.VideoWriter(str(tmp_path / rel), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 240))
        for _ in range(12):
            writer.write(np.zeros((240, 320, 3), np.uint8))
        writer.release()
    (split_dir / "Anomaly_Test.txt").write_text("Abuse/a.avi\nNormal/n.avi\n")

    # person sits inside the default zone only in the "abusive" video: detector keyed off call count
    inside, outside = (250.0, 120.0), (10.0, 10.0)
    state = {"calls": 0}

    def script(fid):
        state["calls"] += 1
        x, y = inside if state["calls"] <= 3 else outside
        return [Detection(fid, 0.0, 0, "person", 0.9, (x, y, x + 30, y + 60), 1)]

    result = runners.evaluate_ucf_crime(cfg, tmp_path, tmp_path / "out", frame_stride=4,
                                        detector=FakeDetector(script))
    assert result["videos_evaluated"] == 2
    rows = (tmp_path / "out" / "ucf_crime_videos.csv").read_text().splitlines()
    assert rows[0] == "video,label,max_score,events,high_events" and len(rows) == 3
    assert result["descriptive_auc_max_score"] == 1.0
    assert "not trained" in result["caveat"]
