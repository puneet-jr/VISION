import pytest

from app.core.exceptions import ConfigError
from app.evaluation import datasets
from app.evaluation.metrics import iou, match_boxes, roc_auc


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_match_boxes_counts_tp_fp_fn():
    gts = [(0, 0, 10, 10), (50, 50, 60, 60)]
    preds = [((0, 0, 10, 10), 0.9), ((200, 200, 210, 210), 0.8)]
    c = match_boxes(preds, gts)
    assert (c.tp, c.fp, c.fn) == (1, 1, 1)
    assert c.precision == 0.5 and c.recall == 0.5 and c.f1 == 0.5


def test_each_ground_truth_matches_once_and_ignore_regions_are_excused():
    gts = [(0, 0, 10, 10)]
    dup = [((0, 0, 10, 10), 0.9), ((0, 0, 10, 10), 0.8)]
    assert (match_boxes(dup, gts).tp, match_boxes(dup, gts).fp) == (1, 1)
    in_ignore = [((100, 100, 110, 110), 0.9)]
    assert match_boxes(in_ignore, [], ignore=[(90, 90, 120, 120)]).fp == 0


def test_roc_auc():
    assert roc_auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert roc_auc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0
    assert roc_auc([0.5, 0.5], [1, 0]) == 0.5
    assert roc_auc([0.5, 0.6], [1, 1]) is None


def test_parse_mot_gt_keeps_scored_pedestrians_only(tmp_path):
    gt = tmp_path / "gt.txt"
    gt.write_text("1,1,10,10,20,40,1,1,1.0\n1,2,0,0,5,5,0,1,1.0\n1,3,0,0,5,5,1,7,1.0\n2,1,12,10,20,40,1,1,0.9\n")
    frames = datasets.parse_mot_gt(gt)
    assert [b.track_id for b in frames[1]] == [1] and frames[2][0].x == 12


def test_parse_visdrone_maps_classes_and_ignored_regions(tmp_path):
    ann = tmp_path / "a.txt"
    ann.write_text("10,10,20,40,1,1,0,0\n0,0,100,100,0,0,0,0\n30,30,10,10,1,4,0,0\n5,5,5,5,1,11,0,0\n")
    scored, ignored = datasets.parse_visdrone_annotations(ann)
    assert [b.label for b in scored] == ["person", "car"] and len(ignored) == 2


def test_ucf_split_labels_and_path_escape(tmp_path):
    split_dir = tmp_path / "Anomaly_Detection_splits"
    split_dir.mkdir()
    (split_dir / "Anomaly_Test.txt").write_text("Abuse/Abuse001.mp4\nTesting_Normal_Videos_Anomaly/Normal_1.mp4\n../../etc/passwd\n")
    items = datasets.read_ucf_split(tmp_path)
    assert [label for _, label in items] == [1, 0]
    with pytest.raises(ConfigError):
        datasets.read_ucf_split(tmp_path, "missing.txt")


def test_missing_mot17_folder_is_a_clear_error(tmp_path):
    with pytest.raises(ConfigError):
        datasets.find_mot17_sequences(tmp_path)
