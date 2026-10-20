import pytest

from app.vision.motion_analyzer import MotionAnalyzer, direction_label


def test_single_observation_is_unknown_not_zero_motion():
    m = MotionAnalyzer()
    s = m.update(1, 0.0, (10, 10))
    assert s.observations == 1 and s.direction == "unknown" and s.approaching is None
    assert m.state(404).observations == 0


def test_speed_and_direction_from_displacement():
    m = MotionAnalyzer()
    m.update(1, 0.0, (0, 0))
    s = m.update(1, 2.0, (100, 0))
    assert s.speed_px_s == pytest.approx(50.0)
    assert s.direction == "right" and s.direction_deg == pytest.approx(0.0)
    up = MotionAnalyzer()
    up.update(1, 0.0, (50, 100))
    assert up.update(1, 1.0, (50, 40)).direction == "up"  # image y decreases => moving up


def test_repeated_positions_are_stationary():
    m = MotionAnalyzer(min_motion_px=3.0)
    for t in range(5):
        s = m.update(1, float(t), (30, 30))
    assert s.speed_px_s == 0.0 and s.direction == "stationary" and s.displacement_px == 0.0


def test_repeated_or_out_of_order_timestamps_never_divide_by_zero():
    m = MotionAnalyzer()
    m.update(1, 1.0, (0, 0))
    s = m.update(1, 1.0, (50, 0))  # same timestamp: replaces the sample
    assert s.observations == 1
    s = m.update(1, 0.5, (90, 0))  # older timestamp: ignored
    assert s.observations == 1 and s.speed_px_s == 0.0


def test_approach_needs_enough_observations():
    m = MotionAnalyzer(min_observations=5)
    for i in range(4):
        s = m.update(1, float(i), (i * 10, 0), zone_distance=200.0 - i * 20)
    assert s.approach == "unknown" and s.approaching is None
    s = m.update(1, 4.0, (40, 0), zone_distance=120.0)
    assert s.approach == "approaching" and s.approaching is True


def test_receding_and_steady():
    receding = MotionAnalyzer(min_observations=3)
    for i in range(4):
        s = receding.update(1, float(i), (0, 0), zone_distance=100.0 + 30 * i)
    assert s.approach == "receding" and s.approaching is False
    steady = MotionAnalyzer(min_observations=3)
    for i in range(4):
        s = steady.update(1, float(i), (0, 0), zone_distance=100.0)
    assert s.approach == "steady" and s.approaching is False


def test_missing_zone_distance_keeps_approach_unknown():
    m = MotionAnalyzer(min_observations=3)
    for i in range(6):
        s = m.update(1, float(i), (i * 5, 0), zone_distance=None)
    assert s.approach == "unknown"


def test_history_is_bounded():
    m = MotionAnalyzer(history_len=10)
    for i in range(100):
        s = m.update(1, float(i), (i, 0))
    assert len(s.trajectory) == 10 and s.observations == 10


def test_prune_removes_stale_tracks():
    m = MotionAnalyzer()
    m.update(1, 0.0, (0, 0))
    m.update(2, 9.0, (0, 0))
    m.prune(10.0, stale_after_s=5.0)
    assert len(m) == 1 and m.state(1).observations == 0


@pytest.mark.parametrize("angle,label", [(0, "right"), (90, "up"), (180, "left"), (270, "down"),
                                         (45, "up-right"), (359, "right")])
def test_direction_label(angle, label):
    assert direction_label(angle) == label
