import json

import pytest

from app.core.exceptions import ConfigError
from app.vision.zone_analyzer import (
    DEFAULT_ZONES_DATA, Zone, ZoneAnalyzer, ZonePresenceTracker, ZoneStatus, distance_to_polygon,
    load_zones, parse_zones, point_in_polygon,
)

SQUARE = ((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0))


def analyzer(near=20.0) -> ZoneAnalyzer:
    return ZoneAnalyzer([Zone("z1", "Zone 1", SQUARE), Zone("z2", "Monitored", ((300, 0), (400, 0), (400, 100)), False)], near)


def test_point_in_polygon():
    assert point_in_polygon((50, 50), SQUARE)
    assert point_in_polygon((0, 50), SQUARE)  # boundary counts as inside
    assert not point_in_polygon((150, 50), SQUARE)
    triangle = ((0, 0), (10, 0), (0, 10))
    assert point_in_polygon((2, 2), triangle) and not point_in_polygon((8, 8), triangle)


def test_distance_to_polygon():
    assert distance_to_polygon((50, 50), SQUARE) == 0.0
    assert distance_to_polygon((110, 50), SQUARE) == pytest.approx(10.0)
    assert distance_to_polygon((103, 104), SQUARE) == pytest.approx(5.0)  # nearest is a corner


@pytest.mark.parametrize("point,status", [
    ((50, 50), ZoneStatus.INSIDE), ((110, 50), ZoneStatus.NEAR), ((200, 50), ZoneStatus.OUTSIDE),
])
def test_analyze_statuses(point, status):
    assert analyzer().analyze(point)[0].status is status


def test_nearest_restricted_ignores_monitored_only_zones():
    contexts = analyzer().analyze((310, 10))  # inside the non-restricted triangle
    nearest = ZoneAnalyzer.nearest_restricted(contexts)
    assert nearest is not None and nearest.zone_id == "z1"
    assert ZoneAnalyzer.nearest_restricted([c for c in contexts if not c.restricted]) is None


def test_scaling_rescales_polygons_and_near_distance():
    scaled = analyzer().scaled(2.0, 0.5, near_distance_px=7.0)
    assert scaled.zones[0].polygon[2] == (200.0, 50.0)
    assert scaled.near_distance_px == 7.0


def test_zone_requires_three_points():
    with pytest.raises(ConfigError):
        Zone("bad", "bad", ((0, 0), (1, 1)))


def test_parse_and_load_zones(tmp_path):
    zones, ref = parse_zones(DEFAULT_ZONES_DATA)
    assert ref == (1280.0, 720.0) and zones[0].restricted
    path = tmp_path / "zones.json"
    path.write_text(json.dumps({"zones": [{"id": "a", "polygon": [[0, 0], [5, 0], [5, 5]]}]}))
    loaded, _ = load_zones(path)
    assert loaded[0].name == "a"
    assert load_zones(tmp_path / "absent.json")[0][0].zone_id == "zone-1"  # built-in fallback


def test_invalid_zone_definitions_rejected():
    with pytest.raises(ConfigError):
        parse_zones({"zones": [{"id": "a"}]})
    dup = {"id": "a", "polygon": [[0, 0], [1, 0], [1, 1]]}
    with pytest.raises(ConfigError):
        parse_zones({"zones": [dup, dup]})


class TestPresence:
    def test_entry_is_reported_once(self):
        p = ZonePresenceTracker(exit_grace_s=2.0)
        assert p.update(1, 0.0, []) == []
        assert p.update(1, 1.0, ["z1"]) == ["z1"]
        assert p.update(1, 2.0, ["z1"]) == []

    def test_dwell_time_is_measured_from_entry(self):
        p = ZonePresenceTracker(2.0)
        p.update(1, 10.0, ["z1"])
        p.update(1, 14.5, ["z1"])
        assert p.dwell_s(1, "z1", 14.5) == pytest.approx(4.5)
        assert p.max_dwell_s(1, 14.5) == pytest.approx(4.5)
        assert p.max_dwell_s(99, 14.5) == 0.0

    def test_brief_absence_keeps_dwell_and_does_not_reenter(self):
        p = ZonePresenceTracker(2.0)
        p.update(1, 0.0, ["z1"])
        p.update(1, 1.0, [])  # border jitter
        assert p.update(1, 1.5, ["z1"]) == []
        assert p.dwell_s(1, "z1", 1.5) == pytest.approx(1.5)

    def test_long_absence_resets_and_reenters(self):
        p = ZonePresenceTracker(2.0)
        p.update(1, 0.0, ["z1"])
        p.update(1, 5.0, [])
        assert p.dwell_s(1, "z1", 5.0) == 0.0
        assert p.update(1, 6.0, ["z1"]) == ["z1"]

    def test_tracks_are_independent_and_prunable(self):
        p = ZonePresenceTracker(2.0)
        p.update(1, 0.0, ["z1"])
        p.update(2, 0.0, ["z1"])
        p.update(2, 9.0, ["z1"])
        p.prune(9.0, stale_after_s=5.0)
        assert p.dwell_s(1, "z1", 9.0) == 0.0
        assert p.dwell_s(2, "z1", 9.0) == pytest.approx(9.0)
