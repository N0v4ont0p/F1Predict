"""Property-based & unit tests for the points / championship math."""
from __future__ import annotations

from hypothesis import given, strategies as st

from f1predict.utils.points import (
    SPRINT_POINTS, points_for_position, system_for_season,
)


def test_winner_scores_most_modern():
    pts = [points_for_position(p, 2025) for p in range(1, 21)]
    assert pts[0] == 25
    assert pts == sorted(pts, reverse=True)  # monotonic non-increasing


@given(pos=st.integers(min_value=1, max_value=30), season=st.integers(min_value=1950, max_value=2025))
def test_points_non_negative_and_bounded(pos, season):
    p = points_for_position(pos, season)
    assert 0 <= p <= 26  # 25 + fastest lap


@given(season=st.integers(min_value=1950, max_value=2025))
def test_points_monotonic_in_position(season):
    prev = float("inf")
    for pos in range(1, 12):
        cur = points_for_position(pos, season, fastest_lap=False)
        assert cur <= prev
        prev = cur


def test_no_points_outside_table():
    assert points_for_position(50, 2025) == 0.0
    assert points_for_position(0, 2025) == 0.0
    assert points_for_position(-3, 2025) == 0.0


def test_fastest_lap_only_top10():
    assert points_for_position(1, 2025, fastest_lap=True) == 26
    assert points_for_position(11, 2025, fastest_lap=True) == 0.0  # outside top 10 -> no bonus


def test_sprint_table():
    assert points_for_position(1, 2025, session="S") == SPRINT_POINTS[1]
    assert points_for_position(9, 2025, session="S") == 0.0


def test_era_selection():
    assert system_for_season(2025) == "modern"
    assert system_for_season(1955) == "points_1950"
    assert system_for_season(1995) == "points_1991"
