from __future__ import annotations

import itertools

import pytest
from hypothesis import given
from hypothesis import strategies as st

from f1p.core import points
from f1p.core.points import FULL_RACE, RaceConditions, Session

SEASONS = st.integers(min_value=points.EARLIEST_SUPPORTED_SEASON, max_value=2030)
FRACTIONS = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)


# --------------------------------------------------------------------------- known tables
@pytest.mark.parametrize(
    ("season", "expected"),
    [
        (1991, (10, 6, 4, 3, 2, 1)),
        (2002, (10, 6, 4, 3, 2, 1)),
        (2003, (10, 8, 6, 5, 4, 3, 2, 1)),
        (2009, (10, 8, 6, 5, 4, 3, 2, 1)),
        (2010, (25, 18, 15, 12, 10, 8, 6, 4, 2, 1)),
        (2026, (25, 18, 15, 12, 10, 8, 6, 4, 2, 1)),
    ],
)
def test_race_scales(season: int, expected: tuple[int, ...]) -> None:
    assert points.base_scale(season) == expected


@pytest.mark.parametrize(
    ("season", "expected"),
    [(2021, (3, 2, 1)), (2022, (8, 7, 6, 5, 4, 3, 2, 1)), (2026, (8, 7, 6, 5, 4, 3, 2, 1))],
)
def test_sprint_scales(season: int, expected: tuple[int, ...]) -> None:
    assert points.base_scale(season, Session.SPRINT) == expected


def test_no_sprints_before_2021() -> None:
    with pytest.raises(ValueError, match="no sprint"):
        points.base_scale(2020, Session.SPRINT)


def test_pre_1991_not_derived_here() -> None:
    with pytest.raises(NotImplementedError):
        points.base_scale(1990)


def test_points_beyond_scoring_places_are_zero() -> None:
    assert points.points_for(11, 2026) == 0.0
    assert points.points_for(22, 2026) == 0.0


def test_unclassified_and_missing_positions_score_nothing() -> None:
    assert points.points_for(1, 2026, classified=False) == 0.0
    assert points.points_for(None, 2026) == 0.0


def test_invalid_position_rejected() -> None:
    with pytest.raises(ValueError, match="position"):
        points.points_for(0, 2026)


# --------------------------------------------------------------------------- fastest lap
@pytest.mark.parametrize(
    ("season", "position", "expected"),
    [
        (2019, 1, 26.0),
        (2024, 10, 2.0),
        (2024, 11, 0.0),  # outside the top ten: no bonus
        (2018, 1, 25.0),  # before the bonus existed
        (2025, 1, 25.0),  # scrapped for 2025
    ],
)
def test_fastest_lap_bonus(season: int, position: int, expected: float) -> None:
    assert points.points_for(position, season, fastest_lap=True) == expected


def test_no_fastest_lap_bonus_in_sprints() -> None:
    assert points.points_for(1, 2022, Session.SPRINT, fastest_lap=True) == 8.0


# --------------------------------------------------------------------------- special races
def test_2014_abu_dhabi_double_points() -> None:
    mult = points.event_multiplier(2014, "yas_marina")
    assert mult == 2.0
    conditions = RaceConditions(multiplier=mult)
    assert points.points_for(1, 2014, conditions=conditions) == 50.0
    assert points.points_for(10, 2014, conditions=conditions) == 2.0


def test_ordinary_events_have_no_multiplier() -> None:
    assert points.event_multiplier(2014, "monza") == 1.0
    assert points.event_multiplier(2015, "yas_marina") == 1.0


def test_2021_belgian_gp_half_points_and_no_fastest_lap() -> None:
    spa = RaceConditions(distance_fraction=1 / 44)
    assert points.scale(2021, conditions=spa) == (12.5, 9, 7.5, 6, 5, 4, 3, 2, 1, 0.5)
    assert points.points_for(1, 2021, conditions=spa, fastest_lap=True) == 12.5


def test_2009_malaysian_gp_half_points() -> None:
    sepang = RaceConditions(distance_fraction=31 / 56)
    assert points.scale(2009, conditions=sepang) == (5, 4, 3, 2.5, 2, 1.5, 1, 0.5)


@pytest.mark.parametrize(
    ("fraction", "expected"),
    [
        (0.10, (6, 4, 3, 2, 1)),
        (0.2499, (6, 4, 3, 2, 1)),
        (0.25, (13, 10, 8, 6, 5, 4, 3, 2, 1)),
        (0.49, (13, 10, 8, 6, 5, 4, 3, 2, 1)),
        (0.50, (19, 14, 12, 9, 8, 6, 5, 3, 2, 1)),
        (0.7499, (19, 14, 12, 9, 8, 6, 5, 3, 2, 1)),
        (0.75, (25, 18, 15, 12, 10, 8, 6, 4, 2, 1)),
    ],
)
def test_2022_plus_shortened_race_scales(fraction: float, expected: tuple[int, ...]) -> None:
    assert points.scale(2023, conditions=RaceConditions(distance_fraction=fraction)) == expected


def test_2022_plus_needs_two_green_flag_laps() -> None:
    assert points.scale(2024, conditions=RaceConditions(0.05, green_flag_laps=1)) == ()
    assert points.scale(2024, conditions=RaceConditions(0.05, green_flag_laps=2)) == (
        6,
        4,
        3,
        2,
        1,
    )


def test_shortened_sprints_are_not_guessed() -> None:
    with pytest.raises(NotImplementedError):
        points.scale(2026, Session.SPRINT, RaceConditions(distance_fraction=0.4))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"distance_fraction": -0.1},
        {"distance_fraction": 1.1},
        {"green_flag_laps": -1},
        {"multiplier": -1},
    ],
)
def test_conditions_validated(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):  # noqa: PT011 - each case has its own message
        RaceConditions(**kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- properties
@given(SEASONS, FRACTIONS)
def test_scales_are_non_increasing_and_non_negative(season: int, fraction: float) -> None:
    table = points.scale(season, conditions=RaceConditions(distance_fraction=fraction))
    assert all(p >= 0 for p in table)
    assert all(a >= b for a, b in itertools.pairwise(table))


@given(SEASONS, FRACTIONS)
def test_shortened_races_never_pay_more_than_full_races(season: int, fraction: float) -> None:
    full = points.scale(season)
    short = points.scale(season, conditions=RaceConditions(distance_fraction=fraction))
    assert len(short) <= len(full)
    assert all(s <= f for s, f in zip(short, full, strict=False))


@given(SEASONS, st.integers(min_value=10, max_value=26), st.data())
def test_points_distributed_over_a_field_match_the_scale(
    season: int, field: int, data: st.DataObject
) -> None:
    fastest = data.draw(st.integers(min_value=1, max_value=field))
    total = sum(
        points.points_for(p, season, fastest_lap=(p == fastest)) for p in range(1, field + 1)
    )
    bonus = 1.0 if points.fastest_lap_bonus_applies(season) and fastest <= 10 else 0.0
    assert total == pytest.approx(sum(points.scale(season)) + bonus)


@given(SEASONS, st.integers(min_value=1, max_value=30))
def test_points_table_is_dense_and_consistent(season: int, n: int) -> None:
    table = points.points_table(season, n)
    assert len(table) == n
    assert table == tuple(points.points_for(p, season) for p in range(1, n + 1))


def test_full_race_constant_is_full_distance() -> None:
    assert FULL_RACE.is_full_distance
