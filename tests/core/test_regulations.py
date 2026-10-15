from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from f1p.core import regulations as reg


@pytest.mark.parametrize(
    ("season", "start", "since"),
    [
        (1950, 1950, 0),
        (1960, 1950, 10),
        (2014, 2014, 0),
        (2021, 2017, 4),
        (2026, 2026, 0),
        (2027, 2026, 1),
    ],
)
def test_era_start_and_age(season: int, start: int, since: int) -> None:
    assert reg.era_start(season) == start
    assert reg.seasons_since_reset(season) == since


def test_era_bounds() -> None:
    assert reg.era(2021) == reg.Era(first=2017, last=2021)
    assert reg.era(2030) == reg.Era(first=2026, last=None)
    assert 2024 in reg.era(2022)
    assert 2026 not in reg.era(2022)
    assert reg.era(2022).name == "era_2022"


def test_reset_seasons() -> None:
    assert reg.is_reset_season(2026)
    assert not reg.is_reset_season(2025)


def test_pre_championship_rejected() -> None:
    with pytest.raises(ValueError, match="1950"):
        reg.era_start(1949)


@pytest.mark.parametrize(
    ("season", "field", "q1", "q2"),
    [(2017, 20, 5, 5), (2026, 22, 6, 6), (2010, 24, 7, 7), (2016, 21, 6, 5)],
)
def test_knockout_formats(season: int, field: int, q1: int, q2: int) -> None:
    fmt = reg.knockout_format(season, field)
    assert (fmt.q1_eliminated, fmt.q2_eliminated) == (q1, q2)
    assert fmt.q3_size == reg.Q3_SIZE


def test_knockout_rejects_old_seasons_and_tiny_fields() -> None:
    with pytest.raises(NotImplementedError):
        reg.knockout_format(2005, 20)
    with pytest.raises(ValueError, match="too small"):
        reg.knockout_format(2026, 10)


@given(st.integers(min_value=2006, max_value=2040), st.integers(min_value=11, max_value=40))
def test_knockout_always_leaves_ten_for_q3(season: int, field: int) -> None:
    fmt = reg.knockout_format(season, field)
    assert fmt.q3_size == reg.Q3_SIZE
    assert 0 <= fmt.q1_eliminated - fmt.q2_eliminated <= 1


@given(st.integers(min_value=1950, max_value=2100))
def test_every_season_belongs_to_its_era(season: int) -> None:
    assert season in reg.era(season)
    assert reg.seasons_since_reset(season) >= 0
