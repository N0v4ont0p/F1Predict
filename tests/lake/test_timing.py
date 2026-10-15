from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from hypothesis import given
from hypothesis import strategies as st

from f1p.lake import timing

DAY = date(2026, 10, 11)
START = datetime(2026, 10, 11, 12, 0, tzinfo=UTC)


def test_utc_datetime() -> None:
    assert timing.utc_datetime(DAY, "12:00") == START
    assert timing.utc_datetime(DAY, None) is None
    assert timing.utc_datetime(None, "12:00") is None


def test_exact_times_use_session_lengths() -> None:
    assert (
        timing.race_results_observed(DAY, START, 6_000_000)
        == START + timedelta(milliseconds=6_000_000) + timing.RESULT_PUBLICATION_LAG
    )
    assert (
        timing.qualifying_observed(DAY, START - timedelta(days=1))
        == START - timedelta(days=1) + timing.QUALIFYING_LENGTH + timing.RESULT_PUBLICATION_LAG
    )


@given(st.dates(min_value=date(1950, 1, 1), max_value=date(2100, 1, 1)))
def test_fallback_rules_are_causally_ordered(day: date) -> None:
    practice = timing.practice_observed(day, None)
    quali = timing.qualifying_observed(day, None)
    sprint = timing.sprint_results_observed(day, None, None)
    c4 = timing.pre_race_as_of(day, None)
    race = timing.race_results_observed(day, None, None)
    c1 = timing.pre_weekend_as_of(day)
    next_c1 = timing.pre_weekend_as_of(day + timedelta(days=7))
    assert c1 < practice <= quali <= c4 < race < next_c1
    assert sprint <= c4


def test_exact_times_keep_quali_before_race_start() -> None:
    quali_start = START - timedelta(days=1, hours=1)
    assert timing.qualifying_observed(DAY, quali_start) < timing.pre_race_as_of(DAY, START)
