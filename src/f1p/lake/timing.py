"""When did each fact become public? (``observed_at``, plan §6.2)

Exact session start times (UTC) exist in F1DB from 2024. For earlier seasons only dates are
known, so we fall back to **conservative, causally ordered** day-level rules:

* practice results  → known from 00:00 UTC on the day *before* race day;
* qualifying / grid → known from 00:00 UTC on race day (a pre-race checkpoint can see them);
* sprint results    → known from 00:00 UTC on race day;
* race results      → known from 12:00 UTC on the day *after* race day (no race, not even
  a late-night street race, ends later than that).

These rules guarantee qualifying is known before the race and race results before the next
event (which is always ≥ 6 days later). Exact times, where present, give finer resolution.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta

PRACTICE_LENGTH = timedelta(minutes=75)
QUALIFYING_LENGTH = timedelta(minutes=90)
SPRINT_QUALIFYING_LENGTH = timedelta(minutes=60)
RESULT_PUBLICATION_LAG = timedelta(minutes=30)
FALLBACK_RACE_LENGTH = timedelta(hours=2, minutes=30)
FALLBACK_SPRINT_LENGTH = timedelta(minutes=45)
PRE_RACE_MARGIN = timedelta(minutes=15)


def utc_datetime(day: date | None, clock: str | None) -> datetime | None:
    """Combine an F1DB date and an ``HH:MM`` UTC time; ``None`` if either is missing."""
    if day is None or not clock:
        return None
    hours, minutes = (int(x) for x in clock.split(":")[:2])
    return datetime.combine(day, time(hours, minutes), tzinfo=UTC)


def _midnight(day: date) -> datetime:
    return datetime.combine(day, time(0, 0), tzinfo=UTC)


def race_results_observed(
    race_date: date, start: datetime | None, winner_ms: int | None
) -> datetime:
    if start is not None:
        length = timedelta(milliseconds=winner_ms) if winner_ms else FALLBACK_RACE_LENGTH
        return start + length + RESULT_PUBLICATION_LAG
    return _midnight(race_date) + timedelta(days=1, hours=12)


def sprint_results_observed(
    race_date: date, start: datetime | None, winner_ms: int | None
) -> datetime:
    if start is not None:
        length = timedelta(milliseconds=winner_ms) if winner_ms else FALLBACK_SPRINT_LENGTH
        return start + length + RESULT_PUBLICATION_LAG
    return _midnight(race_date)


def qualifying_observed(
    race_date: date, start: datetime | None, *, sprint: bool = False
) -> datetime:
    if start is not None:
        length = SPRINT_QUALIFYING_LENGTH if sprint else QUALIFYING_LENGTH
        return start + length + RESULT_PUBLICATION_LAG
    return _midnight(race_date)


def grid_observed(race_date: date, race_start: datetime | None) -> datetime:
    if race_start is not None:
        return race_start - PRE_RACE_MARGIN
    return _midnight(race_date)


def practice_observed(race_date: date, start: datetime | None) -> datetime:
    if start is not None:
        return start + PRACTICE_LENGTH + RESULT_PUBLICATION_LAG
    return _midnight(race_date) - timedelta(days=1)


def pre_race_as_of(race_date: date, race_start: datetime | None) -> datetime:
    """The C4 checkpoint: after qualifying, before the race starts."""
    return grid_observed(race_date, race_start)


def pre_weekend_as_of(race_date: date) -> datetime:
    """The C1 checkpoint: Thursday 00:00 UTC of race week (three days before race day)."""
    return _midnight(race_date) - timedelta(days=3)
