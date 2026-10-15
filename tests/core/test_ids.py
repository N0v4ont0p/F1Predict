from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from f1p.core.ids import EventKey, SessionKey, SessionType, slugify


@pytest.mark.parametrize(
    ("name", "slug"),
    [
        ("Max Verstappen", "max_verstappen"),
        ("Kimi Räikkönen", "kimi_raikkonen"),
        ("Red Bull", "red_bull"),
        ("Yas Marina", "yas_marina"),
        ("  Sergio  Pérez ", "sergio_perez"),
    ],
)
def test_slugify(name: str, slug: str) -> None:
    assert slugify(name) == slug


def test_slugify_rejects_empty() -> None:
    with pytest.raises(ValueError, match="cannot build"):
        slugify("!!!")


def test_event_key_format_and_parse() -> None:
    key = EventKey(2026, 5)
    assert str(key) == "2026-R05"
    assert EventKey.parse("2026-R05") == key


@pytest.mark.parametrize("bad", ["2026-5", "R05", "2026-R5x", ""])
def test_event_key_rejects_garbage(bad: str) -> None:
    with pytest.raises(ValueError, match="event key"):
        EventKey.parse(bad)


@pytest.mark.parametrize(("season", "rnd"), [(1949, 1), (2026, 0), (2026, 100)])
def test_event_key_validates_ranges(season: int, rnd: int) -> None:
    with pytest.raises(ValueError):  # noqa: PT011 - two distinct messages
        EventKey(season, rnd)


@given(st.integers(min_value=1950, max_value=2100), st.integers(min_value=1, max_value=99))
def test_event_key_round_trip(season: int, rnd: int) -> None:
    key = EventKey(season, rnd)
    assert EventKey.parse(str(key)) == key


@given(st.sampled_from(list(SessionType)))
def test_session_key_round_trip(session: SessionType) -> None:
    key = SessionKey(EventKey(2026, 22), session)
    assert SessionKey.parse(str(key)) == key


def test_session_type_flags() -> None:
    assert SessionType.FP2.is_practice
    assert not SessionType.QUALIFYING.is_practice
    assert SessionType.SPRINT.scores_points
    assert not SessionType.QUALIFYING.scores_points
