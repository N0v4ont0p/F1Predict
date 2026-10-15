from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from f1p.core.clock import AsOf, known_at

T0 = datetime(2026, 5, 24, 20, 0, tzinfo=UTC)


def test_as_of_requires_timezone() -> None:
    with pytest.raises(ValueError, match="timezone"):
        AsOf(datetime(2026, 5, 24))


def test_as_of_normalises_to_utc() -> None:
    plus_two = timezone(timedelta(hours=2))
    assert AsOf(datetime(2026, 5, 24, 22, 0, tzinfo=plus_two)).at == T0


def test_parse_and_str_round_trip() -> None:
    a = AsOf.parse("2026-05-24T20:00:00Z")
    assert a.at == T0
    assert str(a) == "2026-05-24T20:00:00Z"
    assert AsOf.parse(str(a)) == a


def _frame(times: list[datetime | None], values: list[int]) -> pl.DataFrame:
    return pl.DataFrame(
        {"observed_at": times, "value": values},
        schema={"observed_at": pl.Datetime("us", "UTC"), "value": pl.Int64},
    )


def test_known_at_keeps_only_published_rows() -> None:
    df = _frame([T0 - timedelta(days=1), T0, T0 + timedelta(seconds=1), None], [1, 2, 3, 4])
    out = known_at(df, AsOf(T0))
    assert out["value"].to_list() == [1, 2]


def test_known_at_works_lazily() -> None:
    lf = _frame([T0, T0 + timedelta(hours=1)], [1, 2]).lazy()
    assert known_at(lf, AsOf(T0)).collect()["value"].to_list() == [1]


offsets = st.integers(min_value=-10_000, max_value=10_000)
int64s = st.integers(min_value=-(2**63), max_value=2**63 - 1)


@settings(max_examples=60)
@given(
    st.lists(st.tuples(offsets, int64s), min_size=1, max_size=40),
    st.lists(int64s, min_size=40, max_size=40),
)
def test_future_perturbation_invariance(rows: list[tuple[int, int]], noise: list[int]) -> None:
    """Rewriting anything published after ``as_of`` must not change what is known at ``as_of``."""
    times = [T0 + timedelta(minutes=m) for m, _ in rows]
    values = [v for _, v in rows]
    perturbed = [n if t > T0 else v for t, v, n in zip(times, values, noise, strict=False)]
    before = known_at(_frame(list(times), values), AsOf(T0))
    after = known_at(_frame(list(times), perturbed), AsOf(T0))
    assert before.equals(after)
