from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from f1p.lake import quality
from f1p.lake.curate_f1db import Curated


def test_all_gates_pass_on_real_subset(curated: Curated) -> None:
    results = quality.enforce(curated.tables)
    assert all(r.passed for r in results)
    assert {r.name for r in results} == {g.__name__ for g in quality.GATES}


def _with(curated: Curated, name: str, df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    return {**curated.tables, name: df}


def test_points_gate_catches_a_wrong_score(curated: Curated) -> None:
    r = curated.tables["results"]
    broken = r.with_columns(
        pl.when(
            (pl.col("event_id") == "2026-R17")
            & (pl.col("session") == "race")
            & (pl.col("position") == 1)
        )
        .then(pl.lit(26.0))
        .otherwise(pl.col("points"))
        .alias("points")
    )
    gate = quality.points_reconciliation(_with(curated, "results", broken))
    assert not gate.passed
    assert gate.failures.height == 1
    assert gate.failures["expected"].item() == 25.0


def test_classification_gate_catches_duplicate_positions(curated: Curated) -> None:
    r = curated.tables["results"]
    broken = r.with_columns(
        pl.when(
            (pl.col("event_id") == "2026-R17")
            & (pl.col("session") == "race")
            & (pl.col("position") == 2)
        )
        .then(pl.lit(1, dtype=pl.Int32))
        .otherwise(pl.col("position"))
        .alias("position")
    )
    gate = quality.classification_integrity(_with(curated, "results", broken))
    assert not gate.passed
    assert gate.failures["event_id"].to_list() == ["2026-R17"]


def test_status_gate_flags_unknown_reasons(curated: Curated) -> None:
    r = curated.tables["results"].with_columns(
        pl.when(pl.col("reason_retired").is_not_null())
        .then(pl.lit("Alien abduction"))
        .otherwise(pl.col("reason_retired"))
        .alias("reason_retired")
    )
    gate = quality.status_taxonomy(_with(curated, "results", r))
    assert not gate.passed
    assert gate.failures["reason_retired"].to_list() == ["Alien abduction"]


def test_coverage_gate_flags_missing_results(curated: Curated) -> None:
    r = curated.tables["results"].filter(pl.col("event_id") != "2026-R17")
    gate = quality.results_coverage(
        _with(curated, "results", r), now=datetime(2026, 10, 20, tzinfo=UTC)
    )
    assert not gate.passed
    assert gate.failures["event_id"].to_list() == ["2026-R17"]


def test_knowability_gate_flags_leaky_timestamps(curated: Curated) -> None:
    q = curated.tables["qualifying"].with_columns(
        pl.when(pl.col("event_id") == "2026-R17")
        .then(pl.col("observed_at") + pl.duration(days=2))
        .otherwise(pl.col("observed_at"))
        .alias("observed_at")
    )
    gate = quality.knowability_order(_with(curated, "qualifying", q))
    assert not gate.passed
    assert "2026-R17" in gate.failures["event_id"].to_list()


def test_enforce_raises_with_details(curated: Curated) -> None:
    r = curated.tables["results"].with_columns(pl.col("points") + 1)
    with pytest.raises(quality.QualityError, match="points_reconciliation"):
        quality.enforce(_with(curated, "results", r))
