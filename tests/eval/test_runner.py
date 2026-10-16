from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from f1p.eval import harness, runner
from f1p.eval.baselines import Uniform
from f1p.lake.curate_f1db import Curated
from f1p.models.world_v1 import WorldV1


def _run(curated: Curated, **kw: object) -> harness.EvalResult:
    return harness.run(
        curated.tables,
        [WorldV1(), Uniform()],
        seasons=(2026, 2026),
        history_from=2021,
        keep_forecasts=True,
        **kw,  # type: ignore[arg-type]
    )


@pytest.fixture(scope="module")
def serial(curated: Curated) -> harness.EvalResult:
    return _run(curated)


def test_parallel_matches_serial(serial: harness.EvalResult, curated: Curated) -> None:
    par = _run(curated, workers=3)
    assert par.per_event.equals(serial.per_event)
    assert serial.forecasts.keys() == par.forecasts.keys()
    for k, fc in serial.forecasts.items():
        np.testing.assert_array_equal(fc.position_probs, par.forecasts[k].position_probs)
    for k, (p, y) in serial.calibration.items():
        p2, y2 = par.calibration[k]
        np.testing.assert_allclose(np.sort(p), np.sort(p2))
        assert y.sum() == y2.sum()


def test_cache_round_trip(serial: harness.EvalResult, curated: Curated, tmp_path: Path) -> None:
    first = _run(curated, cache_dir=tmp_path)
    files = sorted(tmp_path.glob("*.pkl"))
    assert len(files) == 2  # one per model
    second = _run(curated, cache_dir=tmp_path)
    assert second.per_event.equals(serial.per_event)
    assert first.per_event.equals(serial.per_event)
    assert sorted(tmp_path.glob("*.pkl")) == files


def test_prepare_signature_ignores_combiner_settings() -> None:
    a, b = WorldV1(), WorldV1(l2=5.0, training_seasons=4)
    assert runner.prepare_signature(a) == runner.prepare_signature(b)
    assert runner.prepare_signature(a) != runner.prepare_signature(WorldV1(pace_decay=0.5))


def test_table_fingerprint_tracks_content(curated: Curated) -> None:
    import polars as pl

    t = curated.tables
    fp = runner.table_fingerprint(t)
    changed = {**t, "results": t["results"].with_columns(pl.col("points") * 2)}
    assert runner.table_fingerprint(changed) != fp
    assert runner.table_fingerprint(dict(t)) == fp


def test_chunks_are_contiguous_and_complete() -> None:
    items = list(range(10))
    parts = runner._chunks(items, 3)
    assert [len(p) for p in parts] == [4, 3, 3]
    assert [x for p in parts for x in p] == items


def test_signatures_are_stable_across_use_and_processes(curated: Curated) -> None:
    from f1p.cli.eval import available_models

    models = available_models("race")
    before = {
        n: (runner.prepare_signature(m), runner.model_signature(m)) for n, m in models.items()
    }
    assert not any("0x" in runner._settings_repr(m) for m in models.values())
    harness.run(curated.tables, [models["world_v1"]], seasons=(2026, 2026), history_from=2024)
    assert runner.prepare_signature(models["world_v1"]) == before["world_v1"][0]
    assert runner.model_signature(models["world_v1"]) == before["world_v1"][1]
    # a full-settings change is a new holdout item even when the signals are shared
    assert runner.model_signature(WorldV1(l2=5.0)) != runner.model_signature(WorldV1())
