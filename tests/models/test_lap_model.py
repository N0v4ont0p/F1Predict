from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from f1p.eval.context import build_contexts
from f1p.lake.curate_f1db import Curated
from f1p.lake.laps import lap_rows
from f1p.lake.race_pace import Lap
from f1p.models import lap_model as lm
from synthetic import OFFSET, TRUE, synthetic_laps


@pytest.fixture(scope="module")
def tables(curated: Curated) -> dict[str, pl.DataFrame]:
    return {**curated.tables, "laps": synthetic_laps(curated)}


def test_fit_recovers_known_effects(tables: dict[str, pl.DataFrame]) -> None:
    fitted = lm.fit(tables["laps"], tables)
    assert fitted is not None
    assert fitted.degradation("SOFT") > fitted.degradation("MEDIUM") > fitted.degradation("HARD")
    for comp in ("SOFT", "HARD"):
        assert fitted.compound_offset(comp) == pytest.approx(OFFSET[comp], abs=0.0015)
    assert fitted.value("fuel") == pytest.approx(0.0006, abs=0.0002)
    last = fitted.events[-1]
    est = {t: fitted.value(f"car:{t}:{last}") for t in TRUE if f"car:{t}:{last}" in fitted.index}
    ids = sorted(est)
    a = np.array([est[t] for t in ids])
    b = np.array([TRUE[t] for t in ids])
    assert np.corrcoef(a, b)[0, 1] > 0.97
    assert fitted.sigma == pytest.approx(0.003 * 1.15, rel=0.35)  # robust scale of t(4) noise


def test_signal_is_causal_and_ranks_cars(tables: dict[str, pl.DataFrame]) -> None:
    contexts = build_contexts(tables, seasons=(2026, 2026), checkpoints=("C1",))
    ctx = next(c for c, _, _ in contexts if c.event_id == "2026-R10")
    laps, window = lm.window_laps(ctx, 24)
    assert "2026-R10" not in window  # this race's laps are not public on Thursday
    assert set(laps["event_id"].unique().to_list()) == set(window)
    assert len(window) == 9
    sig = lm.lap_pace_signal(ctx, lm.LapModelCache())
    assert abs(float(sig.mean())) < 1e-9
    lineage = dict(
        tables["constructor_seasons"]
        .filter(pl.col("season") == 2026)
        .select("constructor_id", "lineage_id")
        .iter_rows()
    )
    teams = [lineage.get(c) or c for c in ctx.constructors]
    truth = np.array([-TRUE.get(t, 0.015) for t in teams])
    assert np.corrcoef(sig, truth)[0, 1] > 0.9
    # first race of the season: no laps public yet
    first = next(c for c, _, _ in contexts if c.event_id == "2026-R01")
    assert np.all(lm.lap_pace_signal(first, lm.LapModelCache()) == 0)


def test_pit_loss(tables: dict[str, pl.DataFrame]) -> None:
    loss = lm.pit_loss(tables["laps"])
    assert loss.height > 0
    assert (loss["stops"] > 0).all()


def test_world_model_uses_lap_pace(tables: dict[str, pl.DataFrame]) -> None:
    from f1p.models.registry import registered_models

    model = registered_models()["world_v1_lap"]
    contexts = build_contexts(tables, seasons=(2026, 2026), checkpoints=("C1",))
    ctx = next(c for c, _, _ in contexts if c.event_id == "2026-R12")
    sig = model.prepare(ctx)
    assert sig.x.shape == (ctx.n, 2)
    assert sig.x[:, 1].std() > 0


def test_lap_rows_flags_and_stints() -> None:
    laps = [
        Lap("1", n, 90.0 + (30 if n in (10, 11) else 0), n * 90.0, n == 10, n == 11)
        for n in range(1, 21)
    ]
    rows = lap_rows(laps, neutral=[(14 * 90.0 - 1, 15 * 90.0 + 1)])
    by = {r["lap"]: r for r in rows}
    assert not by[1]["clean"]
    assert not by[10]["clean"]
    assert not by[11]["clean"]
    assert by[15]["neutral"]
    assert not by[15]["clean"]
    assert by[5]["clean"]
    # without a tyre feed, stints are counted from in-laps
    assert by[10]["stint"] == 1
    assert by[11]["stint"] == 2
    assert by[11]["tyre_age"] == 1
