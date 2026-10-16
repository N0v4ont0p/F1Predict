from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from f1p.eval import harness
from f1p.eval.baselines import InRacePL, InRaceSignal, Uniform, running_order
from f1p.eval.context import (
    IN_RACE,
    IN_RACE_FRACTIONS,
    build_contexts,
    checkpoint_lap,
    cut_lap,
    live_context,
)
from f1p.eval.in_race import _gaps, race_so_far, race_state
from f1p.lake.curate_f1db import Curated
from f1p.models.in_race import pit_cycle, pit_cycle_baseline, stops_owed
from synthetic import synthetic_laps, with_positions


@pytest.fixture(scope="module")
def tables(curated: Curated) -> dict[str, pl.DataFrame]:
    return {**curated.tables, "laps": with_positions(synthetic_laps(curated))}


def test_cut_lap() -> None:
    assert cut_lap(57) == 19
    assert cut_lap(58) == 20  # ceil
    assert cut_lap(3) == 2
    assert cut_lap(2) == 2


def test_checkpoint_laps() -> None:
    assert IN_RACE == ("C5a", "C5", "C5b", "C5c")
    assert IN_RACE_FRACTIONS["C5"] == pytest.approx(1 / 3)  # unchanged: C5 stays comparable
    # a 57-lap race: ceil(57/4), ceil(57/3), ceil(57/2), ceil(2*57/3)
    assert [checkpoint_lap(cp, 57) for cp in IN_RACE] == [15, 19, 29, 38]
    for laps in range(10, 90):
        ks = [checkpoint_lap(cp, laps) for cp in IN_RACE]
        assert ks == sorted(ks)
        assert ks[0] >= 2
        assert ks[-1] < laps
    assert checkpoint_lap("C5", 57) == cut_lap(57)
    with pytest.raises(KeyError):
        checkpoint_lap("C4", 57)


@pytest.mark.parametrize("cp", IN_RACE)
def test_c5_contexts(tables: dict[str, pl.DataFrame], cp: str) -> None:
    both = build_contexts(tables, seasons=(2026, 2026), checkpoints=("C4", cp))
    c4 = {c.event_id: c for c, _, _ in both if c.checkpoint == "C4"}
    c5 = [(c, o) for c, o, _ in both if c.checkpoint == cp]
    timed = set(tables["laps"]["event_id"].unique().to_list())
    assert c5
    assert {c.event_id for c, _ in c5} <= timed  # only races with lap timing
    for ctx, _ in c5:
        ev = tables["events"].filter(pl.col("event_id") == ctx.event_id).row(0, named=True)
        assert ctx.meta["race_laps"] == ev["laps"]
        assert ctx.meta["lap"] == checkpoint_lap(cp, ev["laps"])
        # the tables are cut where C4 cuts them; the field is the grid
        assert ctx.as_of == c4[ctx.event_id].as_of
        assert ctx.drivers == c4[ctx.event_id].drivers
        so_far = race_so_far(ctx)
        assert so_far["event_id"].unique().to_list() == [ctx.event_id]
        assert so_far["lap"].max() == ctx.meta["lap"]
    # a quali target has no in-race checkpoint
    assert not build_contexts(tables, seasons=(2026, 2026), checkpoints=(cp,), target="quali")
    # and without lap timing there is none either
    plain = {k: v for k, v in tables.items() if k != "laps"}
    assert not build_contexts(plain, seasons=(2026, 2026), checkpoints=(cp,))


def test_later_checkpoints_cut_later(tables: dict[str, pl.DataFrame]) -> None:
    ctxs = build_contexts(tables, seasons=(2026, 2026), checkpoints=IN_RACE)
    by_event: dict[str, dict[str, int]] = {}
    for c, _, _ in ctxs:
        by_event.setdefault(c.event_id, {})[c.checkpoint] = c.meta["lap"]
    full = [laps for laps in by_event.values() if len(laps) == len(IN_RACE)]
    assert full
    for laps in full:
        ks = [laps[cp] for cp in IN_RACE]
        assert ks == sorted(ks)
        assert len(set(ks)) == len(ks)


def test_race_so_far_needs_an_in_race_checkpoint(tables: dict[str, pl.DataFrame]) -> None:
    ctx, _, _ = build_contexts(tables, seasons=(2026, 2026), checkpoints=("C4",))[0]
    with pytest.raises(ValueError, match="in-race"):
        race_so_far(ctx)


@pytest.mark.parametrize("cp", IN_RACE)
def test_race_state(tables: dict[str, pl.DataFrame], cp: str) -> None:
    for ctx, outcome, _ in build_contexts(tables, seasons=(2026, 2026), checkpoints=(cp,)):
        st = race_state(ctx)
        assert sorted(st.position.tolist()) == list(range(1, ctx.n + 1))
        # running cars are ahead of retired ones; every classified car was running at lap k
        assert st.position[st.running].max(initial=0) < st.position[~st.running].min(initial=99)
        idx = ctx.index()
        assert all(st.running[idx[d]] for d in outcome.classified)
        # gaps grow down the running order, from 0 for the leader
        live = np.flatnonzero(st.running & np.isfinite(st.gap_s))
        by_pos = live[np.argsort(st.position[live])]
        assert st.gap_s[by_pos[0]] == 0.0
        assert np.all(np.diff(st.gap_s[by_pos]) >= 0)
        assert np.all(np.isnan(st.gap_s[~st.running]))
        assert st.laps_left == ctx.meta["race_laps"] - ctx.meta["lap"]


def test_gap_chain_fills_missing_intervals() -> None:
    laps = pl.DataFrame(
        {
            "driver_id": ["a", "b", "c", "a", "b", "c"],
            "lap": [1, 1, 1, 2, 2, 2],
            "lap_ms": [90_000, 91_000, 93_000, 90_000, 90_500, 90_000],
            "position": [1, 2, 3, 1, 2, 3],
            "gap_ahead_s": [None, 1.0, 2.0, None, 1.4, None],  # c's interval is missing
        }
    )
    gap = _gaps(laps, 2, ("a", "b", "c", "x"))
    # b: the interval; c: summed lap times (183.0 - 180.0 = 3.0), never ahead of b
    np.testing.assert_allclose(gap[:3], [0.0, 1.4, 3.0])
    assert np.isnan(gap[3])  # a driver without laps


@pytest.mark.parametrize("cp", IN_RACE)
def test_nothing_after_lap_k_is_seen(tables: dict[str, pl.DataFrame], cp: str) -> None:
    """Scrambling a race's laps after k changes none of its in-race signals: the cut holds.
    (Earlier races are public in full, so only the race's own laps are scrambled.)"""
    ctxs = build_contexts(tables, seasons=(2026, 2026), checkpoints=(cp,))
    laps = tables["laps"]
    rng = np.random.default_rng(1)
    for same in ctxs[:3]:
        ctx = same[0]
        later = (pl.col("event_id") == ctx.event_id) & (pl.col("lap") > ctx.meta["lap"])
        tail = laps.filter(later)
        scrambled = pl.concat(
            [
                laps.filter(~later),
                tail.with_columns(
                    pl.Series("position", rng.permutation(tail["position"].to_numpy())),
                    pl.Series("lap_ms", rng.permutation(tail["lap_ms"].to_numpy())),
                    pl.lit(True).alias("in_lap"),
                ),
            ]
        )
        other = build_contexts(
            {**tables, "laps": scrambled}, seasons=(2026, 2026), checkpoints=(cp,)
        )
        twin = next(c for c, _, _ in other if c.event_id == ctx.event_id)
        for feature in (running_order, pit_cycle):
            a, b = feature(ctx), feature(twin)
            assert a is not None
            assert b is not None
            np.testing.assert_array_equal(a.score, b.score)
            np.testing.assert_array_equal(a.running, b.running)


def test_stops_owed() -> None:
    p = np.array([0.0, 0.5, 0.5, 0.0])
    np.testing.assert_allclose(stops_owed(np.array([0.0, 1.0, 2.0, 3.0]), p), [1.5, 0.5, 0, 0])


def test_in_race_baselines_through_the_harness(tables: dict[str, pl.DataFrame]) -> None:
    models = [Uniform(), InRacePL("running_order", running_order), pit_cycle_baseline()]
    res = harness.run(
        tables,
        models,
        seasons=(2026, 2026),
        history_from=2026,
        checkpoints=IN_RACE,
        keep_forecasts=True,
    )
    df = res.per_event
    assert set(df["checkpoint"].unique().to_list()) == set(IN_RACE)
    assert set(df["model"].unique().to_list()) == {"uniform", "running_order", "pit_cycle"}
    contexts = {
        (c.event_id, c.checkpoint): c
        for c, _, _ in build_contexts(tables, seasons=(2026, 2026), checkpoints=IN_RACE)
    }
    for (model, cp, event_id), fc in res.forecasts.items():
        if model == "uniform":
            continue
        ctx = contexts[(event_id, cp)]
        st = race_state(ctx)
        assert fc.p_finish is not None
        assert np.all(fc.p_finish[~st.running] == 0.0)  # a retired car cannot finish ...
        assert np.all(fc.p_win[~st.running] == 0.0)  # ... nor win
        assert np.isclose(fc.p_win.sum(), 1.0)
        assert fc.meta["beta"] >= 0.0


def test_in_race_scores_put_retirements_last() -> None:
    sig = InRaceSignal(np.array([1.0, 2.0, np.nan, 3.0]), np.array([True, True, False, True]))
    s = InRacePL.scores(sig)
    assert np.isfinite(s).all()
    assert s[2] > s[[0, 1, 3]].max()


def test_live_c5_context(tables: dict[str, pl.DataFrame]) -> None:
    event = tables["laps"]["event_id"][0]
    ctx = live_context(tables, event, "C5", lap=12)
    assert ctx.meta["lap"] == 12
    assert race_so_far(ctx)["lap"].max() == 12
    # without a lap: the checkpoint's share of the distance
    half = live_context(tables, event, "C5b")
    assert half.meta["lap"] == checkpoint_lap("C5b", half.meta["race_laps"])
