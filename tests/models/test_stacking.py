from __future__ import annotations

import numpy as np
import pytest

from f1p.eval import harness
from f1p.eval.baselines import RankPL, grid_ranks, standings_ranks
from f1p.eval.forecast import Forecast, pl_sample_orders
from f1p.lake.curate_f1db import Curated
from f1p.models import stacking
from f1p.models.stacking import fit_calibrator, stack_result
from f1p.models.world_v1 import WorldV1


@pytest.fixture(scope="module")
def result(curated: Curated) -> harness.EvalResult:
    return harness.run(
        curated.tables,
        [
            WorldV1(),
            RankPL("standings", standings_ranks, frozenset({"C1", "C4"})),
            RankPL("grid", grid_ranks, frozenset({"C4"})),  # a member of C4 only
        ],
        seasons=(2021, 2026),
        history_from=2013,
        keep_forecasts=True,
    )


def test_members_per_checkpoint_and_recalibration(
    curated: Curated, result: harness.EvalResult
) -> None:
    members = ["world_v1", "standings", "grid"]
    stacked, fits = stack_result(
        curated.tables,
        result,
        members,
        seasons=(2026, 2026),
        training_seasons=6,
        recalibrate=("isotonic",),
    )
    # C1 pools the members that forecast C1; C4 all three
    assert fits[(2026, "C1")].members == ("world_v1", "standings")
    assert fits[(2026, "C4")].members == ("world_v1", "standings", "grid")
    pe = stacked.per_event
    c1 = set(pe.filter(pe["checkpoint"] == "C1")["model"].unique().to_list())
    assert c1 == {"world_v1", "standings", "stack", "stack_iso"}
    assert pe.filter(pe["model"] == "stack_iso").height == pe.filter(pe["model"] == "stack").height
    # the recalibrated P(win) is a distribution, keeps the favourites' order, and only P(win)
    # changes (the full-order scores are the pool's)
    for cp in ("C1", "C4"):
        fit = fits[(2026, cp)]
        assert fit.calibrator is not None
        a = pe.filter((pe["model"] == "stack") & (pe["checkpoint"] == cp)).sort("event_id")
        b = pe.filter((pe["model"] == "stack_iso") & (pe["checkpoint"] == cp)).sort("event_id")
        np.testing.assert_allclose(a["finisher_pl_loglik"], b["finisher_pl_loglik"])
        np.testing.assert_allclose(a["podium_brier"], b["podium_brier"])
    summary = stacked.summary()
    assert "ece_top1" in summary.columns
    assert summary["ece_top1"].drop_nulls().len() == summary.height


def test_without_recalibration(curated: Curated, result: harness.EvalResult) -> None:
    stacked, fits = stack_result(
        curated.tables,
        result,
        ["world_v1", "standings"],
        seasons=(2026, 2026),
        training_seasons=6,
    )  # the default: no recalibration
    assert set(stacked.per_event["model"].unique().to_list()) == {"world_v1", "standings", "stack"}
    assert all(f.calibrator is None and not f.calibrations for f in fits.values())
    with pytest.raises(ValueError, match="recalibration"):
        stack_result(
            curated.tables,
            result,
            ["world_v1"],
            seasons=(2026, 2026),
            training_seasons=6,
            recalibrate=("platt",),
        )


def test_temperature_recalibrations_through_the_stack(
    curated: Curated, result: harness.EvalResult
) -> None:
    modes = ("temperature", "temperature_all")
    stacked, fits = stack_result(
        curated.tables,
        result,
        ["world_v1", "standings"],
        seasons=(2026, 2026),
        training_seasons=6,
        recalibrate=modes,
    )
    pe = stacked.per_event
    assert {"stack", "stack_temp", "stack_tempall"} <= set(pe["model"].unique().to_list())
    for (_, cp), fit in fits.items():
        for mode in modes:
            tau = fit.calibrations[mode]
            assert stacking.TEMPERATURE_BOUNDS[0] <= tau <= stacking.TEMPERATURE_BOUNDS[1]
        raw = pe.filter((pe["model"] == "stack") & (pe["checkpoint"] == cp)).sort("event_id")
        temp = pe.filter((pe["model"] == "stack_temp") & (pe["checkpoint"] == cp)).sort("event_id")
        # the P(win) temperature leaves the order's scores alone
        np.testing.assert_allclose(raw["finisher_pl_loglik"], temp["finisher_pl_loglik"])
        np.testing.assert_allclose(raw["podium_brier"], temp["podium_brier"])


def _synthetic(
    seasons: tuple[int, ...], races: int, shrink: float, seed: int
) -> tuple[list, list[int], dict[str, object]]:  # type: ignore[type-arg]
    """Races whose truth is Plackett-Luce on ``true``; member ``a`` forecasts
    ``shrink · true`` (under-confident when shrink < 1), member ``b`` noise."""
    from f1p.eval.metrics import Outcome

    rng = np.random.default_rng(seed)
    n = 10
    drivers = tuple(f"d{i}" for i in range(n))
    samples, sample_seasons, outcomes = [], [], {}
    for s in seasons:
        for r in range(races):
            eid = f"{s}-R{r + 1:02d}"
            true = rng.normal(0, 1.2, n)
            order = pl_sample_orders(true, 1, rng)[0]
            outcomes[eid] = Outcome(eid, tuple(drivers[i] for i in order))
            fa = Forecast.from_log_strength(eid, "C4", "a", drivers, shrink * true, n_samples=200)
            fb = Forecast.from_log_strength(
                eid, "C4", "b", drivers, rng.normal(0, 1, n), n_samples=200
            )
            fcs = [fa, fb]
            samples.append((stacking.design(fcs, drivers), order.astype(np.intp), fcs))
            sample_seasons.append(s)
    return samples, sample_seasons, outcomes


def test_held_out_pools_never_see_their_own_season() -> None:
    samples, seasons, _ = _synthetic((2020, 2021), 25, 1.0, 0)
    held = stacking._held_out_pools(samples, seasons, ["a", "b"], "s")
    assert [fc.event_id for fc in held] == [f2[0].event_id for _, _, f2 in samples]
    # 2020's races come from the pool fitted on 2021 alone, and vice versa
    for s, other in ((2020, 2021), (2021, 2020)):
        rest = [(x, f) for (x, f, _), si in zip(samples, seasons, strict=True) if si == other]
        w = stacking.fit_pool(rest, ["a", "b"]).weights
        mine = next(fc for fc in held if fc.event_id.startswith(str(s)))
        assert mine.meta["w_a"] == pytest.approx(w["a"])
    # one training season: nothing held out (the caller falls back to in-sample)
    assert stacking._held_out_pools(samples[:25], seasons[:25], ["a", "b"], "s") == []


def test_temperature_fits_recover_the_truth() -> None:
    rng = np.random.default_rng(1)
    n, races = 12, 3000
    true = rng.normal(0, 1.0, (races, n))
    winners = [int(pl_sample_orders(t, 1, rng)[0][0]) for t in true]

    def softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()

    for shrink in (0.7, 1.0, 1.4):
        p = [softmax(shrink * t) for t in true]
        tau = stacking.fit_temperature(p, winners, l2=0.0)
        assert tau == pytest.approx(1.0 / shrink, rel=0.08), shrink
    # bounded, and shrunk towards 1 when the data are few
    p = [softmax(0.2 * t) for t in true]
    assert stacking.fit_temperature(p, winners, l2=0.0) == pytest.approx(2.0, abs=1e-3)
    few = [softmax(0.7 * t) for t in true[:30]]
    free = stacking.fit_temperature(few, winners[:30], l2=0.0)
    shrunk = stacking.fit_temperature(few, winners[:30], l2=20.0)
    assert abs(np.log(shrunk)) < abs(np.log(free))
    # the whole-order temperature, on full Plackett-Luce orders
    orders = [pl_sample_orders(t, 1, rng)[0].astype(np.intp) for t in true[:800]]
    for shrink in (0.7, 1.3):
        tau = stacking.fit_temperature_all(
            [(shrink * t, o, None) for t, o in zip(true[:800], orders, strict=True)], l2=0.0
        )
        assert tau == pytest.approx(1.0 / shrink, rel=0.08), shrink


def test_temper() -> None:
    p = np.array([0.5, 0.3, 0.15, 0.05, 0.0])
    for tau in (0.5, 1.0, 2.0):
        q = stacking.temper(p, tau)
        assert q.sum() == pytest.approx(1.0)
        assert list(np.argsort(-q[:4])) == [0, 1, 2, 3]
    np.testing.assert_allclose(stacking.temper(p, 1.0), p, atol=1e-9)
    assert stacking.temper(p, 2.0)[0] > p[0] > stacking.temper(p, 0.5)[0]


def test_calibrator_is_monotone_and_needs_both_outcomes() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 0.6, 4000)
    won = (rng.uniform(size=p.size) < np.sqrt(p)).astype(float)  # an under-confident forecast
    iso = fit_calibrator(p, won)
    q = iso.predict(np.sort(p))
    assert np.all(np.diff(q) >= 0)
    assert iso.predict(np.array([0.25]))[0] == pytest.approx(0.5, abs=0.06)
    assert fit_calibrator(p, np.zeros_like(p)) is None
