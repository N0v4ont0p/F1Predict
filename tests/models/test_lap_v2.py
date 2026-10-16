from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from f1p.lake.curate_f1db import Curated
from f1p.models import lap_data as ld
from synthetic import DEG, OFFSET, TRUE, synthetic_laps

pytest.importorskip("nutpie")
pytest.importorskip("pymc")

from f1p.models import lap_v2 as v2
from f1p.models import lap_validate as lv


@pytest.fixture(scope="module")
def tables(curated: Curated) -> dict[str, pl.DataFrame]:
    return {**curated.tables, "laps": synthetic_laps(curated)}


@pytest.fixture(scope="module")
def posterior(tables: dict[str, pl.DataFrame]) -> v2.LapPosterior:
    events = (
        tables["events"]
        .filter(pl.col("event_id").is_in(tables["laps"]["event_id"].unique().to_list()))
        .sort("season", "round")["event_id"]
        .to_list()
    )
    target = events[6]
    laps = tables["laps"].filter(pl.col("event_id").is_in(events[:6]))
    hold = tables["laps"].filter(pl.col("event_id") == target)
    w = ld.build_window(laps, tables, next_event=target, holdout=hold)
    assert w is not None
    # the synthetic laps have no track evolution: fit the model without that term (with it, the
    # zero-variance curves trade off against first-stint wear and mix slowly — plan §A.5)
    priors = v2.V2Priors(track_evolution=False)
    cfg = v2.SampleConfig(draws=300, tune=1000, chains=4, seed=3)
    return v2.fit(w, cfg, priors=priors, keep=400)


def test_recovers_known_effects(posterior: v2.LapPosterior) -> None:
    d = posterior.draws
    assert posterior.n_draws == 400
    comp = dict(zip(ld.OFFSET_COMPOUNDS, d["compound"].mean(axis=0), strict=True))
    for c in ("SOFT", "HARD"):
        assert comp[c] == pytest.approx(OFFSET[c], abs=0.0015)
    # the identifiable quantity: pace on each compound at a typical tyre age, vs MEDIUM
    basis = ld.ispline_basis(np.array([10.0]))[0]
    curve = np.exp(d["wear_log"]) @ basis  # (draws, compounds)
    idx = {c: ld.COMPOUNDS.index(c) for c in ("MEDIUM", "SOFT", "HARD")}
    for c in ("SOFT", "HARD"):
        level = d["compound"][:, ld.OFFSET_COMPOUNDS.index(c)]
        pace10 = level + curve[:, idx[c]] - curve[:, idx["MEDIUM"]]
        truth = OFFSET[c] + (DEG[c] - DEG["MEDIUM"]) * 10
        assert float(pace10.mean()) == pytest.approx(truth, abs=0.0015)
    assert float(d["fuel"].mean()) == pytest.approx(0.0006 * ld.FUEL_SCALE, abs=0.006)
    # wear after 20 laps: SOFT > MEDIUM > HARD, near the generating slopes
    basis = ld.ispline_basis(np.array([20.0]))[0]
    wear = np.exp(d["wear_log"]).mean(axis=0) @ basis
    w20 = dict(zip(ld.COMPOUNDS, wear, strict=True))
    assert w20["SOFT"] > w20["MEDIUM"] > w20["HARD"]
    for c in ("SOFT", "MEDIUM", "HARD"):
        assert w20[c] == pytest.approx(DEG[c] * 20, rel=0.5)


def test_pace_at_the_forecast_race(posterior: v2.LapPosterior) -> None:
    w = posterior.data
    drivers = [x for x, _ in w.entries]
    pace = v2.car_pace_draws(posterior, drivers)
    assert pace.shape == (posterior.n_draws, len(drivers))
    assert np.isfinite(pace).all()
    truth = np.array([TRUE.get(t, 0.015) for _, t in w.entries])
    assert np.corrcoef(pace.mean(axis=0), truth)[0, 1] > 0.9
    assert np.all(pace.std(axis=0) > 0)  # a distribution, not a point
    assert np.isnan(v2.car_pace_draws(posterior, ["nobody"])).all()


def test_diagnostics_and_holdout_density(
    posterior: v2.LapPosterior, tables: dict[str, pl.DataFrame]
) -> None:
    diag = posterior.diagnostics
    assert diag.sampler == "nutpie"
    # a CI-sized fit (300 draws): what forecasts read must have converged; the nuisance
    # parameters (tyre-wear shapes) get a smoke bound. Production fits must pass
    # Diagnostics.ok (plan §A.4).
    assert diag.key_rhat < 1.05
    assert diag.max_rhat < 1.2
    assert diag.min_bfmi > 0.3
    assert diag.divergences >= 0
    assert set(diag.to_dict()) >= {"max_rhat", "min_ess_bulk", "ok", "ess_per_s"}
    d2 = lv.v2_lap_density(posterior)
    d1 = lv.v1_lap_density(posterior.data, tables)
    assert d2.shape == d1.shape
    assert np.isfinite(d2).all()
    ok = np.isfinite(d1)
    # the synthetic laps are what both models assume (Student-t noise): v2 must not lose
    assert d2[ok].mean() > d1[ok].mean() - 0.05


def test_unknown_sampler(posterior: v2.LapPosterior) -> None:
    model = v2.build_model(posterior.data)
    with pytest.raises(ValueError, match="unknown sampler"):
        v2.sample(model, v2.SampleConfig(sampler="metropolis"))


def test_posterior_round_trip(posterior: v2.LapPosterior, tmp_path: Path) -> None:
    path = v2.save_posterior(posterior, tmp_path / "2026-R07", priors=v2.V2Priors())
    draws, meta = v2.load_draws(path)
    assert meta["entries"] == [list(e) for e in posterior.data.entries]
    assert meta["diagnostics"]["sampler"] == "nutpie"
    for name in ("pace_next", "wear_log", "fuel", "pit_event"):
        np.testing.assert_array_equal(draws[name], posterior.draws[name])
    assert "holdout_mu" not in draws  # bulky validation draws are not persisted


def test_sbc_plumbing(tables: dict[str, pl.DataFrame]) -> None:
    events = (
        tables["events"]
        .filter(pl.col("event_id").is_in(tables["laps"]["event_id"].unique().to_list()))
        .sort("season", "round")["event_id"]
        .to_list()
    )
    laps = tables["laps"].filter(pl.col("event_id").is_in(events[:2]))
    w = ld.build_window(laps, tables, next_event=events[2], holdout=None)
    assert w is not None
    cfg = v2.SampleConfig(draws=60, tune=60, chains=1)
    res = lv.sbc(w, 2, cfg=cfg, params=("fuel", "sigma_log"), thin_to=19)
    assert set(res.ranks["param"].unique().to_list()) == {"fuel", "sigma_log"}
    assert res.ranks.height == 4
    assert res.ranks["rank"].is_between(0, 19).all()
    assert set(res.p_values) == {"fuel", "sigma_log"}


def test_lap_pace_v2_feature(
    posterior: v2.LapPosterior, tables: dict[str, pl.DataFrame], tmp_path: Path
) -> None:
    from f1p.eval.context import build_contexts
    from f1p.models.world_v1 import WorldV1

    event = posterior.data.events[-1]
    store = v2.PosteriorStore(tmp_path)
    empty = store.fingerprint()
    v2.save_posterior(posterior, tmp_path / event)
    store = v2.PosteriorStore(tmp_path)
    assert store.fingerprint() != empty  # a new posterior invalidates cached signals
    pace = store.pace(event)
    assert pace is not None
    assert store.pace("1999-R01") is None
    drivers = [d for d, _ in posterior.data.entries]
    mean, unc = v2.lap_pace_v2_signal([*drivers, "nobody"], event, store)
    assert abs(float(mean.mean())) < 1e-9  # centred
    assert np.all(np.isfinite(mean))
    truth = np.array([TRUE.get(t, 0.015) for _, t in posterior.data.entries])
    assert np.corrcoef(mean[: len(drivers)], -truth)[0, 1] > 0.9  # higher = faster
    np.testing.assert_allclose(unc[:-1], mean[:-1] * [pace[d][1] for d in drivers])
    zero = v2.lap_pace_v2_signal(drivers, "1999-R01", store)
    assert np.all(zero[0] == 0)
    assert np.all(zero[1] == 0)

    model = WorldV1(
        features={"C1": ("rating", "lap_pace_v2", "lap_pace_v2_unc")}, lap_v2=str(tmp_path)
    )
    ctx = next(
        c
        for c, _, _ in build_contexts(tables, seasons=(2026, 2026), checkpoints=("C1",))
        if c.event_id == event
    )
    sig = model.prepare(ctx)
    assert sig.x.shape == (ctx.n, 3)
    assert np.any(sig.x[:, 1] != 0)
    other = WorldV1(features=model.features, lap_v2=str(tmp_path / "elsewhere"))
    assert model.prepare_signature() != other.prepare_signature()
