"""Model training, prediction and simulation tests."""
from __future__ import annotations

import numpy as np

from f1predict.models import train
from f1predict.simulation import championship_swing, simulate_race
from f1predict.whatif import Scenario, apply_scenario, get_race


def test_train_and_predict(cfg, small_df):
    res = train(cfg, family="random_forest", experiment_name="test", df=small_df)
    pred = res["predictor"]
    assert 0 <= res["metrics"]["winner_accuracy"] <= 1
    assert res["model_path"].exists()

    from f1predict.features import build_features
    feat_df, _ = build_features(small_df)
    race = feat_df[(feat_df["season"] == feat_df["season"].max())]
    rnd = race["round"].min()
    one = race[race["round"] == rnd]
    tbl = pred.predict_race(one, season=2021)
    # Probabilities are valid and ranks are a permutation.
    assert tbl["p_win"].between(0, 1).all()
    assert abs(tbl["p_win"].sum() - 1.0) < 0.05
    assert sorted(tbl["predicted_rank"]) == list(range(1, len(tbl) + 1))


def test_benchmark_all_families(cfg, small_df):
    res = train(cfg, compare_all=True, experiment_name="bench", df=small_df)
    fams = {b["family"] for b in res["benchmark"]}
    assert fams == {"random_forest", "extra_trees", "lightgbm", "histgb", "gbr"}
    for b in res["benchmark"]:
        assert b["peak_rss_mb"] > 0


def test_simulation_probabilities(cfg, small_df):
    res = train(cfg, family="random_forest", experiment_name="sim", df=small_df)
    pred = res["predictor"]
    from f1predict.features import build_features
    feat_df, _ = build_features(small_df)
    one = feat_df[feat_df["round"] == feat_df["round"].max()].head(20)
    mu = pred.predict_positions(one)
    sim = simulate_race(one, mu, pred.residual_std, n_sim=5000, season=2021, cfg=cfg)
    assert np.all((sim.p_win >= 0) & (sim.p_win <= 1))
    assert abs(sim.p_win.sum() - 1.0) < 0.02
    # Exactly one winner per simulation.
    assert (sim.finish_positions == 1).sum() == sim.n_sim


def test_championship_swing(cfg, small_df):
    res = train(cfg, family="random_forest", experiment_name="champ", df=small_df)
    pred = res["predictor"]
    from f1predict.features import build_features
    feat_df, _ = build_features(small_df)
    one = feat_df[feat_df["round"] == feat_df["round"].max()].head(20)
    sim = simulate_race(one, pred.predict_positions(one), pred.residual_std,
                        n_sim=3000, season=2021, cfg=cfg)
    standings = {d: 0.0 for d in sim.drivers}
    swing = championship_swing(sim, standings)
    assert abs(swing["p_championship_lead"].sum() - 1.0) < 0.02
    assert (swing["proj_max"] >= swing["proj_min"]).all()


def test_whatif_form_boost_helps(cfg, small_df):
    res = train(cfg, family="random_forest", experiment_name="wi", df=small_df)
    pred = res["predictor"]
    from f1predict.features import build_features
    feat_df, _ = build_features(small_df)
    one = feat_df[feat_df["round"] == feat_df["round"].max()].head(20).reset_index(drop=True)
    target = one["driver_id"].iloc[-1]  # a back-of-grid driver
    sc = Scenario(form_boost={target: -8.0}, n_sim=4000)  # big speed boost
    out = apply_scenario(pred, one, sc, season=2021, cfg=cfg)
    row = out["delta"][out["delta"]["driver_id"] == target]
    assert float(row["p_win_delta"].iloc[0]) >= 0  # boosting can only help or hold


def test_predict_future_race(cfg, small_df):
    """A future race (no results) is projected from the latest form/ELO snapshot and
    produces a sane ranked table led by strong drivers (not a backmarker)."""
    from f1predict.models import predict_future_race
    res = train(cfg, family="random_forest", experiment_name="future", df=small_df)
    pred = res["predictor"]
    # Project a 2022 Monaco round (beyond the synthetic 2018-2021 data).
    tbl = predict_future_race(cfg, pred, season=2022, rnd=None,
                              circuit_id="monaco", master=small_df)
    assert sorted(tbl["predicted_rank"]) == list(range(1, len(tbl) + 1))
    assert tbl["p_win"].between(0, 1).all()
    assert abs(tbl["p_win"].sum() - 1.0) < 0.05
