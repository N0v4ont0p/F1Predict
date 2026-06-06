"""Tests for multi-session prediction (race / qualifying / sprint), the sprint calendar,
and the data-update plumbing."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from f1predict.cli import app
from f1predict.cli.main import _refresh_seasons
from f1predict.data_pipeline import build_master, weekend_has_sprint, sprint_rounds
from f1predict.data_pipeline.schema import RAW_COLUMNS
from f1predict.features import build_features, session_target, session_feature_cols
from f1predict.features.factory import QUALI_LEAKAGE_FEATURES
from f1predict.models import load_production
from f1predict.models.train import train, SESSIONS

runner = CliRunner()


# --------------------------------------------------------------------------- features
def test_session_target_mapping():
    assert session_target("race") == "target_position"
    assert session_target("qualifying") == "target_quali"
    assert session_target("sprint") == "target_sprint"
    with pytest.raises(ValueError):
        session_target("bogus")


def test_build_features_has_session_targets(small_df):
    feat_df, _ = build_features(small_df)
    for col in ("target_position", "target_quali", "target_sprint"):
        assert col in feat_df.columns
        assert (feat_df[col] > 0).all()
        assert not feat_df[col].isna().any()


def test_quali_form_features_present_and_leakfree(small_df):
    feat_df, cols = build_features(small_df)
    form = [c for c in cols if c.startswith("f_quali_form_")]
    assert form, "expected rolling qualifying-form features"
    # Rolling past form is NOT in the leakage set (it's the predictive signal).
    assert not (set(form) & QUALI_LEAKAGE_FEATURES)


def test_session_feature_cols_qualifying_drops_leakage(small_df):
    _, cols = build_features(small_df)
    quali_cols = session_feature_cols(cols, "qualifying")
    # None of the current-weekend grid/quali features survive for the quali model.
    assert not (set(quali_cols) & QUALI_LEAKAGE_FEATURES)
    # Race & sprint keep the full feature set.
    assert session_feature_cols(cols, "race") == cols
    assert session_feature_cols(cols, "sprint") == cols
    # Specific known-leaky columns are gone.
    for leaky in ("f_grid", "f_quali_pos", "f_grid_penalty", "f_teammate_quali_delta"):
        assert leaky not in quali_cols


# --------------------------------------------------------------------------- sprint calendar
def _sprint_master() -> pd.DataFrame:
    """Tiny master-shaped frame: round 1 is a sprint weekend, round 2 is not."""
    rows = []
    for rnd, circ, has in [(1, "interlagos", 1), (2, "monaco", 0)]:
        for did in ("a", "b", "c"):
            row = {c: (0 if t in ("int32", "float32") else "") for c, t in RAW_COLUMNS.items()}
            row.update(season=2025, round=rnd, circuit_id=circ, race_name=f"{circ} GP",
                       driver_id=did, position=1, has_sprint=has,
                       sprint_position=(1 if has else 0))
            rows.append(row)
    return pd.DataFrame(rows)


def test_weekend_has_sprint_from_master():
    m = _sprint_master()
    assert weekend_has_sprint(2025, 1, "interlagos", master=m) is True
    assert weekend_has_sprint(2025, 2, "monaco", master=m) is False
    # Identify by circuit alone.
    assert weekend_has_sprint(2025, None, "interlagos", master=m) is True


def test_sprint_rounds_from_master():
    m = _sprint_master()
    rounds = sprint_rounds(2025, master=m)
    circuits = {r["circuit_id"] for r in rounds}
    assert "interlagos" in circuits
    assert "monaco" not in circuits


# --------------------------------------------------------------------------- update plumbing
def test_refresh_seasons_includes_current_year(cfg):
    import datetime
    cfg.set("data.start_season", 2014)
    cfg.set("data.end_season", datetime.date.today().year)
    seasons = _refresh_seasons(cfg)
    assert datetime.date.today().year in seasons


def test_build_master_force_refresh_accepts_params(cfg):
    # force_refresh / refresh_seasons must be accepted and produce a valid master.
    df = build_master(cfg, seasons=[2018, 2019], source="synthetic",
                      force_refresh=True, refresh_seasons=[2019])
    assert df["season"].nunique() == 2


# --------------------------------------------------------------------------- training
def test_train_race_and_qualifying_sessions(cfg, small_df):
    race = train(cfg, df=small_df, session="race", experiment_name="t")
    quali = train(cfg, df=small_df, session="qualifying", experiment_name="t")
    # Qualifying trains on a strictly smaller (leakage-free) feature set.
    assert len(quali["feature_cols"]) < len(race["feature_cols"])
    # Each session has its own production pointer + loads back with the right metadata.
    store = cfg.path("paths.models_store")
    assert (store / "production.txt").exists()
    assert (store / "production_qualifying.txt").exists()
    assert load_production(cfg, session="race").metadata["session"] == "race"
    assert load_production(cfg, session="qualifying").metadata["session"] == "qualifying"


def test_train_sprint_without_data_raises(cfg, small_df):
    # Synthetic data has no sprint weekends -> a clear refusal, not a crash.
    with pytest.raises(ValueError, match="sprint"):
        train(cfg, df=small_df, session="sprint", experiment_name="t")


def test_predict_race_session_points(cfg, small_df):
    pred = train(cfg, df=small_df, session="race", experiment_name="t")["predictor"]
    feat_df, _ = build_features(small_df)
    one = feat_df[(feat_df["season"] == 2021)]
    one = one[one["round"] == one["round"].min()].copy()

    # Qualifying scores no championship points.
    q = pred.predict_race(one, season=2021, session="qualifying")
    assert np.allclose(q["expected_points"].to_numpy(), 0.0)

    # Sprint uses the sprint table (winner tops out at 8 pts).
    s = pred.predict_race(one, season=2021, session="sprint")
    assert s["expected_points"].max() <= 8.0 + 1e-6
    # Race uses the full points table (winner worth far more than a sprint win).
    r = pred.predict_race(one, season=2021, session="race")
    assert r["expected_points"].max() > s["expected_points"].max()


# --------------------------------------------------------------------------- CLI
def test_predict_session_arg_in_help():
    res = runner.invoke(app, ["predict", "--help"])
    assert res.exit_code == 0
    assert "session" in res.stdout.lower()


def test_update_command_help():
    res = runner.invoke(app, ["update", "--help"])
    assert res.exit_code == 0
    assert "latest" in res.stdout.lower()


def test_train_session_options_in_help():
    res = runner.invoke(app, ["train", "--help"])
    assert res.exit_code == 0
    out = res.stdout.lower()
    assert "session" in out and "all-sessions" in out


def test_predict_sprint_on_non_sprint_weekend_rejected(tmp_path, small_df):
    # Build a synthetic master (no sprint weekends) and ask for a sprint -> friendly refusal.
    master = tmp_path / "master.parquet"
    small_df.to_parquet(master)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"paths:\n  master_dataset: {master}\n"
                   f"  models_store: {tmp_path / 'models_store'}\n")
    # 'monaco2021' resolves to a real (non-sprint) weekend in the synthetic master.
    res = runner.invoke(app, ["predict", "monaco2021", "sprint", "--force", "--config", str(cfg)])
    assert res.exit_code == 1
    assert "not a sprint weekend" in res.stdout.lower()
