"""Data pipeline + feature engineering tests (incl. temporal-safety sanity)."""
from __future__ import annotations

import pandas as pd

from f1predict.data_pipeline import build_master
from f1predict.data_pipeline.schema import RAW_COLUMNS
from f1predict.features import build_features


def test_synthetic_build_schema(small_df):
    assert set(RAW_COLUMNS).issubset(small_df.columns)
    assert len(small_df) > 0
    # Every race has exactly one winner (position == 1).
    races = small_df.groupby(["season", "round"])
    for _, race in races:
        assert (race["position"] == 1).sum() == 1


def test_build_master_persists(cfg):
    df = build_master(cfg, seasons=[2018, 2019], source="synthetic")
    assert cfg.path("paths.master_dataset").exists()
    assert df["season"].nunique() == 2


def test_features_no_leakage_columns(small_df):
    feat_df, cols = build_features(small_df)
    # Feature columns must not include raw outcomes.
    for banned in ("position", "points", "finished", "status"):
        assert banned not in cols


def test_features_no_nan(small_df):
    feat_df, cols = build_features(small_df)
    assert not feat_df[cols].isna().any().any()


def test_target_position_valid(small_df):
    feat_df, _ = build_features(small_df)
    # Target positions are positive and DNFs ranked beyond the field.
    assert (feat_df["target_position"] >= 1).all()


def test_rolling_feature_is_backward_only(small_df):
    """A driver's first-ever race must have a neutral (filled) rolling form value, never
    derived from that same race's result."""
    feat_df, cols = build_features(small_df)
    form_cols = [c for c in cols if c.startswith("f_drv_pos_")]
    assert form_cols  # exists
    # First appearance per driver: rolling mean excludes current row, so it's the global
    # fill (median) not the driver's own finishing position — verify no perfect correlation.
    first = feat_df.sort_values(["season", "round"]).groupby("driver_id").head(1)
    corr = first[form_cols[0]].corr(first["target_position"])
    assert pd.isna(corr) or abs(corr) < 0.99
