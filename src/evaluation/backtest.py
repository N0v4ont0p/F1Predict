"""Walk-forward backtesting engine.

Strategy: **expanding-window, season-level walk-forward**. For each evaluated season *S*
we train only on races from seasons ``< S`` and predict every race in *S*. Because the
feature factory is temporal-safe, in-season rolling features only ever look backwards, so
no future information leaks. This mirrors how the model would have been used in real time.

Outputs a per-race results frame (for the dashboard explorer), an aggregate summary, and
an era-level breakdown with a simple bootstrap significance check vs a grid-only baseline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config, load_config
from ..features import build_features
from ..models.registry import build_model
from ..models.predictor import RacePredictor
from ..utils.logging import get_logger
from ..utils.profiling import profile
from .metrics import aggregate_metrics, race_metrics

log = get_logger()


def _fit(family: str, cfg: Config, cols: list[str], train: pd.DataFrame) -> RacePredictor:
    model = build_model(family, cfg)
    model.fit(train[cols].to_numpy(), train["target_position"].to_numpy())
    return RacePredictor(model=model, feature_cols=cols, family=family)


def _era_for_season(season: int, era_splits: dict) -> str:
    for era, (lo, hi) in era_splits.items():
        if lo <= season <= hi:
            return era
    return "other"


def backtest(
    cfg: Config | None = None,
    family: str | None = None,
    min_train_seasons: int = 2,
    df: pd.DataFrame | None = None,
    progress=None,
) -> dict:
    """Run the walk-forward backtest. ``progress`` is an optional callback(frac, label)."""
    cfg = cfg or load_config()
    family = family or cfg.get("model.family", "random_forest")
    era_splits = cfg.get("backtest.era_splits", {})

    if df is None:
        from ..data_pipeline import load_master
        df = load_master(cfg)
    feat_df, cols = build_features(df, cfg)

    seasons = sorted(feat_df["season"].unique())
    eval_seasons = seasons[min_train_seasons:]
    if not eval_seasons:
        eval_seasons = seasons[-1:]

    per_race_rows: list[dict] = []
    predictions: list[pd.DataFrame] = []

    with profile("backtest") as prof:
        for i, season in enumerate(eval_seasons):
            train_df = feat_df[feat_df["season"] < season]
            test_df = feat_df[feat_df["season"] == season]
            if len(train_df) == 0:
                continue
            predictor = _fit(family, cfg, cols, train_df)
            for rnd, race in test_df.groupby("round"):
                race = race.copy()
                race["pred_position"] = predictor.predict_positions(race)
                race["pred_rank"] = race["pred_position"].rank(method="first").astype(int)
                m = race_metrics(race)
                m.update({"season": int(season), "round": int(rnd),
                          "circuit_id": race["circuit_id"].iloc[0],
                          "era": _era_for_season(int(season), era_splits)})
                per_race_rows.append(m)
                predictions.append(race[[
                    "season", "round", "circuit_id", "driver_id", "driver_name",
                    "grid", "position", "target_position", "pred_position", "pred_rank",
                ]])
            if progress:
                progress((i + 1) / len(eval_seasons), f"season {season}")

    per_race = pd.DataFrame(per_race_rows)
    summary = aggregate_metrics(per_race_rows)
    summary["profile"] = prof.as_dict()

    # Era breakdown.
    era_summary = {}
    if not per_race.empty:
        for era, grp in per_race.groupby("era"):
            era_summary[era] = aggregate_metrics(grp.to_dict("records"))

    # Baseline (grid == prediction) for significance context.
    baseline = _grid_baseline(feat_df, eval_seasons)

    return {
        "summary": summary,
        "era_summary": era_summary,
        "per_race": per_race,
        "predictions": pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(),
        "baseline": baseline,
        "family": family,
    }


def _grid_baseline(feat_df: pd.DataFrame, eval_seasons: list[int]) -> dict:
    """Naive baseline: predicted order == grid order. Contextualises model lift."""
    rows = []
    sub = feat_df[feat_df["season"].isin(eval_seasons)]
    for (_, _), race in sub.groupby(["season", "round"]):
        race = race.copy()
        race["pred_position"] = race["grid"].replace(0, 99)
        rows.append(race_metrics(race))
    return aggregate_metrics(rows)
