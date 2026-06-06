"""Shared analysis context: standings, race selection, and race preparation.

These helpers are used by both the CLI and the dashboard so the two surfaces always
agree on numbers. They operate on the engineered feature frame so predictions and
simulations share identical inputs.
"""
from __future__ import annotations

import pandas as pd

from ..config import Config, load_config
from ..features import build_features


_FEATURE_CACHE: dict[tuple, tuple] = {}


def get_feature_frame(df: pd.DataFrame | None = None, cfg: Config | None = None):
    cfg = cfg or load_config()
    if df is None:
        from ..data_pipeline import load_master
        df = load_master(cfg)
    # ``load_master`` returns a stable cached object, so identity + length is a safe,
    # cheap key within a session — the heavy feature build then runs at most once.
    key = (id(df), len(df), str(cfg.get("features.version", "")))
    cached = _FEATURE_CACHE.get(key)
    if cached is not None:
        return cached
    out = build_features(df, cfg)
    _FEATURE_CACHE.clear()  # bound memory: keep only the most recent frame
    _FEATURE_CACHE[key] = out
    return out


def clear_feature_cache() -> None:
    """Drop the cached engineered feature frame (used by the shell `/reload`)."""
    _FEATURE_CACHE.clear()


def compute_standings(df: pd.DataFrame, season: int, upto_round: int | None = None) -> pd.DataFrame:
    """Driver championship standings for ``season`` up to (and including) ``upto_round``."""
    sub = df[df["season"] == season]
    if upto_round is not None:
        sub = sub[sub["round"] <= upto_round]
    standings = (
        sub.groupby(["driver_id", "driver_name", "constructor_name"], as_index=False)["points"]
        .sum()
        .sort_values("points", ascending=False)
        .reset_index(drop=True)
    )
    standings.index = standings.index + 1
    standings.index.name = "rank"
    return standings


def list_races(feat_df: pd.DataFrame) -> pd.DataFrame:
    """Distinct races available for prediction (season, round, name, circuit)."""
    return (
        feat_df[["season", "round", "race_name", "circuit_id"]]
        .drop_duplicates()
        .sort_values(["season", "round"])
        .reset_index(drop=True)
    )


def get_race(feat_df: pd.DataFrame, season: int, rnd: int) -> pd.DataFrame:
    race = feat_df[(feat_df["season"] == season) & (feat_df["round"] == rnd)]
    return race.copy().reset_index(drop=True)


def upcoming_race(feat_df: pd.DataFrame) -> tuple[int, int]:
    """Return ``(season, round)`` of the most recent race — treated as 'next up' for demo
    prediction. With live data this would point at the next scheduled event."""
    last = feat_df.sort_values(["season", "round"]).iloc[-1]
    return int(last["season"]), int(last["round"])
