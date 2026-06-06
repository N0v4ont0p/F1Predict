"""Shared dashboard widgets (race selector, etc.)."""
from __future__ import annotations

import streamlit as st

from .data import get_features


def race_selector(key: str = "race"):
    """Sidebar/inline race selector returning ``(season, round, race_row)``."""
    feat_df, _ = get_features()
    races = (feat_df[["season", "round", "race_name"]]
             .drop_duplicates().sort_values(["season", "round"]))
    seasons = sorted(races["season"].unique(), reverse=True)
    col1, col2 = st.columns(2)
    season = col1.selectbox("Season", seasons, key=f"{key}_season")
    season_races = races[races["season"] == season]
    labels = {f"R{int(r['round'])} · {r['race_name']}": int(r["round"])
              for _, r in season_races.iterrows()}
    label = col2.selectbox("Race", list(labels.keys()), key=f"{key}_round")
    rnd = labels[label]
    race = feat_df[(feat_df["season"] == season) & (feat_df["round"] == rnd)].copy()
    return int(season), int(rnd), race
