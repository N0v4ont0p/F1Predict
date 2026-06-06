"""Page: Backtesting Explorer & Model Performance."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from ..data import get_config
from ..theme import PLOTLY_TEMPLATE


@st.cache_data(show_spinner="Running walk-forward backtest…")
def _run_backtest(min_races: int, family: str):
    from ...evaluation import backtest
    res = backtest(get_config(), family=family, min_train_seasons=min_races)
    return res["summary"], res["era_summary"], res["per_race"], res["baseline"]


def render():
    st.header("📊 Backtesting Explorer")
    col1, col2 = st.columns(2)
    family = col1.selectbox("Model family", ["random_forest", "lightgbm", "histgb"])
    min_races = col2.slider("Min training seasons", 1, 5, 2)

    summary, era_summary, per_race, baseline = _run_backtest(min_races, family)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", f"{summary['mae']:.2f}", f"{summary['mae']-baseline['mae']:+.2f} vs grid")
    c2.metric("Spearman ρ", f"{summary['spearman']:.3f}")
    c3.metric("Winner Acc", f"{summary['winner_accuracy']:.1%}")
    c4.metric("Races", summary["n_races"])

    if per_race.empty:
        st.info("Not enough seasons to backtest. Build more data.")
        return

    st.subheader("Accuracy Over Time")
    per_race["label"] = per_race["season"].astype(str) + "-R" + per_race["round"].astype(str)
    trend = per_race.groupby("season")[["winner_hit", "podium_overlap", "mae"]].mean().reset_index()
    fig = px.line(trend, x="season", y=["winner_hit", "podium_overlap"], markers=True)
    fig.update_layout(PLOTLY_TEMPLATE["layout"], height=340)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Performance by Circuit (MAE)")
    by_circ = per_race.groupby("circuit_id")["mae"].mean().reset_index().sort_values("mae")
    fig2 = px.bar(by_circ, x="circuit_id", y="mae")
    fig2.update_layout(PLOTLY_TEMPLATE["layout"], height=340)
    st.plotly_chart(fig2, use_container_width=True)

    if era_summary:
        st.subheader("By Era")
        st.dataframe(pd.DataFrame(era_summary).T, use_container_width=True)

    st.subheader("Per-Race Results")
    st.dataframe(per_race, use_container_width=True, height=300)
