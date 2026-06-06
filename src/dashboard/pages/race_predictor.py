"""Page: Race Predictor (main dashboard)."""
from __future__ import annotations

import plotly.express as px
import streamlit as st

from ..data import get_model
from ..theme import PLOTLY_TEMPLATE
from ..widgets import race_selector


def render():
    st.header("🏁 Race Predictor")
    season, rnd, race = race_selector("predict")
    if race.empty:
        st.warning("No data for this race.")
        return
    model = get_model()
    tbl = model.predict_race(race, season=season)

    fav = tbl.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted Winner", fav["driver_name"], f"{fav['p_win']:.0%} win")
    c2.metric("From Grid", f"P{int(fav['grid'])}")
    surprise = tbl.assign(gain=tbl["grid"] - tbl["predicted_rank"]).sort_values("gain", ascending=False).iloc[0]
    c3.metric("Biggest Mover", surprise["driver_name"], f"+{int(surprise['grid']-surprise['predicted_rank'])} places")
    c4.metric("Field Size", len(tbl))

    st.subheader("Predicted Classification")
    show = tbl[["predicted_rank", "driver_name", "constructor_name", "grid",
                "p_win", "p_podium", "p_points", "expected_points"]].copy()
    show.columns = ["#", "Driver", "Team", "Grid", "P(win)", "P(podium)", "P(points)", "xPts"]
    st.dataframe(
        show.style.format({"P(win)": "{:.1%}", "P(podium)": "{:.1%}",
                           "P(points)": "{:.1%}", "xPts": "{:.1f}"})
        .background_gradient(subset=["P(win)", "P(podium)"], cmap="Reds"),
        use_container_width=True, hide_index=True, height=560,
    )

    st.subheader("Win Probability")
    top = tbl.head(10)
    fig = px.bar(top, x="p_win", y="driver_name", orientation="h",
                 labels={"p_win": "P(win)", "driver_name": ""})
    fig.update_layout(PLOTLY_TEMPLATE["layout"], yaxis={"categoryorder": "total ascending"},
                      height=400)
    st.plotly_chart(fig, use_container_width=True)

    if st.button("▶ Run Monte Carlo on this race"):
        st.session_state["mc_season"] = season
        st.session_state["mc_round"] = rnd
        st.session_state["_goto"] = "Monte Carlo Simulator"
        st.rerun()
