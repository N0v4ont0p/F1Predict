"""Page: Monte Carlo Race Simulator."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from ..data import get_config, get_model
from ..theme import PLOTLY_TEMPLATE, ACCENT
from ..widgets import race_selector


def render():
    st.header("🎲 Monte Carlo Race Simulator")
    season, rnd, race = race_selector("mc")
    if race.empty:
        st.warning("No data for this race.")
        return

    col1, col2 = st.columns([3, 1])
    n_sim = col1.slider("Number of simulations", 1000, 50000, 10000, step=1000)
    weather = col2.selectbox("Weather", ["dry", "rain"])

    model = get_model()
    from ...simulation import simulate_race
    mu = model.predict_positions(race)
    with st.spinner(f"Running {n_sim:,} simulations…"):
        sim = simulate_race(race, mu, model.residual_std, n_sim=n_sim,
                            season=season, weather=weather, cfg=get_config())
    frame = sim.to_frame()

    fav = frame.iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Most Likely Winner", fav["driver_name"], f"{fav['p_win']:.1%}")
    c2.metric("Simulations", f"{n_sim:,}")
    c3.metric("Weather", weather.title())

    st.subheader("Winner Probability")
    top = frame.head(10)
    fig = px.bar(top, x="p_win", y="driver_name", orientation="h")
    fig.update_layout(PLOTLY_TEMPLATE["layout"], height=380,
                      yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Points Distribution (top 6)")
    fig2 = go.Figure()
    top6 = frame.head(6)["driver_id"].tolist()
    names = dict(zip(sim.drivers, sim.driver_names))
    for did in top6:
        idx = sim.drivers.index(did)
        fig2.add_trace(go.Violin(y=sim.points_samples[:, idx], name=names[did],
                                 box_visible=True, meanline_visible=True))
    fig2.update_layout(PLOTLY_TEMPLATE["layout"], height=420)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Podium Probability Heat")
    show = frame[["driver_name", "p_win", "p_podium", "p_points", "exp_points"]].copy()
    show.columns = ["Driver", "P(win)", "P(podium)", "P(points)", "xPts"]
    st.dataframe(
        show.style.format({"P(win)": "{:.1%}", "P(podium)": "{:.1%}",
                           "P(points)": "{:.1%}", "xPts": "{:.1f}"})
        .background_gradient(subset=["P(win)", "P(podium)", "P(points)"], cmap="Reds"),
        use_container_width=True, hide_index=True)

    st.download_button("⬇ Download full simulation (CSV)",
                       frame.to_csv(index=False), file_name=f"sim_{season}_{rnd}.csv")
