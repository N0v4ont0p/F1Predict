"""Page: Championship Simulator & What-If Lab."""
from __future__ import annotations

import plotly.express as px
import streamlit as st

from ..data import get_config, get_master, get_model
from ..theme import PLOTLY_TEMPLATE
from ..widgets import race_selector


def render():
    st.header("🏆 Championship & What-If Lab")
    season, rnd, race = race_selector("wi")
    if race.empty:
        st.warning("No data for this race.")
        return

    df = get_master()
    from ...whatif import Scenario, apply_scenario, compute_standings

    standings = compute_standings(df, season, rnd)
    st.subheader(f"Standings after {season} R{rnd}")
    st.dataframe(standings.head(10), use_container_width=True)

    st.subheader("What-If Controls")
    drivers = race[["driver_id", "driver_name"]].drop_duplicates()
    name_to_id = dict(zip(drivers["driver_name"], drivers["driver_id"]))
    col1, col2, col3 = st.columns(3)
    driver_name = col1.selectbox("Driver", list(name_to_id.keys()))
    did = name_to_id[driver_name]
    grid_slot = col2.number_input("Pin to grid slot", 0, len(drivers), 0)
    weather = col3.selectbox("Weather", ["dry", "rain"])
    form_boost = st.slider("Recent-form boost (positive = faster)", -5.0, 5.0, 0.0, 0.5)
    car_boost = st.slider("Car performance Δ (%)", -10, 10, 0)

    sc = Scenario(name="ui", weather=weather)
    if form_boost:
        sc.form_boost[did] = -form_boost
    if grid_slot:
        sc.grid_override[did] = int(grid_slot)
    if car_boost:
        cid = race[race["driver_id"] == did]["constructor_id"].iloc[0]
        sc.car_perf_delta[cid] = car_boost / 100.0

    model = get_model()
    stand_dict = dict(zip(standings["driver_id"], standings["points"]))
    res = apply_scenario(model, race, sc, season=season, standings=stand_dict, cfg=get_config())

    st.markdown(f"**Scenario:** `{sc.describe()}`")
    delta = res["delta"]
    st.subheader("Impact vs Baseline")
    show = delta[["driver_name", "p_win", "p_win_delta", "p_podium_delta",
                  "exp_points", "exp_points_delta"]].copy()
    show.columns = ["Driver", "P(win)", "ΔP(win)", "ΔP(podium)", "xPts", "ΔxPts"]
    st.dataframe(
        show.style.format({"P(win)": "{:.1%}", "ΔP(win)": "{:+.1%}",
                           "ΔP(podium)": "{:+.1%}", "xPts": "{:.1f}", "ΔxPts": "{:+.1f}"})
        .background_gradient(subset=["ΔP(win)"], cmap="RdYlGn"),
        use_container_width=True, hide_index=True, height=480)

    if "championship_scenario" in res:
        st.subheader("Projected Championship Lead Probability")
        champ = res["championship_scenario"].head(8)
        fig = px.bar(champ, x="p_championship_lead", y="driver_name", orientation="h")
        fig.update_layout(PLOTLY_TEMPLATE["layout"], height=360,
                          yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    if st.button("💾 Save scenario for comparison"):
        st.session_state.setdefault("saved_scenarios", []).append(
            {"name": sc.describe(), "delta": show})
        st.success("Scenario saved.")
    saved = st.session_state.get("saved_scenarios", [])
    if saved:
        st.caption(f"{len(saved)} scenario(s) saved this session.")
