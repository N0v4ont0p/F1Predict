"""Page: Upcoming Race Intelligence & Live Form."""
from __future__ import annotations

import streamlit as st

from ..data import get_features, get_master, get_model
from ..widgets import race_selector


def render():
    st.header("📡 Upcoming Race Intelligence")
    from ...whatif import upcoming_race

    feat_df, _ = get_features()
    season, rnd = upcoming_race(feat_df)
    st.info(f"Next up (demo): **{season} Round {rnd}**. Use the selector to inspect any race.")
    season, rnd, race = race_selector("up")
    if race.empty:
        return

    model = get_model()
    tbl = model.predict_race(race, season=season)

    st.subheader("Key Storylines")
    fav = tbl.iloc[0]
    mover = tbl.assign(g=tbl["grid"] - tbl["predicted_rank"]).sort_values("g", ascending=False).iloc[0]
    faller = tbl.assign(g=tbl["grid"] - tbl["predicted_rank"]).sort_values("g").iloc[0]
    st.markdown(f"- 🏆 **Favourite:** {fav['driver_name']} ({fav['p_win']:.0%} win, "
                f"{fav['p_podium']:.0%} podium)")
    st.markdown(f"- 📈 **Biggest predicted gainer:** {mover['driver_name']} "
                f"(P{int(mover['grid'])} → P{int(mover['predicted_rank'])})")
    st.markdown(f"- 📉 **At risk of dropping:** {faller['driver_name']} "
                f"(P{int(faller['grid'])} → P{int(faller['predicted_rank'])})")
    closest = tbl.head(3)
    st.markdown(f"- ⚔️ **Podium fight:** {', '.join(closest['driver_name'].tolist())}")

    st.subheader("Predicted vs Grid")
    comp = tbl[["driver_name", "grid", "predicted_rank", "p_win", "p_podium"]].copy()
    comp.columns = ["Driver", "Grid", "Predicted", "P(win)", "P(podium)"]
    st.dataframe(comp.style.format({"P(win)": "{:.1%}", "P(podium)": "{:.1%}"}),
                 use_container_width=True, hide_index=True, height=500)
