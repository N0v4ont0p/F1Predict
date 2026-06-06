"""Page: Reports & Export Center."""
from __future__ import annotations

import streamlit as st

from ..data import get_config
from ..widgets import race_selector


def render():
    st.header("📄 Reports & Export Center")
    season, rnd, race = race_selector("rep")
    if race.empty:
        return
    n_sim = st.slider("Monte Carlo simulations for report", 1000, 30000, 10000, step=1000)

    if st.button("📝 Generate HTML race report"):
        from ...reporting import build_race_report
        with st.spinner("Building report…"):
            path = build_race_report(get_config(), season, rnd,
                                     f"reports/race_{season}_{rnd}.html", n_sim=n_sim)
        html = path.read_text()
        st.success(f"Report generated: {path}")
        st.download_button("⬇ Download HTML", html, file_name=path.name, mime="text/html")
        with st.expander("Preview"):
            st.components.v1.html(html, height=600, scrolling=True)

    from ..data import get_model
    model = get_model()
    tbl = model.predict_race(race, season=season)
    st.subheader("Quick Export")
    st.download_button("⬇ Predictions (CSV)", tbl.to_csv(index=False),
                       file_name=f"pred_{season}_{rnd}.csv")
