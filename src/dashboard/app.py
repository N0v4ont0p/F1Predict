"""f1predict — extravagant Streamlit dashboard entry point.

Launch with ``f1predict dashboard`` (or ``streamlit run src/dashboard/app.py``).

The app is intentionally single-entry with sidebar routing (robust across Streamlit
versions) while page logic lives in :mod:`f1predict.dashboard.pages.*`.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow ``streamlit run src/dashboard/app.py`` to find the f1predict package.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import streamlit as st  # noqa: E402

try:  # works both as a package and as a loose script
    from f1predict.dashboard import theme
    from f1predict.dashboard.pages import (
        backtesting, championship_whatif, explainability, model_comparison,
        monte_carlo, race_predictor, reports, upcoming,
    )
except ModuleNotFoundError:  # pragma: no cover - fallback for script execution
    import importlib
    theme = importlib.import_module("dashboard.theme")
    backtesting = importlib.import_module("dashboard.pages.backtesting")
    championship_whatif = importlib.import_module("dashboard.pages.championship_whatif")
    explainability = importlib.import_module("dashboard.pages.explainability")
    model_comparison = importlib.import_module("dashboard.pages.model_comparison")
    monte_carlo = importlib.import_module("dashboard.pages.monte_carlo")
    race_predictor = importlib.import_module("dashboard.pages.race_predictor")
    reports = importlib.import_module("dashboard.pages.reports")
    upcoming = importlib.import_module("dashboard.pages.upcoming")

PAGES = {
    "Race Predictor": race_predictor.render,
    "Championship & What-If": championship_whatif.render,
    "Monte Carlo Simulator": monte_carlo.render,
    "Backtesting Explorer": backtesting.render,
    "Explainability": explainability.render,
    "Upcoming Race": upcoming.render,
    "Model Comparison": model_comparison.render,
    "Reports & Export": reports.render,
}


def main():
    st.set_page_config(page_title="f1predict", page_icon="🏎️", layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown(theme.CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(theme.header("F1 analytics suite"), unsafe_allow_html=True)
        st.markdown("---")
        default = st.session_state.pop("_goto", None)
        names = list(PAGES.keys())
        index = names.index(default) if default in names else 0
        page = st.radio("Navigate", names, index=index, label_visibility="collapsed")
        st.markdown("---")
        st.caption("Optimised for MacBook Air M5 · 24GB")
        st.caption("Synthetic demo data unless a real master is built.")

    PAGES[page]()


main()
