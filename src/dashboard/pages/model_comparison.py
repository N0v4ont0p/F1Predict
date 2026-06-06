"""Page: Model Comparison & Experiment Leaderboard."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from ..data import get_config
from ..theme import PLOTLY_TEMPLATE


def render():
    st.header("⚙️ Model Comparison & Experiments")
    from ...models import load_experiments

    exps = load_experiments(get_config())
    if not exps:
        st.info("No experiments yet. Run `f1predict train --compare-all` first.")
        return

    rows = []
    for e in exps:
        m = e.get("metrics", {})
        rows.append({
            "time": e.get("timestamp", ""), "name": e.get("experiment_name", ""),
            "family": e.get("family", ""), "mae": m.get("mae"),
            "winner_acc": m.get("winner_accuracy"), "podium_acc": m.get("podium_accuracy"),
            "commit": e.get("git_commit", ""),
        })
    df = pd.DataFrame(rows).sort_values("mae")

    st.subheader("Experiment Leaderboard")
    st.dataframe(df.style.format({"mae": "{:.3f}", "winner_acc": "{:.1%}",
                                  "podium_acc": "{:.1%}"}),
                 use_container_width=True, hide_index=True)

    # Latest benchmark (per-family) if present.
    latest = next((e for e in reversed(exps) if e.get("benchmark")), None)
    if latest:
        st.subheader("Latest Benchmark — Families Compared")
        bench = pd.DataFrame(latest["benchmark"])
        cols = [c for c in ["family", "mae", "winner_accuracy", "podium_accuracy",
                            "peak_rss_mb", "train_seconds"] if c in bench]
        st.dataframe(bench[cols], use_container_width=True, hide_index=True)
        if "peak_rss_mb" in bench:
            fig = px.scatter(bench, x="train_seconds", y="mae", size="peak_rss_mb",
                             color="family", text="family",
                             labels={"train_seconds": "Train time (s)", "mae": "MAE"})
            fig.update_layout(PLOTLY_TEMPLATE["layout"], height=420)
            st.plotly_chart(fig, use_container_width=True)

    st.caption("Best MAE highlighted first. Memory & time reported for the 24GB M5 budget.")
