"""Page: Explainability & Feature Intelligence (model-native importances)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from ..data import get_features, get_model
from ..theme import PLOTLY_TEMPLATE
from ..widgets import race_selector


def _importances(model_wrapper) -> pd.DataFrame:
    est = model_wrapper.model
    cols = model_wrapper.feature_cols
    if hasattr(est, "feature_importances_"):
        imp = est.feature_importances_
    else:  # HistGB has no direct attribute -> uniform fallback
        imp = np.ones(len(cols)) / len(cols)
    return pd.DataFrame({"feature": cols, "importance": imp}).sort_values(
        "importance", ascending=False)


def render():
    st.header("🔍 Explainability & Feature Intelligence")
    model = get_model()

    st.subheader("Global Feature Importance")
    imp = _importances(model).head(20)
    fig = px.bar(imp, x="importance", y="feature", orientation="h")
    fig.update_layout(PLOTLY_TEMPLATE["layout"], height=520,
                      yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Why did the model predict this? (per race)")
    season, rnd, race = race_selector("xai")
    if race.empty:
        return
    tbl = model.predict_race(race, season=season)
    driver = st.selectbox("Driver", tbl["driver_name"].tolist())
    row = race[race["driver_name"] == driver]
    if row.empty:
        return
    cols = model.feature_cols
    contrib = pd.DataFrame({
        "feature": cols,
        "value": row[cols].iloc[0].to_numpy(),
        "importance": _importances(model).set_index("feature").loc[cols, "importance"].to_numpy(),
    })
    contrib["weighted"] = contrib["value"] * contrib["importance"]
    contrib = contrib.reindex(contrib["weighted"].abs().sort_values(ascending=False).index).head(12)
    pred_pos = float(tbl[tbl["driver_name"] == driver]["pred_position"].iloc[0])
    st.metric(f"Predicted position for {driver}", f"{pred_pos:.1f}")
    fig2 = px.bar(contrib, x="weighted", y="feature", orientation="h",
                  labels={"weighted": "value × importance"})
    fig2.update_layout(PLOTLY_TEMPLATE["layout"], height=420,
                       yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Tip: install the `explain` extra (`pip install -e .[explain]`) for full SHAP plots.")
