"""Cached data/model access for the dashboard (keeps interactions instant)."""
from __future__ import annotations

import streamlit as st

from ..config import load_config


@st.cache_resource(show_spinner=False)
def get_config():
    return load_config()


@st.cache_data(show_spinner="Loading master dataset…")
def get_master():
    from ..data_pipeline import load_master
    return load_master(get_config())


@st.cache_data(show_spinner="Engineering features…")
def get_features():
    from ..features import build_features
    df = get_master()
    feat_df, cols = build_features(df, get_config())
    return feat_df, cols


@st.cache_resource(show_spinner="Loading model…")
def get_model():
    from ..models import load_production
    return load_production(get_config())


@st.cache_data(show_spinner=False)
def get_races():
    from ..whatif import list_races
    feat_df, _ = get_features()
    return list_races(feat_df)
