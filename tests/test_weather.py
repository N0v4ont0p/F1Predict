"""Weather & track-environment tests — all offline (climatology fallback, no network)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1predict.config import load_config
from f1predict.data_pipeline import get_race_weather, enrich_weather, WEATHER_COLUMNS
from f1predict.features.tracks import track_env, track_stats, track_climate
from f1predict.features import build_features
from f1predict.simulation import simulate_race


@pytest.fixture()
def offline_cfg(tmp_path):
    c = load_config()
    c.set("paths.cache_dir", str(tmp_path / "cache"))
    return c


def test_track_stats_bounds():
    for circuit in ("suzuka", "monza", "monaco", "spa"):
        st = track_stats(circuit)
        for key in ("trk_degradation", "trk_abrasion", "trk_downforce"):
            assert 0.0 <= st[key] <= 1.0, f"{circuit}.{key} out of [0,1]"
        assert st["trk_lap_km"] > 0
        env = track_env(circuit)
        assert -90 <= env["lat"] <= 90
        assert -180 <= env["lon"] <= 180


def test_track_stats_unknown_circuit_defaults():
    st = track_stats("not_a_real_circuit_xyz")
    assert set(st) >= {"trk_degradation", "trk_abrasion", "trk_lap_km"}
    for v in st.values():
        assert v == v  # no NaN


def test_climatology_always_available():
    clim = track_climate("suzuka")
    assert clim["wx_temp_c"] is not None
    assert 0.0 <= clim["wx_rain_prob"] <= 1.0
    assert 0.0 <= clim["wx_humidity"] <= 1.0


def test_get_race_weather_offline_uses_climatology(offline_cfg):
    wx = get_race_weather("suzuka", pd.Timestamp("2026-04-12"), offline_cfg, online=False)
    assert set(WEATHER_COLUMNS) <= set(wx)
    assert wx["wx_source"] == "climatology"
    assert wx["wx_temp_c"] is not None
    assert 0.0 <= wx["wx_rain_prob"] <= 1.0


def test_get_race_weather_never_raises_on_unknown(offline_cfg):
    wx = get_race_weather("totally_unknown", None, offline_cfg, online=False)
    assert "wx_temp_c" in wx


def test_enrich_weather_offline(offline_cfg):
    df = pd.DataFrame({
        "circuit_id": ["suzuka", "monza", "suzuka"],
        "date": pd.to_datetime(["2024-04-07", "2024-09-01", "2025-04-06"]),
    })
    out = enrich_weather(df, offline_cfg, online=False)
    assert len(out) == len(df)
    for col in WEATHER_COLUMNS:
        assert col in out.columns
    assert out["wx_temp_c"].notna().all()


def test_weather_features_present_in_factory(small_df):
    feats, _cols = build_features(small_df)
    for col in ("wx_temp_norm", "wx_humidity", "wx_rain_prob", "f_tyre_stress",
                "trk_degradation", "trk_abrasion"):
        assert col in feats.columns, f"missing weather/track feature {col}"
    assert feats["f_tyre_stress"].notna().all()


def test_simulation_rain_increases_variance():
    race = pd.DataFrame({
        "driver_id": [f"d{i}" for i in range(8)],
        "driver_name": [f"D{i}" for i in range(8)],
        "constructor_id": [f"c{i // 2}" for i in range(8)],
        "race_name": ["GP"] * 8,
        "trk_degradation": [0.6] * 8,
        "f_tyre_stress": [0.4] * 8,
        "wx_rain_prob": [0.1] * 8,
    })
    mu = np.arange(8.0)
    dry = simulate_race(race, mu, 2.0, n_sim=4000, season=2024, seed=7)
    wet = simulate_race(race, mu, 2.0, n_sim=4000, season=2024, weather="rain", seed=7)
    assert wet.meta["sigma"] > dry.meta["sigma"]
    assert wet.meta["rain_prob"] > dry.meta["rain_prob"]
    # rain compresses the favourite's dominance
    assert wet.p_win.max() < dry.p_win.max()
