"""Tests for regulation intelligence, the regulation-native Monte-Carlo engine, and
calibration metrics added in the regulation-native upgrade."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1predict.features import era_modifiers, era_summary
from f1predict.evaluation import brier_score, top_n_accuracy, reliability_curve
from f1predict.simulation import simulate_race, empirical_dnf_rates


# --------------------------------------------------------------------------- regulations
def test_reset_year_is_more_chaotic_and_less_reliable():
    reset = era_modifiers(2026)        # 0 seasons since reset
    settled = era_modifiers(2019)      # deep into a stable rule-set
    assert reset["chaos"] > settled["chaos"]
    assert reset["reliability"] > settled["reliability"]


def test_era_summary_shape():
    es = era_summary(2026)
    assert es["era"] == "reg_2026"
    assert es["is_reset_year"] is True
    assert es["constructor_trust"] < 0.5          # low trust in a reset year
    assert len(es["changes"]) >= 3
    assert set(es["modifiers"]) >= {"chaos", "reliability", "overtake_difficulty"}


# --------------------------------------------------------------------------- simulation
def _race(n=6):
    return pd.DataFrame({
        "driver_id": [f"d{i}" for i in range(n)],
        "driver_name": [f"Driver {i}" for i in range(n)],
        "constructor_id": [f"c{i // 2}" for i in range(n)],
        "race_name": ["Test GP"] * n,
    })


def test_simulation_probabilities_are_valid():
    race = _race()
    mu = np.arange(len(race), dtype=float)  # d0 fastest
    sim = simulate_race(race, mu, sigma=2.0, n_sim=4000, season=2024, seed=1)
    f = sim.to_frame()
    assert np.isclose(f["p_win"].sum(), 1.0, atol=0.02)
    assert (f["p_win"] >= 0).all() and (f["p_win"] <= 1).all()
    assert "p_dnf" in f.columns and (f["p_dnf"] >= 0).all()
    # Fastest car should win most often.
    assert f.iloc[0]["driver_name"] == "Driver 0"


def test_higher_dnf_prob_increases_simulated_retirements():
    race = _race()
    mu = np.arange(len(race), dtype=float)
    low = simulate_race(race, mu, 2.0, n_sim=4000, season=2024,
                        dnf_prob=np.full(len(race), 0.02), safety_car=False, seed=2)
    high = simulate_race(race, mu, 2.0, n_sim=4000, season=2024,
                         dnf_prob=np.full(len(race), 0.40), safety_car=False, seed=2)
    assert high.p_dnf.mean() > low.p_dnf.mean()


def test_reset_year_simulation_more_volatile_than_settled():
    race = _race()
    mu = np.arange(len(race), dtype=float)
    settled = simulate_race(race, mu, 2.0, n_sim=6000, season=2019, seed=3)
    reset = simulate_race(race, mu, 2.0, n_sim=6000, season=2026, seed=3)
    # More chaos => the favourite wins less often in the reset year.
    assert reset.p_win.max() <= settled.p_win.max() + 1e-9


def test_empirical_dnf_rates_within_bounds(small_df):
    season = int(small_df["season"].max())
    race = small_df[small_df["season"] == season]
    race = race[race["round"] == race["round"].min()].reset_index(drop=True)
    rates = empirical_dnf_rates(race, small_df, season)
    assert len(rates) == len(race)
    assert (rates >= 0.01).all() and (rates <= 0.55).all()


# --------------------------------------------------------------------------- calibration
def test_brier_score_bounds():
    assert brier_score([1.0, 0.0], [1.0, 0.0]) == 0.0          # perfect
    assert brier_score([0.0, 1.0], [1.0, 0.0]) == 1.0          # worst
    assert 0.0 < brier_score([0.5, 0.5], [1.0, 0.0]) < 1.0


def test_top_n_accuracy():
    pred = np.array([1, 2, 3, 4, 5])
    actual = np.array([1, 2, 3, 5, 4])
    assert top_n_accuracy(pred, actual, n=3) == 1.0            # same top 3
    assert top_n_accuracy(pred, np.array([5, 4, 3, 2, 1]), n=2) == 0.0


def test_reliability_curve_shapes():
    rng = np.random.default_rng(0)
    p = rng.random(500)
    y = (rng.random(500) < p).astype(float)
    centres, obs, counts = reliability_curve(p, y, bins=10)
    assert len(centres) == len(obs) == len(counts) == 10
    assert counts.sum() == 500
