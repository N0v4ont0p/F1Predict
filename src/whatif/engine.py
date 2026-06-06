"""What-if analysis engine — shared by the CLI and the dashboard.

A :class:`Scenario` captures a set of perturbations to a race (grid changes, weather, form
boosts, car-performance deltas). :func:`apply_scenario` re-runs prediction + Monte Carlo
under the scenario and reports the **delta** versus the baseline so users can instantly see
the impact on positions, points and the championship.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config, load_config
from ..models.predictor import RacePredictor
from ..simulation.engine import championship_swing, simulate_race


@dataclass
class Scenario:
    name: str = "scenario"
    weather: str = "dry"                                   # dry | rain
    grid_override: dict[str, int] = field(default_factory=dict)     # driver_id -> grid slot
    form_boost: dict[str, float] = field(default_factory=dict)      # driver_id -> mu delta
    car_perf_delta: dict[str, float] = field(default_factory=dict)  # constructor_id -> pct (+faster)
    n_sim: int = 10000

    def describe(self) -> str:
        bits = [f"weather={self.weather}"]
        if self.grid_override:
            bits.append("grid=" + ",".join(f"{k}:P{v}" for k, v in self.grid_override.items()))
        if self.form_boost:
            bits.append("form=" + ",".join(f"{k}:{v:+.2f}" for k, v in self.form_boost.items()))
        if self.car_perf_delta:
            bits.append("car=" + ",".join(f"{k}:{v:+.1%}" for k, v in self.car_perf_delta.items()))
        return " | ".join(bits)


def _adjusted_mu(predictor: RacePredictor, race_df: pd.DataFrame,
                 scenario: Scenario) -> np.ndarray:
    """Predicted positions with car-performance deltas folded into the grid feature."""
    race = race_df.copy()
    if scenario.car_perf_delta and "f_grid" in race:
        for cid, pct in scenario.car_perf_delta.items():
            mask = race["constructor_id"] == cid
            # A faster car (+pct) effectively improves grid/pace -> lower position number.
            race.loc[mask, "f_grid"] = race.loc[mask, "f_grid"] * (1 - pct)
    return predictor.predict_positions(race)


def apply_scenario(
    predictor: RacePredictor,
    race_df: pd.DataFrame,
    scenario: Scenario,
    season: int = 2025,
    standings: dict[str, float] | None = None,
    cfg: Config | None = None,
) -> dict[str, Any]:
    """Run baseline + scenario simulations and return tables plus deltas."""
    cfg = cfg or load_config()
    sigma = predictor.residual_std

    base_mu = predictor.predict_positions(race_df)
    base_sim = simulate_race(race_df, base_mu, sigma, n_sim=scenario.n_sim,
                             season=season, weather="dry", cfg=cfg)

    scen_mu = _adjusted_mu(predictor, race_df, scenario)
    scen_sim = simulate_race(
        race_df, scen_mu, sigma, n_sim=scenario.n_sim, season=season,
        weather=scenario.weather, form_boost=scenario.form_boost or None,
        grid_override=scenario.grid_override or None, cfg=cfg,
    )

    base_tbl = base_sim.to_frame().set_index("driver_id")
    scen_tbl = scen_sim.to_frame().set_index("driver_id")
    delta = pd.DataFrame({
        "driver_name": scen_tbl["driver_name"],
        "p_win": scen_tbl["p_win"],
        "p_win_delta": scen_tbl["p_win"] - base_tbl["p_win"],
        "p_podium": scen_tbl["p_podium"],
        "p_podium_delta": scen_tbl["p_podium"] - base_tbl["p_podium"],
        "exp_points": scen_tbl["exp_points"],
        "exp_points_delta": scen_tbl["exp_points"] - base_tbl["exp_points"],
        "mean_finish_delta": scen_tbl["mean_finish"] - base_tbl["mean_finish"],
    }).sort_values("p_win", ascending=False)

    result = {
        "scenario": scenario,
        "baseline": base_sim,
        "scenario_sim": scen_sim,
        "delta": delta.reset_index(),
    }
    if standings is not None:
        result["championship_baseline"] = championship_swing(base_sim, standings)
        result["championship_scenario"] = championship_swing(scen_sim, standings)
    return result
