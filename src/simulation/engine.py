"""Monte Carlo race & championship simulation engine.

Vectorised NumPy implementation capable of 10k–50k simulations in well under a second on
the M5. Given a model's expected finishing positions (``mu``) and a residual spread
(``sigma``), we sample noisy race orders, derive finishing positions, and accumulate
distributions for winner / podium / points / championship impact.

Supports **conditioning**: pin a driver to a grid slot, apply a global wet-weather order
shuffle (increases variance), or boost/penalise specific drivers' form.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config, load_config
from ..features.regulations import era_modifiers as _era_modifiers
from ..utils.points import points_for_position


@dataclass
class SimulationResult:
    drivers: list[str]
    driver_names: list[str]
    n_sim: int
    finish_positions: np.ndarray          # (n_sim, n_drivers) integer positions
    p_win: np.ndarray
    p_podium: np.ndarray
    p_points: np.ndarray
    exp_points: np.ndarray
    points_samples: np.ndarray            # (n_sim, n_drivers) points per simulation
    p_dnf: np.ndarray | None = None       # per-driver simulated retirement probability
    meta: dict[str, Any] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        out = pd.DataFrame({
            "driver_id": self.drivers,
            "driver_name": self.driver_names,
            "p_win": self.p_win,
            "p_podium": self.p_podium,
            "p_points": self.p_points,
            "exp_points": self.exp_points,
            "p_dnf": self.p_dnf if self.p_dnf is not None else np.zeros(len(self.drivers)),
            "mean_finish": self.finish_positions.mean(axis=0),
        })
        return out.sort_values("p_win", ascending=False).reset_index(drop=True)


def simulate_race(
    race_df: pd.DataFrame,
    mu: np.ndarray,
    sigma: float,
    n_sim: int = 10000,
    season: int = 2025,
    weather: str = "dry",
    form_boost: dict[str, float] | None = None,
    grid_override: dict[str, int] | None = None,
    dnf_prob: np.ndarray | None = None,
    reg_modifiers: dict[str, float] | None = None,
    safety_car: bool = True,
    seed: int = 7,
    cfg: Config | None = None,
) -> SimulationResult:
    """Run a **regulation-native** Monte Carlo simulation for a single race.

    ``mu`` is the model's predicted finishing position per driver (lower == better). On top
    of the base pace draw we layer the dynamics that actually decide F1 races:

    * **Regulation chaos** — variance is scaled by the season's era modifiers
      (:func:`~features.regulations.era_modifiers`), so a 2026-reset race is far more
      volatile than a settled 2019 race.
    * **Overtaking difficulty** — in eras where passing is hard, the starting order sticks:
      we shrink order variance, making grid/pace more decisive.
    * **Reliability / DNFs** — each driver can retire per simulation (``dnf_prob`` or an
      era-reliability default). Retirements are classified at the back and score no points.
    * **Safety cars** — with an era-dependent probability a race is "bunched", compressing
      pace gaps and amplifying late upsets.

    ``weather='rain'`` inflates variance further. ``form_boost`` maps driver_id -> delta on
    ``mu`` (negative == faster). ``grid_override`` nudges ``mu`` toward a pinned grid slot.
    All new behaviour is opt-out: defaults reproduce sensible, era-aware racing.
    """
    cfg = cfg or load_config()
    rng = np.random.default_rng(seed)
    n = len(mu)
    drivers = race_df["driver_id"].tolist()
    names = race_df["driver_name"].astype(str).tolist()

    mods = reg_modifiers if reg_modifiers is not None else _era_modifiers(season)

    # Track & weather context (from the race frame's features when available). Degradation
    # and rainfall add genuine, physically-motivated variance on top of the era baseline.
    track_deg = _col_mean(race_df, "trk_degradation", 0.5)
    tyre_stress = _col_mean(race_df, "f_tyre_stress", track_deg * 0.5)
    rain_prob = _col_mean(race_df, "wx_rain_prob", 0.0)
    if weather.lower() in ("rain", "wet"):
        rain_prob = max(rain_prob, 0.85)

    mu = np.asarray(mu, dtype=float).copy()
    if form_boost:
        for did, delta in form_boost.items():
            if did in drivers:
                mu[drivers.index(did)] += delta
    if grid_override:
        for did, slot in grid_override.items():
            if did in drivers:
                mu[drivers.index(did)] = 0.5 * mu[drivers.index(did)] + 0.5 * slot

    # --- effective pace spread, modulated by regulations, track & weather ------------
    noise_scale = float(cfg.get("simulation.noise_scale", 0.9))
    sigma_eff = max(sigma, 0.5) * noise_scale
    sigma_eff *= float(mods.get("chaos", 1.0)) * float(mods.get("tire_wear_var", 1.0)) ** 0.5
    # Hard-to-overtake eras compress position changes (grid sticks).
    sigma_eff *= 1.0 - 0.30 * float(mods.get("overtake_difficulty", 0.5))
    # High-degradation / high tyre-stress races punish management errors → more spread.
    sigma_eff *= 1.0 + 0.25 * tyre_stress
    # Rain is the great equaliser: variance scales smoothly with rain probability.
    sigma_eff *= 1.0 + 0.9 * rain_prob
    if weather.lower() in ("rain", "wet"):
        sigma_eff *= 1.2  # an explicit wet call adds a little extra chaos on top

    draws = rng.normal(mu[None, :], sigma_eff, size=(n_sim, n))

    # --- safety-car bunching: compress pace gaps in a fraction of races --------------
    if safety_car:
        # Rain and high-degradation races see more safety cars.
        sc_rate = float(mods.get("safety_car_rate", 0.4))
        sc_rate = min(0.85, sc_rate * (1.0 + 0.6 * rain_prob + 0.3 * tyre_stress))
        sc_mask = rng.random(n_sim) < sc_rate
        if sc_mask.any():
            row_mean = draws[sc_mask].mean(axis=1, keepdims=True)
            draws[sc_mask] = row_mean + (draws[sc_mask] - row_mean) * 0.6

    # --- reliability: sample retirements, classified at the back ---------------------
    if dnf_prob is None:
        base = float(cfg.get("simulation.base_dnf", 0.07))
        dnf_prob = np.full(n, base) * float(mods.get("reliability", 1.0))
    dnf_prob = np.clip(np.asarray(dnf_prob, dtype=float), 0.0, 0.85)
    # Wet races cause more incidents/retirements across the board.
    dnf_prob = np.clip(dnf_prob * (1.0 + 0.5 * rain_prob), 0.0, 0.9)
    dnf_mask = rng.random((n_sim, n)) < dnf_prob[None, :]
    # Push retirements far behind any finisher (still a valid permutation per sim).
    draws = np.where(dnf_mask, draws + 1e6, draws)

    order = np.argsort(draws, axis=1)
    finish = np.empty_like(order)
    rows = np.arange(n_sim)[:, None]
    finish[rows, order] = np.arange(n)[None, :]
    finish += 1  # 1-based

    # Points per simulation; retirements (large position) score nothing.
    pts_table = np.array([points_for_position(p, season) for p in range(0, n + 2)])
    points_samples = pts_table[np.clip(finish, 0, n + 1)]
    points_samples = np.where(dnf_mask, 0.0, points_samples)

    return SimulationResult(
        drivers=drivers, driver_names=names, n_sim=n_sim,
        finish_positions=finish,
        p_win=(finish == 1).mean(axis=0),
        p_podium=(finish <= 3).mean(axis=0),
        p_points=((finish <= 10) & ~dnf_mask).mean(axis=0),
        exp_points=points_samples.mean(axis=0),
        points_samples=points_samples,
        p_dnf=dnf_mask.mean(axis=0),
        meta={"weather": weather, "n_sim": n_sim, "sigma": sigma_eff, "season": season,
              "rain_prob": round(float(rain_prob), 3),
              "tyre_stress": round(float(tyre_stress), 3),
              "track_degradation": round(float(track_deg), 3),
              "modifiers": {k: round(float(v), 3) for k, v in mods.items()}},
    )


def _col_mean(df: pd.DataFrame, col: str, default: float) -> float:
    """Mean of a feature column if present & numeric, else a default (sim is robust to
    race frames that predate the weather/track-stats features)."""
    if col in df.columns:
        try:
            v = float(pd.to_numeric(df[col], errors="coerce").mean())
            if v == v:  # not NaN
                return v
        except (TypeError, ValueError):
            pass
    return default


def championship_swing(sim: SimulationResult, current_standings: dict[str, float]) -> pd.DataFrame:
    """Quantify how much this race swings the title fight.

    Returns each driver's projected post-race points distribution (min/mean/max) and the
    probability they hold/seize the championship lead after the race.
    """
    base = np.array([current_standings.get(d, 0.0) for d in sim.drivers])
    projected = base[None, :] + sim.points_samples           # (n_sim, n_drivers)
    leader_each_sim = projected.argmax(axis=1)
    p_lead = np.bincount(leader_each_sim, minlength=len(sim.drivers)) / sim.n_sim
    return pd.DataFrame({
        "driver_id": sim.drivers,
        "driver_name": sim.driver_names,
        "current_points": base,
        "proj_mean": projected.mean(axis=0),
        "proj_min": projected.min(axis=0),
        "proj_max": projected.max(axis=0),
        "p_championship_lead": p_lead,
    }).sort_values("proj_mean", ascending=False).reset_index(drop=True)


def empirical_dnf_rates(
    race_df: pd.DataFrame,
    master: pd.DataFrame,
    season: int,
    lookback: int = 3,
    cfg: Config | None = None,
) -> np.ndarray:
    """Estimate each driver's retirement probability for an upcoming race.

    Blends the driver's own recent DNF frequency with their constructor's recent record
    (constructor reliability dominates for rookies / cold-start drivers), then scales by the
    season's regulation reliability modifier so reset years are correctly more fragile.
    Returns an array aligned to ``race_df`` rows.
    """
    cfg = cfg or load_config()
    base = float(cfg.get("simulation.base_dnf", 0.07))
    hist = master[(master["season"] >= season - lookback) & (master["season"] <= season)]
    hist = hist[hist["position"].notna()]

    drv_rate = hist.groupby("driver_id")["dnf"].mean() if "dnf" in hist else {}
    con_rate = hist.groupby("constructor_id")["dnf"].mean() if "dnf" in hist else {}
    rel = float(_era_modifiers(season).get("reliability", 1.0))

    out = []
    for _, row in race_df.iterrows():
        d = drv_rate.get(row["driver_id"], np.nan) if hasattr(drv_rate, "get") else np.nan
        c = con_rate.get(row["constructor_id"], np.nan) if hasattr(con_rate, "get") else np.nan
        parts = [v for v in (d, c) if v == v]  # drop NaNs
        rate = float(np.mean(parts)) if parts else base
        # Shrink toward the global base for stability. The empirical rate already reflects a
        # team's reliability, so we only apply a *softened* regulation multiplier (sqrt) to
        # avoid double-counting, and cap at a realistic ceiling.
        rate = (0.6 * rate + 0.4 * base) * rel ** 0.5
        out.append(rate)
    return np.clip(np.asarray(out, dtype=float), 0.01, 0.55)
