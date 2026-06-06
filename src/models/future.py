"""Future-race prediction — predict an upcoming Grand Prix that has **no results yet**.

For a future round (e.g. a later 2026 race, or any 2027+ event) we have no grid, no
qualifying and no result. We synthesise a realistic entry list and feature row per driver:

1. **Lineup** = the driver/constructor pairings from the most recent completed race (the
   current grid). This automatically reflects mid-season changes already in the data.
2. **Expected grid** = drivers ordered by their *latest* combined ELO blended with recent
   season form. Qualifying is unknown, so the model's strongest causal priors (skill + car)
   stand in for it — exactly what you'd do before a race weekend.
3. The placeholder race is appended to the master, the **temporal-safe feature factory** runs,
   and the engineered row for the future race carries every driver's up-to-date ELO, momentum,
   reliability, track affinity and the **2026 regulation-reset down-weighting**. We then ask
   the production model for a ranked prediction with win/podium/points probabilities.

This shares the exact feature + model code path used for historical races, so future and
backtest predictions are always consistent.
"""
from __future__ import annotations

import pandas as pd

from ..config import Config, load_config
from ..features import build_features
from ..features.elo import latest_ratings
from ..utils.logging import get_logger

log = get_logger()


def _current_lineup(master: pd.DataFrame) -> pd.DataFrame:
    """Most recent race's classified entrants = the current driver/constructor grid."""
    last_season = int(master["season"].max())
    sub = master[master["season"] == last_season]
    last_round = int(sub["round"].max())
    grid = sub[sub["round"] == last_round][
        ["driver_id", "driver_code", "driver_name", "constructor_id", "constructor_name"]
    ].drop_duplicates("driver_id").reset_index(drop=True)
    return grid


def build_future_race(
    cfg: Config,
    season: int,
    rnd: int | None,
    circuit_id: str,
    master: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, int]:
    """Return ``(master_plus_future, round)`` with a synthetic future race appended.

    The synthetic weekend's ``has_sprint`` flag is seeded from the sprint calendar so that
    sprint-aware features (and sprint prediction) behave correctly for upcoming events.
    """
    if master is None:
        from ..data_pipeline import load_master
        master = load_master(cfg)

    # Resolve a usable round number for grouping/ordering.
    if rnd is None:
        same = master[(master["season"] == season) & (master["circuit_id"] == circuit_id)]
        if len(same):
            rnd = int(same["round"].iloc[0])
        else:
            rnd = int(master[master["season"] == season]["round"].max() or 0) + 1 \
                if (master["season"] == season).any() else 99

    # If the race already has real results, just return the master unchanged.
    existing = master[(master["season"] == season) & (master["round"] == rnd)]
    if len(existing) and (existing["position"] > 0).any():
        return master, rnd

    lineup = _current_lineup(master)
    ratings = latest_ratings(master)
    drv_r = ratings["driver"]
    con_r = ratings["constructor"]

    # Expected grid: combined skill (ELO) ordering — stands in for qualifying.
    lineup["_score"] = lineup.apply(
        lambda r: 0.65 * drv_r.get(r["driver_id"], 1500.0)
        + 0.35 * con_r.get(r["constructor_id"], 1500.0),
        axis=1,
    )
    lineup = lineup.sort_values("_score", ascending=False).reset_index(drop=True)
    lineup["grid"] = lineup.index + 1

    # Does this upcoming weekend run a sprint? (data-driven: schedule, else fallback)
    from ..data_pipeline import weekend_has_sprint
    has_sprint = 1 if weekend_has_sprint(season, rnd, circuit_id, cfg, master) else 0

    meta = _circuit_meta(master, season, rnd, circuit_id)
    rows = []
    for _, r in lineup.iterrows():
        rows.append({
            "season": season, "round": rnd, "date": meta["date"],
            "race_name": meta["race_name"], "circuit_id": circuit_id,
            "country": meta["country"], "locality": meta["locality"],
            "driver_id": r["driver_id"], "driver_code": r["driver_code"],
            "driver_name": r["driver_name"], "constructor_id": r["constructor_id"],
            "constructor_name": r["constructor_name"],
            "quali_position": int(r["grid"]), "quali_best_ms": 0.0,
            "grid": int(r["grid"]), "position": 0, "status": "Scheduled",
            "points": 0.0, "laps": 0, "finished": 1, "dnf": 0,
            "sprint_position": 0, "sprint_points": 0.0, "has_sprint": has_sprint,
        })
    future = pd.DataFrame(rows)
    combined = pd.concat([master, future], ignore_index=True)
    return combined, rnd


def _circuit_meta(master: pd.DataFrame, season: int, rnd: int, circuit_id: str) -> dict:
    hist = master[master["circuit_id"] == circuit_id]
    race_name = str(hist["race_name"].iloc[-1]) if len(hist) else circuit_id
    country = str(hist["country"].iloc[-1]) if len(hist) else ""
    locality = str(hist["locality"].iloc[-1]) if len(hist) else ""
    return {"race_name": race_name, "country": country, "locality": locality,
            "date": f"{season}-01-01"}


def predict_future_race(
    cfg: Config | None,
    predictor,
    season: int,
    rnd: int | None,
    circuit_id: str,
    master: pd.DataFrame | None = None,
    session: str = "race",
) -> pd.DataFrame:
    """Build the future race row set and return the model's ranked prediction table.

    ``session`` (``race``/``qualifying``/``sprint``) selects how the result is scored. The
    supplied ``predictor`` should already be the model trained for that session.
    """
    cfg = cfg or load_config()
    combined, rnd = build_future_race(cfg, season, rnd, circuit_id, master)
    feat_df, _ = build_features(combined, cfg)
    race = feat_df[(feat_df["season"] == season) & (feat_df["round"] == rnd)].copy()
    if race.empty:
        raise ValueError(f"could not assemble future race for {circuit_id} {season}")
    return predictor.predict_race(race.reset_index(drop=True), season=season, session=session)
