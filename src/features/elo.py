"""Temporal-safe ELO rating engine for drivers & constructors.

ELO is the single most powerful prior for race prediction: it distils a competitor's entire
result history into one number that updates after every race. Crucially it is **causal** —
the rating used as a feature for race *N* is the rating *before* race *N* is scored, so there
is zero leakage.

Design choices tuned for F1:

* **Pairwise within-race updates.** Every classified finisher is compared to every other; the
  one who finished ahead "beats" the other. This is far more robust than comparing to a field
  average and naturally handles variable grid sizes across eras.
* **Separate driver and constructor ladders.** Driver skill and car performance are different
  signals; combined they explain most of the grid order.
* **Regulation-reset regression.** At the start of a new rule-set (2014/2017/2022/2026) every
  *constructor* rating is partially regressed to the mean — a dominant car often isn't dominant
  after a reset. Driver ratings carry over (skill persists across rule changes). This is what
  lets the model treat 2026 as a genuine reset instead of assuming 2025's order.
* **K-factor decay with experience.** New entrants move fast (high K) and settle as they race
  more, mirroring how quickly we learn a rookie's true level.
* **DNF handling.** A mechanical DNF is a weak signal about driver skill, so driver pairs where
  one retired non-classified are down-weighted; constructor ratings still take the reliability
  hit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .regulations import MAJOR_RESETS

BASE_RATING = 1500.0


def _expected(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))


def compute_elo(
    df: pd.DataFrame,
    driver_k: float = 32.0,
    constructor_k: float = 24.0,
    reset_regression: float = 0.35,
) -> pd.DataFrame:
    """Return ``df`` with pre-race ELO feature columns attached (temporal-safe).

    Adds: ``elo_driver``, ``elo_constructor``, ``elo_combined`` (pre-race ratings),
    ``elo_driver_grid_gap`` (rating relative to the race's field), and
    ``elo_constructor_field_gap``.

    ``df`` must be sorted chronologically by (season, round) and contain one row per driver
    entry with ``driver_id``, ``constructor_id``, ``position`` (0 == DNF), ``finished``.
    """
    driver_r: dict[str, float] = {}
    cons_r: dict[str, float] = {}
    driver_n: dict[str, int] = {}

    out_d = np.empty(len(df), dtype="float64")
    out_c = np.empty(len(df), dtype="float64")
    out_dgap = np.empty(len(df), dtype="float64")
    out_cgap = np.empty(len(df), dtype="float64")

    df = df.reset_index(drop=True)
    last_reset_applied = -1
    row_pos = {idx: i for i, idx in enumerate(df.index)}

    for (season, rnd), race in df.groupby(["season", "round"], sort=False):
        season = int(season)
        # --- regulation reset: regress constructor ratings toward the mean -------------
        if season in MAJOR_RESETS and season != last_reset_applied:
            if cons_r:
                mean_c = float(np.mean(list(cons_r.values())))
                for cid in cons_r:
                    cons_r[cid] += (mean_c - cons_r[cid]) * reset_regression
            last_reset_applied = season

        idxs = list(race.index)
        drivers = race["driver_id"].tolist()
        cons = race["constructor_id"].tolist()
        positions = race["position"].tolist()
        finished = race["finished"].tolist()

        # Pre-race ratings become the features for this race.
        cur_d = np.array([driver_r.get(d, BASE_RATING) for d in drivers])
        cur_c = np.array([cons_r.get(c, BASE_RATING) for c in cons])
        field_d = cur_d.mean()
        field_c = cur_c.mean()
        for k, idx in enumerate(idxs):
            i = row_pos[idx]
            out_d[i] = cur_d[k]
            out_c[i] = cur_c[k]
            out_dgap[i] = cur_d[k] - field_d
            out_cgap[i] = cur_c[k] - field_c

        # --- pairwise updates (after recording features) ------------------------------
        n = len(drivers)
        # Effective finishing rank: classified by position, DNFs ranked last by laps order.
        rank = np.array([p if p > 0 else 1000 + j for j, p in enumerate(positions)], dtype=float)
        d_delta = np.zeros(n)
        c_delta = np.zeros(n)
        for a in range(n):
            for b in range(a + 1, n):
                if rank[a] == rank[b]:
                    continue
                s_a = 1.0 if rank[a] < rank[b] else 0.0
                # Driver pair weight: a clean race is informative; if either DNF'd, halve it.
                w = 1.0 if (finished[a] and finished[b]) else 0.5
                exp_a = _expected(cur_d[a], cur_d[b])
                d_delta[a] += w * (s_a - exp_a)
                d_delta[b] += w * ((1 - s_a) - (1 - exp_a))
                exp_ca = _expected(cur_c[a], cur_c[b])
                c_delta[a] += (s_a - exp_ca)
                c_delta[b] += ((1 - s_a) - (1 - exp_ca))

        scale = 1.0 / max(n - 1, 1)
        for k, d in enumerate(drivers):
            kf = driver_k * (1.6 if driver_n.get(d, 0) < 10 else 1.0)
            driver_r[d] = driver_r.get(d, BASE_RATING) + kf * scale * d_delta[k]
            driver_n[d] = driver_n.get(d, 0) + 1
        for k, c in enumerate(cons):
            cons_r[c] = cons_r.get(c, BASE_RATING) + constructor_k * scale * c_delta[k]

    df = df.copy()
    df["elo_driver"] = out_d.astype("float32")
    df["elo_constructor"] = out_c.astype("float32")
    df["elo_combined"] = (0.6 * out_d + 0.4 * out_c).astype("float32")
    df["elo_driver_field_gap"] = out_dgap.astype("float32")
    df["elo_constructor_field_gap"] = out_cgap.astype("float32")
    return df


def latest_ratings(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Compute final post-history ratings (for projecting future races).

    Runs the same pairwise engine but returns the *post-last-race* ratings for every driver
    and constructor, plus appearance counts. Used by the future-race predictor to seed an
    upcoming round that has no results yet.
    """
    driver_r: dict[str, float] = {}
    cons_r: dict[str, float] = {}
    driver_n: dict[str, int] = {}
    driver_k, constructor_k, reset_regression = 32.0, 24.0, 0.35
    last_reset_applied = -1

    for (season, rnd), race in df.sort_values(["season", "round"]).groupby(
        ["season", "round"], sort=False
    ):
        season = int(season)
        if season in MAJOR_RESETS and season != last_reset_applied:
            if cons_r:
                mean_c = float(np.mean(list(cons_r.values())))
                for cid in cons_r:
                    cons_r[cid] += (mean_c - cons_r[cid]) * reset_regression
            last_reset_applied = season
        drivers = race["driver_id"].tolist()
        cons = race["constructor_id"].tolist()
        positions = race["position"].tolist()
        finished = race["finished"].tolist()
        n = len(drivers)
        cur_d = np.array([driver_r.get(d, BASE_RATING) for d in drivers])
        cur_c = np.array([cons_r.get(c, BASE_RATING) for c in cons])
        rank = np.array([p if p > 0 else 1000 + j for j, p in enumerate(positions)], dtype=float)
        d_delta = np.zeros(n)
        c_delta = np.zeros(n)
        for a in range(n):
            for b in range(a + 1, n):
                if rank[a] == rank[b]:
                    continue
                s_a = 1.0 if rank[a] < rank[b] else 0.0
                w = 1.0 if (finished[a] and finished[b]) else 0.5
                exp_a = _expected(cur_d[a], cur_d[b])
                d_delta[a] += w * (s_a - exp_a)
                d_delta[b] += w * ((1 - s_a) - (1 - exp_a))
                exp_ca = _expected(cur_c[a], cur_c[b])
                c_delta[a] += (s_a - exp_ca)
                c_delta[b] += ((1 - s_a) - (1 - exp_ca))
        scale = 1.0 / max(n - 1, 1)
        for k, d in enumerate(drivers):
            kf = driver_k * (1.6 if driver_n.get(d, 0) < 10 else 1.0)
            driver_r[d] = driver_r.get(d, BASE_RATING) + kf * scale * d_delta[k]
            driver_n[d] = driver_n.get(d, 0) + 1
        for k, c in enumerate(cons):
            cons_r[c] = cons_r.get(c, BASE_RATING) + constructor_k * scale * c_delta[k]

    return {"driver": driver_r, "constructor": cons_r, "driver_n": driver_n}
