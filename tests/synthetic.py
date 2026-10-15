"""Synthetic race laps generated from known effects (shared by the lap-model and simulator
tests)."""

from __future__ import annotations

import numpy as np
import polars as pl

from f1p.lake.curate_f1db import Curated
from f1p.lake.laps import lap_rows
from f1p.lake.race_pace import Lap

# true pace (fraction of a lap slower than the fastest car) per team lineage
TRUE = {
    "mclaren": 0.000,
    "red_bull": 0.002,
    "mercedes": 0.004,
    "ferrari": 0.003,
    "aston_martin": 0.009,
    "williams": 0.011,
    "racing_bulls": 0.010,
    "alpine": 0.013,
    "haas": 0.012,
    "audi": 0.014,
    "cadillac": 0.020,
}
DEG = {"SOFT": 0.0012, "MEDIUM": 0.0008, "HARD": 0.0005}
OFFSET = {"SOFT": -0.004, "MEDIUM": 0.0, "HARD": 0.004}


def synthetic_laps(curated: Curated, season: int = 2026, seed: int = 0) -> pl.DataFrame:
    """Two-stint races for every classified car, generated from known effects."""
    rng = np.random.default_rng(seed)
    t = curated.tables
    res = t["results"].filter(
        (pl.col("season") == season) & (pl.col("session") == "race") & pl.col("classified")
    )
    cs = t["constructor_seasons"].filter(pl.col("season") == season)
    lineage = {c: (lin or c) for c, lin in cs.select("constructor_id", "lineage_id").iter_rows()}
    frames = []
    for (event_id,), grp in res.group_by("event_id"):
        base = 90.0 + rng.normal(0, 5)
        n_laps = 50
        observed = grp["observed_at"].max()
        rows = []
        for r in grp.iter_rows(named=True):
            team = lineage.get(r["constructor_id"], r["constructor_id"])
            pace = TRUE.get(team, 0.015)
            stop = int(rng.integers(18, 30))
            first, second = ("MEDIUM", "HARD") if rng.random() < 0.6 else ("SOFT", "HARD")
            laps = []
            for lap in range(1, n_laps + 1):
                comp, age = (first, lap) if lap <= stop else (second, lap - stop)
                log_t = (
                    np.log(base)
                    + pace
                    + OFFSET[comp]
                    + DEG[comp] * age
                    + 0.0006 * (n_laps - lap)
                    + rng.standard_t(4) * 0.003
                )
                laps.append(
                    Lap(
                        str(r["car_number"]),
                        lap,
                        float(np.exp(log_t)),
                        float(lap * 90),
                        lap == stop,
                        lap == stop + 1,
                        None,
                        None,
                    )
                )
            for row in lap_rows(laps, neutral=[]):
                row["compound"] = first if row["lap"] <= stop else second
                row["tyre_age"] = row["lap"] if row["lap"] <= stop else row["lap"] - stop
                row["driver_id"] = r["driver_id"]
                rows.append(row)
        frames.append(
            pl.DataFrame(rows).with_columns(
                pl.lit(event_id).alias("event_id"),
                pl.lit("synthetic").alias("source"),
                pl.lit(observed).alias("observed_at"),
            )
        )
    return pl.concat(frames)
