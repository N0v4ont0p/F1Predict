"""Race pace from lap times (plan §6-7, Phase 4 groundwork).

A finishing order mixes pace with luck (incidents, strategy, retirements); lap times measure
pace. For every race since 2018 this module extracts each car's **clean laps** from the
live-timing TimingData stream:

* lap 1, in-laps and out-laps are dropped;
* any lap that overlaps a safety car, virtual safety car or red flag is dropped
  (TrackStatus stream);
* laps slower than 107% of the car's own median are dropped (traffic, damage, cool-down).

``race_pace`` holds, per car, the number of clean laps, the median clean lap and the
**pace deficit**: how much slower (in %) that median is than the fastest car's.
Observed when the race result is public.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from f1p.core.logging import get_logger
from f1p.lake import schema as S
from f1p.sources import livetiming
from f1p.sources.http import FetchError, HttpClient

log = get_logger("lake.race_pace")

NEUTRAL = frozenset({"4", "5", "6", "7"})  # SC, red flag, VSC deployed, VSC ending
SLOW_LAP = 1.07
MIN_CLEAN_LAPS = 5


@dataclass(frozen=True, slots=True)
class Lap:
    car: str
    lap: int
    seconds: float
    end: float  # session time the lap was completed
    in_lap: bool
    out_lap: bool


def lap_seconds(value: str) -> float | None:
    if not value:
        return None
    try:
        parts = value.split(":")
        return int(parts[0]) * 60 + float(parts[1]) if len(parts) == 2 else float(parts[0])
    except ValueError:
        return None


def parse_laps(patches: Iterable[tuple[float, dict[str, Any]]]) -> list[Lap]:
    """Completed laps per car from TimingData patches (lap 1 has no time and is absent)."""
    completed: dict[str, int] = {}
    raw: dict[str, list[tuple[float, int, float]]] = defaultdict(list)
    in_laps: dict[str, set[int]] = defaultdict(set)
    out_laps: dict[str, set[int]] = defaultdict(set)
    for t, patch in patches:
        lines = patch.get("Lines")
        if not isinstance(lines, dict):
            continue
        for car, upd in lines.items():
            if not isinstance(upd, dict):
                continue
            if isinstance(upd.get("NumberOfLaps"), int):
                completed[car] = upd["NumberOfLaps"]
            done = completed.get(car)
            if done is not None:
                if upd.get("InPit") is True:
                    in_laps[car].add(done + 1)
                if upd.get("PitOut") is True:
                    out_laps[car].add(done + 1)
            llt = upd.get("LastLapTime")
            if isinstance(llt, dict) and (secs := lap_seconds(llt.get("Value", ""))) and done:
                raw[car].append((t, done, secs))
    laps = []
    for car, rows in raw.items():
        seen: set[int] = set()
        for t, lap, secs in rows:
            if lap in seen:
                continue  # repeated value for the same lap
            seen.add(lap)
            laps.append(Lap(car, lap, secs, t, lap in in_laps[car], lap in out_laps[car]))
    return laps


def neutralised(
    patches: Iterable[tuple[float, dict[str, Any]]], end: float = 1e9
) -> list[tuple[float, float]]:
    """Session-time intervals under SC / VSC / red flag (TrackStatus stream)."""
    out: list[tuple[float, float]] = []
    start: float | None = None
    for t, patch in patches:
        status = str(patch.get("Status", ""))
        if status in NEUTRAL and start is None:
            start = t
        elif status not in NEUTRAL and start is not None:
            out.append((start, t))
            start = None
    if start is not None:
        out.append((start, end))
    return out


def pace_table(laps: list[Lap], neutral: list[tuple[float, float]]) -> pl.DataFrame:
    """Per car: clean laps, median clean lap (ms) and pace deficit to the fastest car (%)."""

    def overlaps(lap: Lap) -> bool:
        a, b = lap.end - lap.seconds, lap.end
        return any(a < hi and b > lo for lo, hi in neutral)

    by_car: dict[str, list[float]] = defaultdict(list)
    for lap in laps:
        if lap.lap <= 1 or lap.in_lap or lap.out_lap or overlaps(lap):
            continue
        by_car[lap.car].append(lap.seconds)
    rows = []
    for car, times in by_car.items():
        arr = np.asarray(times)
        if arr.size == 0:
            continue
        arr = arr[arr <= SLOW_LAP * np.median(arr)]
        if arr.size >= MIN_CLEAN_LAPS:
            rows.append(
                {"car_number": car, "clean_laps": int(arr.size), "median_s": float(np.median(arr))}
            )
    if not rows:
        return pl.DataFrame(
            schema={
                "car_number": pl.String,
                "clean_laps": pl.Int32,
                "median_lap_ms": pl.Int64,
                "pace_deficit_pct": pl.Float64,
            }
        )
    df = pl.DataFrame(rows)
    best = df["median_s"].min()
    return df.select(
        "car_number",
        pl.col("clean_laps").cast(pl.Int32),
        (pl.col("median_s") * 1000).round().cast(pl.Int64).alias("median_lap_ms"),
        ((pl.col("median_s") / best - 1.0) * 100.0).alias("pace_deficit_pct"),
    )


def build(
    http: HttpClient,
    raw_dir: Path,
    tables: dict[str, pl.DataFrame],
    *,
    seasons: tuple[int, int],
) -> pl.DataFrame:
    """Fetch, parse and assemble ``race_pace`` for every raced event in ``seasons``."""
    lo, hi = max(seasons[0], livetiming.FIRST_SEASON), seasons[1]
    events = tables["events"].filter(pl.col("season").is_between(lo, hi))
    results = tables["results"].filter(pl.col("session") == "race")
    frames = []
    for season in range(lo, hi + 1):
        try:
            refs = [
                r
                for r in livetiming.sessions(http, season)
                if r.kind == "Race" and r.name == "Race"
            ]
        except FetchError as exc:
            log.warning("no live-timing index for %s: %s", season, exc)
            continue
        season_events = events.filter(pl.col("season") == season)
        for ref in refs:
            match = season_events.filter(
                (pl.col("race_date") - pl.lit(ref.day)).dt.total_days().abs() <= 1
            )
            if match.height != 1:
                continue
            event_id = str(match["event_id"][0])
            res = results.filter(pl.col("event_id") == event_id)
            if res.is_empty():
                continue  # not raced yet
            try:
                td = livetiming.fetch_stream(http, raw_dir, ref, "TimingData")
                ts = livetiming.fetch_stream(http, raw_dir, ref, "TrackStatus")
            except FetchError as exc:
                log.warning("live timing for %s unavailable: %s", event_id, exc)
                continue
            pace = pace_table(
                parse_laps(livetiming.read_stream(td)), neutralised(livetiming.read_stream(ts))
            )
            numbers = res.select(pl.col("car_number").cast(pl.String), "driver_id").unique(
                "car_number"
            )
            frames.append(
                pace.join(numbers, on="car_number").with_columns(
                    pl.lit(event_id).alias("event_id"),
                    pl.lit(res["observed_at"].max()).alias("observed_at"),
                )
            )
    spec = S.RACE_PACE
    if not frames:
        return S.conform(pl.DataFrame(schema=spec.columns), spec)
    df = (
        S.conform(pl.concat(frames, how="diagonal_relaxed"), spec)
        .unique(["event_id", "driver_id"], keep="first")
        .sort("event_id", "pace_deficit_pct")
    )
    S.validate(df, spec)
    return df
