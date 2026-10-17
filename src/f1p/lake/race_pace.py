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
from f1p.sources import jolpica, livetiming
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
    position: int | None = None  # running position when the lap was completed
    gap_ahead: float | None = None  # seconds to the car ahead at the line (None: leader / lapped)


def lap_seconds(value: str) -> float | None:
    if not value:
        return None
    try:
        parts = value.split(":")
        return int(parts[0]) * 60 + float(parts[1]) if len(parts) == 2 else float(parts[0])
    except ValueError:
        return None


def _interval(value: Any) -> float | None:
    """``"+0.605"`` → 0.605; lapped cars ("1L") and the leader have no interval."""
    if isinstance(value, dict):
        value = value.get("Value")
    if not isinstance(value, str) or not value.startswith("+"):
        return None
    try:
        return float(value[1:])
    except ValueError:
        return None


def parse_laps(patches: Iterable[tuple[float, dict[str, Any]]]) -> list[Lap]:
    """Completed laps per car from TimingData patches (lap 1 has no time and is absent),
    with the running position and the interval to the car ahead at the line."""
    completed: dict[str, int] = {}
    position: dict[str, int] = {}
    gap: dict[str, float | None] = {}
    raw: dict[str, list[tuple[float, int, float, int | None, float | None]]] = defaultdict(list)
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
            line = upd.get("Line", upd.get("Position"))
            if isinstance(line, int) or (isinstance(line, str) and line.isdigit()):
                position[car] = int(line)
            if "IntervalToPositionAhead" in upd:
                gap[car] = _interval(upd["IntervalToPositionAhead"])
            done = completed.get(car)
            if done is not None:
                if upd.get("InPit") is True:
                    in_laps[car].add(done + 1)
                if upd.get("PitOut") is True:
                    out_laps[car].add(done + 1)
            llt = upd.get("LastLapTime")
            if isinstance(llt, dict) and (secs := lap_seconds(llt.get("Value", ""))) and done:
                raw[car].append((t, done, secs, position.get(car), gap.get(car)))
    laps = []
    for car, rows in raw.items():
        seen: set[int] = set()
        for t, lap, secs, pos, ahead in rows:
            if lap in seen:
                continue  # repeated value for the same lap
            seen.add(lap)
            laps.append(
                Lap(car, lap, secs, t, lap in in_laps[car], lap in out_laps[car], pos, ahead)
            )
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


FIELD_SLOW = 1.10  # a lap is neutralised if the whole field ran it this much slower


def neutral_laps_from_field(laps: list[Lap]) -> set[int]:
    """Lap numbers run under SC / VSC / red flag, inferred from the field's own times — for
    sources without a track-status feed: the field's median lap time is far above the norm."""
    by_lap: dict[int, list[float]] = defaultdict(list)
    for lap in laps:
        if lap.lap > 1:
            by_lap[lap.lap].append(lap.seconds)
    if not by_lap:
        return set()
    med = {n: float(np.median(v)) for n, v in by_lap.items()}
    norm = float(np.median(list(med.values())))
    return {n for n, m in med.items() if m > FIELD_SLOW * norm}


def pace_table(
    laps: list[Lap],
    neutral: list[tuple[float, float]],
    neutral_laps: set[int] | None = None,
) -> pl.DataFrame:
    """Per car: clean laps, median clean lap (ms) and pace deficit to the fastest car (%).

    Neutralised laps are given as session-time intervals (live timing) or as lap numbers
    (sources without timestamps)."""
    skip = neutral_laps or set()

    def overlaps(lap: Lap) -> bool:
        a, b = lap.end - lap.seconds, lap.end
        return any(a < hi and b > lo for lo, hi in neutral)

    by_car: dict[str, list[float]] = defaultdict(list)
    for lap in laps:
        if lap.lap <= 1 or lap.in_lap or lap.out_lap or lap.lap in skip or overlaps(lap):
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


def jolpica_laps(http: HttpClient, season: int, rnd: int) -> list[Lap]:
    """Laps from Jolpica, keyed by car number, with in/out laps from the pit-stop list."""
    numbers = jolpica.car_numbers(http, season, rnd)
    pits = jolpica.pit_laps(http, season, rnd)
    out = []
    for driver, lap, secs, pos in jolpica.laps(http, season, rnd):
        car = numbers.get(driver)
        if car is None:
            continue
        stops = pits.get(driver, set())
        out.append(Lap(car, lap, secs, 0.0, lap in stops, (lap - 1) in stops, pos))
    return out


def jolpica_pace(http: HttpClient, season: int, rnd: int) -> pl.DataFrame:
    laps = jolpica_laps(http, season, rnd)
    return pace_table(laps, [], neutral_laps_from_field(laps))


def livetiming_pace(http: HttpClient, raw_dir: Path, ref: livetiming.SessionRef) -> pl.DataFrame:
    td = livetiming.fetch_stream(http, raw_dir, ref, "TimingData")
    ts = livetiming.fetch_stream(http, raw_dir, ref, "TrackStatus")
    return pace_table(
        parse_laps(livetiming.read_stream(td)), neutralised(livetiming.read_stream(ts))
    )


def _race_refs(
    http: HttpClient, season: int, events: pl.DataFrame
) -> dict[str, livetiming.SessionRef]:
    """Live-timing race sessions of ``season`` matched to F1DB event ids (empty if unserved)."""
    try:
        refs = [
            r for r in livetiming.sessions(http, season) if r.kind == "Race" and r.name == "Race"
        ]
    except FetchError as exc:
        log.warning("no live-timing index for %s (%s) — using Jolpica", season, exc)
        return {}
    out = {}
    for ref in refs:
        match = events.filter((pl.col("race_date") - pl.lit(ref.day)).dt.total_days().abs() <= 1)
        if match.height == 1:
            out[str(match["event_id"][0])] = ref
    return out


def build(
    http: HttpClient,
    raw_dir: Path,
    tables: dict[str, pl.DataFrame],
    *,
    seasons: tuple[int, int],
    jolpica_fallback: bool = True,
) -> pl.DataFrame:
    """``race_pace`` for every raced event in ``seasons``: live timing where served, Jolpica
    (if ``jolpica_fallback``) where not."""
    lo, hi = max(seasons[0], livetiming.FIRST_SEASON), seasons[1]
    events = tables["events"].filter(pl.col("season").is_between(lo, hi))
    results = tables["results"].filter(pl.col("session") == "race")
    frames = []
    for season in range(lo, hi + 1):
        season_events = events.filter(pl.col("season") == season).sort("round")
        refs = _race_refs(http, season, season_events)
        for ev in season_events.iter_rows(named=True):
            event_id = ev["event_id"]
            log.info("race pace: %s", event_id)
            res = results.filter(pl.col("event_id") == event_id)
            if res.is_empty():
                continue  # not raced yet
            pace, source = None, None
            if event_id in refs:
                try:
                    pace, source = livetiming_pace(http, raw_dir, refs[event_id]), "livetiming"
                except FetchError as exc:
                    log.warning("live timing for %s unavailable: %s", event_id, exc)
            if (pace is None or pace.is_empty()) and jolpica_fallback:
                try:
                    pace, source = jolpica_pace(http, season, ev["round"]), "jolpica"
                except FetchError as exc:
                    log.warning("Jolpica laps for %s unavailable: %s", event_id, exc)
            if pace is None or pace.is_empty():
                continue
            numbers = res.select(pl.col("car_number").cast(pl.String), "driver_id").unique(
                "car_number"
            )
            frames.append(
                pace.join(numbers, on="car_number").with_columns(
                    pl.lit(event_id).alias("event_id"),
                    pl.lit(source).alias("source"),
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


def compare_sources(
    http: HttpClient,
    raw_dir: Path,
    tables: dict[str, pl.DataFrame],
    *,
    season: int,
    rounds: list[int],
) -> pl.DataFrame:
    """Per-car pace deficit from both sources on races both serve (validation of the fallback)."""
    events = tables["events"].filter(pl.col("season") == season)
    refs = _race_refs(http, season, events)
    rows = []
    for ev in events.filter(pl.col("round").is_in(rounds)).iter_rows(named=True):
        ref = refs.get(ev["event_id"])
        if ref is None:
            continue
        a = livetiming_pace(http, raw_dir, ref).select(
            "car_number", pl.col("pace_deficit_pct").alias("livetiming")
        )
        b = jolpica_pace(http, season, ev["round"]).select(
            "car_number", pl.col("pace_deficit_pct").alias("jolpica")
        )
        rows.append(
            a.join(b, on="car_number").with_columns(pl.lit(ev["event_id"]).alias("event_id"))
        )
    return pl.concat(rows) if rows else pl.DataFrame()
