"""Every race lap since 2018, with tyres and traffic (plan item 6, Phase 4b data).

One row per car and completed lap of a Grand Prix:

* lap time, running position and the interval to the car ahead at the line (TimingData);
* stint number, compound and **tyre age** (TimingAppData — the archive's own tyre-life
  counter, which also covers used sets);
* in-lap / out-lap flags and whether the lap overlapped a safety car, VSC or red flag
  (TrackStatus);
* ``clean``: not lap 1, not an in- or out-lap, not neutralised and within 107% of the car's
  own median — the laps a pace model should learn from.

Where the live-timing archive does not serve a race (all of 2022, a few others), laps come
from Jolpica: times and positions only, stints from the pit-stop list, no compounds, tyre age
counted from the stint start (sets assumed new).
"""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from f1p.core.logging import get_logger
from f1p.lake import schema as S
from f1p.lake.race_pace import (
    SLOW_LAP,
    Lap,
    _race_refs,
    jolpica_laps,
    neutral_laps_from_field,
    neutralised,
    parse_laps,
)
from f1p.sources import livetiming
from f1p.sources.http import FetchError, HttpClient

log = get_logger("lake.laps")

COMPOUNDS = ("SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET")
TYRE_LAG = 10.0  # s after the line by which the archive has bumped the tyre-life counter


class StintTimeline:
    """Per car: the (stint, tyre age) state over session time, and each stint's compound."""

    def __init__(self) -> None:
        self.times: dict[str, list[float]] = defaultdict(list)
        self.states: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.compound: dict[str, dict[int, str]] = defaultdict(dict)

    def at(self, car: str, t: float) -> tuple[int, int] | None:
        """``(stint index, tyre age in laps)`` at session time ``t``."""
        times = self.times.get(car)
        if not times:
            return None
        i = bisect_right(times, t) - 1
        return self.states[car][i] if i >= 0 else None


def parse_stints(patches: Iterable[tuple[float, dict[str, Any]]]) -> StintTimeline:
    """Fold TimingAppData patches into a :class:`StintTimeline`.

    Stints arrive first as a list, then as ``{"index": {...}}`` patches; ``TotalLaps`` is
    the set's age in laps (including laps run before this stint for a used set)."""
    tl = StintTimeline()
    current: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for t, patch in patches:
        lines = patch.get("Lines")
        if not isinstance(lines, dict):
            continue
        for car, upd in lines.items():
            if not isinstance(upd, dict) or "Stints" not in upd:
                continue
            stints = upd["Stints"]
            items = enumerate(stints) if isinstance(stints, list) else stints.items()
            changed = False
            for k, val in items:
                if not isinstance(val, dict):
                    continue
                try:
                    idx = int(k)
                except (TypeError, ValueError):
                    continue
                current[car].setdefault(idx, {}).update(val)
                comp = val.get("Compound")
                if isinstance(comp, str) and comp.upper() in COMPOUNDS:
                    tl.compound[car][idx] = comp.upper()
                changed = True
            if changed and current[car]:
                idx = max(current[car])
                age = current[car][idx].get("TotalLaps")
                tl.times[car].append(t)
                tl.states[car].append((idx, int(age) if isinstance(age, int) else 0))
    return tl


def _clean_mask(rows: list[dict[str, Any]]) -> None:
    """Set ``clean`` in place: usable laps within 107% of the car's own median."""
    by_car: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_car[r["car_number"]].append(r)
    for laps in by_car.values():
        usable = [
            r for r in laps if r["lap"] > 1 and not (r["in_lap"] or r["out_lap"] or r["neutral"])
        ]
        if not usable:
            for r in laps:
                r["clean"] = False
            continue
        cap = SLOW_LAP * float(np.median([r["lap_ms"] for r in usable]))
        keep = {id(r) for r in usable if r["lap_ms"] <= cap}
        for r in laps:
            r["clean"] = id(r) in keep


def lap_rows(
    laps: list[Lap],
    *,
    neutral: list[tuple[float, float]] | None = None,
    neutral_laps: set[int] | None = None,
    stints: StintTimeline | None = None,
) -> list[dict[str, Any]]:
    """Rows for one race. Neutralised laps come as session-time intervals (live timing) or
    lap numbers (Jolpica); without a stint timeline, stints are counted from in-laps."""
    skip = neutral_laps or set()
    windows = neutral or []
    rows: list[dict[str, Any]] = []
    by_car: dict[str, list[Lap]] = defaultdict(list)
    for lap in laps:
        by_car[lap.car].append(lap)
    for car, car_laps in by_car.items():
        car_laps.sort(key=lambda x: x.lap)
        stint, age, prev_in = 0, 0, False
        for lap in car_laps:
            a, b = lap.end - lap.seconds, lap.end
            is_neutral = lap.lap in skip or any(a < hi and b > lo for lo, hi in windows)
            compound = None
            state = stints.at(car, lap.end + TYRE_LAG) if stints is not None else None
            if state is not None:
                stint, age = state
                compound = stints.compound.get(car, {}).get(stint) if stints else None
            else:
                if prev_in:
                    stint, age = stint + 1, 0
                age += 1
                prev_in = lap.in_lap
            rows.append(
                {
                    "car_number": car,
                    "lap": lap.lap,
                    "lap_ms": round(lap.seconds * 1000),
                    "position": lap.position,
                    "gap_ahead_s": lap.gap_ahead,
                    "stint": stint + 1,
                    "compound": compound,
                    "tyre_age": age if state is not None or stints is None else None,
                    "in_lap": lap.in_lap,
                    "out_lap": lap.out_lap,
                    "neutral": is_neutral,
                }
            )
    _clean_mask(rows)
    return rows


def livetiming_laps(
    http: HttpClient, raw_dir: Path, ref: livetiming.SessionRef
) -> list[dict[str, Any]]:
    td = livetiming.fetch_stream(http, raw_dir, ref, "TimingData")
    ts = livetiming.fetch_stream(http, raw_dir, ref, "TrackStatus")
    try:
        app = livetiming.fetch_stream(http, raw_dir, ref, "TimingAppData")
        stints: StintTimeline | None = parse_stints(livetiming.read_stream(app))
    except FetchError as exc:
        log.warning("no tyre data for %s: %s", ref.path, exc)
        stints = None
    return lap_rows(
        parse_laps(livetiming.read_stream(td)),
        neutral=neutralised(livetiming.read_stream(ts)),
        stints=stints,
    )


def jolpica_lap_rows(http: HttpClient, season: int, rnd: int) -> list[dict[str, Any]]:
    laps = jolpica_laps(http, season, rnd)
    return lap_rows(laps, neutral_laps=neutral_laps_from_field(laps))


def build(
    http: HttpClient,
    raw_dir: Path,
    tables: dict[str, pl.DataFrame],
    *,
    seasons: tuple[int, int],
    jolpica_fallback: bool = True,
) -> pl.DataFrame:
    """``laps`` for every raced Grand Prix in ``seasons`` (2018+)."""
    lo, hi = max(seasons[0], livetiming.FIRST_SEASON), seasons[1]
    events = tables["events"].filter(pl.col("season").is_between(lo, hi))
    results = tables["results"].filter(pl.col("session") == "race")
    frames = []
    for season in range(lo, hi + 1):
        season_events = events.filter(pl.col("season") == season).sort("round")
        refs = _race_refs(http, season, season_events)
        for ev in season_events.iter_rows(named=True):
            event_id = ev["event_id"]
            res = results.filter(pl.col("event_id") == event_id)
            if res.is_empty():
                continue
            rows, source = [], None
            if event_id in refs:
                try:
                    rows, source = livetiming_laps(http, raw_dir, refs[event_id]), "livetiming"
                except FetchError as exc:
                    log.warning("live timing for %s unavailable: %s", event_id, exc)
            if not rows and jolpica_fallback:
                try:
                    rows, source = jolpica_lap_rows(http, season, ev["round"]), "jolpica"
                except FetchError as exc:
                    log.warning("Jolpica laps for %s unavailable: %s", event_id, exc)
            if not rows:
                continue
            numbers = res.select(pl.col("car_number").cast(pl.String), "driver_id").unique(
                "car_number"
            )
            frames.append(
                pl.DataFrame(rows, infer_schema_length=None)
                .join(numbers, on="car_number")
                .with_columns(
                    pl.lit(event_id).alias("event_id"),
                    pl.lit(source).alias("source"),
                    pl.lit(res["observed_at"].max()).alias("observed_at"),
                )
            )
    spec = S.LAPS
    if not frames:
        return S.conform(pl.DataFrame(schema=spec.columns), spec)
    df = (
        S.conform(pl.concat(frames, how="diagonal_relaxed"), spec)
        .unique(["event_id", "driver_id", "lap"], keep="first")
        .sort("event_id", "driver_id", "lap")
    )
    S.validate(df, spec)
    return df
