"""Circuit geometry for the 3D pages (plan item 9), from car positions.

For every circuit on a season's calendar, one clean racing lap of the most recent Grand Prix
there (2023 onwards) is taken from OpenF1's car-position feed (≈3.7 Hz x/y/z). The winner's
lap is used: not lap 1, not an out-lap, within 3% of their median lap. The trace is
resampled to evenly spaced points along its length, lightly smoothed as a closed loop, and
tagged with the timing sectors (from that lap's sector times). Point 0 is the timing line,
and the points run in the direction of travel.

Coordinates are OpenF1's (decimetres, arbitrary origin); the web app normalises them.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from f1p.core.logging import get_logger
from f1p.lake import schema as S
from f1p.sources import openf1
from f1p.sources.http import FetchError, HttpClient

log = get_logger("lake.circuit_geometry")

POINTS = 480
SMOOTH = 2  # half-width (points) of the circular moving average


def _iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "")


def clean_lap(laps: list[dict[str, Any]]) -> dict[str, Any] | None:
    timed = [
        lap
        for lap in laps
        if lap.get("lap_duration") and lap.get("date_start") and not lap.get("is_pit_out_lap")
    ]
    if len(timed) < 8:
        return None
    med = float(np.median([lap["lap_duration"] for lap in timed]))
    for lap in sorted(timed, key=lambda x: x["lap_number"]):
        if lap["lap_number"] >= 5 and lap["lap_duration"] <= 1.03 * med:
            return lap
    return None


def resample(
    xyz: NDArray[np.float64], t: NDArray[np.float64], splits: tuple[float, float], n: int = POINTS
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Evenly spaced points along the closed trace, smoothed, with the sector of each."""
    seg = np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1)
    keep = np.concatenate([[True], seg > 1e-6])
    xyz, t = xyz[keep], t[keep]
    closed = np.vstack([xyz, xyz[:1]])
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(closed[:, :2], axis=0), axis=1))])
    target = np.linspace(0.0, s[-1], n, endpoint=False)
    out = np.column_stack([np.interp(target, s, closed[:, k]) for k in range(3)])
    if SMOOTH:
        kernel = np.ones(2 * SMOOTH + 1) / (2 * SMOOTH + 1)
        out = np.column_stack(
            [
                np.convolve(np.concatenate([c[-SMOOTH:], c, c[:SMOOTH]]), kernel, "valid")
                for c in out.T
            ]
        )
    t_closed = np.concatenate([t, [t[-1] + (t[-1] - t[-2] if t.size > 1 else 0.0)]])
    t_at = np.interp(target, s, t_closed)
    sector = 1 + (t_at >= splits[0]).astype(np.int64) + (t_at >= splits[1]).astype(np.int64)
    return out, sector


def circuit_trace(
    http: HttpClient, session_key: int, driver_number: str
) -> tuple[NDArray[np.float64], NDArray[np.int64]] | None:
    laps = http.get_json(
        f"{openf1.BASE}/laps", {"session_key": session_key, "driver_number": driver_number}
    )
    lap = clean_lap(laps if isinstance(laps, list) else [])
    if lap is None:
        return None
    start = datetime.fromisoformat(lap["date_start"])
    end = start + timedelta(seconds=float(lap["lap_duration"]))
    url = (
        f"{openf1.BASE}/location?session_key={session_key}&driver_number={driver_number}"
        f"&date>={_iso(start)}&date<={_iso(end)}"
    )
    rows = http.get_json(url)
    if not isinstance(rows, list) or len(rows) < 50:
        return None
    rows = sorted(rows, key=lambda r: r["date"])
    xyz = np.array([[r["x"], r["y"], r["z"]] for r in rows], dtype=np.float64)
    t = np.array([(datetime.fromisoformat(r["date"]) - start).total_seconds() for r in rows])
    s1 = float(lap.get("duration_sector_1") or lap["lap_duration"] / 3)
    s2 = float(lap.get("duration_sector_2") or lap["lap_duration"] / 3)
    return resample(xyz, t, (s1, s1 + s2))


def build(http: HttpClient, tables: dict[str, pl.DataFrame], *, season: int) -> pl.DataFrame:
    """Geometry of every circuit on ``season``'s calendar (latest race there since 2023)."""
    events = tables["events"]
    results = tables["results"].filter((pl.col("session") == "race") & (pl.col("position") == 1))
    circuits = events.filter(pl.col("season") == season)["circuit_id"].unique().to_list()
    sessions: dict[int, dict[Any, int]] = {}
    rows = []
    for circuit in sorted(circuits):
        past = (
            events.filter(
                (pl.col("circuit_id") == circuit) & (pl.col("season") >= openf1.FIRST_SEASON)
            )
            .join(results.select("event_id", "car_number"), on="event_id")
            .sort("season", "round", descending=True)
        )
        done = False
        for ev in past.iter_rows(named=True):
            if ev["season"] not in sessions:
                try:
                    sessions[ev["season"]] = openf1.race_sessions(http, ev["season"])
                except FetchError as exc:
                    log.warning("no OpenF1 sessions for %s: %s", ev["season"], exc)
                    sessions[ev["season"]] = {}
            key = next(
                (
                    k
                    for d, k in sessions[ev["season"]].items()
                    if abs((d - ev["race_date"]).days) <= 1
                ),
                None,
            )
            if key is None:
                continue
            try:
                trace = circuit_trace(http, key, str(ev["car_number"]))
            except FetchError as exc:
                log.warning("no position data for %s: %s", ev["event_id"], exc)
                continue
            if trace is None:
                continue
            pts, sector = trace
            rows.extend(
                {
                    "circuit_id": circuit,
                    "point": i,
                    "x": float(p[0]),
                    "y": float(p[1]),
                    "z": float(p[2]),
                    "sector": int(sec),
                    "source_event_id": ev["event_id"],
                }
                for i, (p, sec) in enumerate(zip(pts, sector, strict=True))
            )
            done = True
            break
        if not done:
            log.warning("no geometry for %s", circuit)
    spec = S.CIRCUIT_GEOMETRY
    if not rows:
        return S.conform(pl.DataFrame(schema=spec.columns), spec)
    df = S.conform(pl.DataFrame(rows), spec).sort("circuit_id", "point")
    S.validate(df, spec)
    return df
