"""jolpica-f1 API client (Ergast-compatible successor) — multi-endpoint, cached.

Fetches **race results**, **qualifying** and **sprint** for a season and merges them into
the enriched canonical schema. Network failures raise :class:`JolpicaError`.

Politeness: jolpica is rate-limited (~4 req/s burst, 500/hr). We page with a small sleep
and the builder caches each season to Parquet so we only hit the network once.
"""
from __future__ import annotations

import time

import pandas as pd
import requests

from ..utils.logging import get_logger
from .schema import RAW_COLUMNS

log = get_logger()
DEFAULT_BASE = "https://api.jolpi.ca/ergast/f1"


class JolpicaError(RuntimeError):
    pass


def _get(url: str, timeout: int) -> dict:
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, ValueError) as exc:
        raise JolpicaError(f"jolpica fetch failed: {url}: {exc}") from exc


def _paginate(path: str, base: str, timeout: int, sleep: float):
    """Yield each MRData payload across all pages for an endpoint path."""
    offset, limit = 0, 100
    while True:
        payload = _get(f"{base}/{path}?limit={limit}&offset={offset}", timeout)
        yield payload
        total = int(payload.get("MRData", {}).get("total", 0))
        offset += limit
        if offset >= total or total == 0:
            break
        time.sleep(sleep)


def _time_to_ms(t: str | None) -> float:
    """Parse a lap time like '1:23.456' or '83.456' into milliseconds."""
    if not t:
        return 0.0
    try:
        if ":" in t:
            m, s = t.split(":")
            return (int(m) * 60 + float(s)) * 1000.0
        return float(t) * 1000.0
    except ValueError:
        return 0.0


def _dnf_status(status: str) -> int:
    s = status.lower()
    if status == "Finished" or status.startswith("+"):
        return 0
    # Non-classification reasons that are genuine retirements.
    return 1


def fetch_season(
    season: int,
    base_url: str = DEFAULT_BASE,
    timeout: int = 25,
    sleep: float = 0.4,
) -> pd.DataFrame:
    """Fetch and merge results + qualifying + sprint for a full season."""
    # --- race results -----------------------------------------------------------------
    rows: dict[tuple, dict] = {}
    race_meta: dict[int, dict] = {}
    for payload in _paginate(f"{season}/results.json", base_url, timeout, sleep):
        for race in payload.get("MRData", {}).get("RaceTable", {}).get("Races", []):
            rnd = int(race.get("round", 0))
            circ = race.get("Circuit", {})
            loc = circ.get("Location", {})
            race_meta[rnd] = {
                "race_name": race.get("raceName", ""),
                "circuit_id": circ.get("circuitId", ""),
                "country": loc.get("country", ""),
                "locality": loc.get("locality", ""),
                "date": race.get("date", ""),
            }
            for res in race.get("Results", []):
                drv, con = res.get("Driver", {}), res.get("Constructor", {})
                did = drv.get("driverId", "")
                status = res.get("status", "")
                pt = res.get("positionText", "")
                pos = int(pt) if pt.isdigit() else 0
                rows[(rnd, did)] = {
                    "season": season, "round": rnd, **race_meta[rnd],
                    "driver_id": did, "driver_code": drv.get("code", did[:3].upper()),
                    "driver_name": f"{drv.get('givenName','')} {drv.get('familyName','')}".strip(),
                    "constructor_id": con.get("constructorId", ""),
                    "constructor_name": con.get("name", ""),
                    "grid": int(res.get("grid", 0) or 0),
                    "position": pos, "status": status,
                    "points": float(res.get("points", 0) or 0),
                    "laps": int(res.get("laps", 0) or 0),
                    "finished": 1 if (status == "Finished" or status.startswith("+")) else 0,
                    "dnf": _dnf_status(status),
                    "quali_position": 0, "quali_best_ms": 0.0,
                    "sprint_position": 0, "sprint_points": 0.0, "has_sprint": 0,
                }
    if not rows:
        raise JolpicaError(f"jolpica returned no race results for {season}")

    # --- qualifying -------------------------------------------------------------------
    try:
        for payload in _paginate(f"{season}/qualifying.json", base_url, timeout, sleep):
            for race in payload.get("MRData", {}).get("RaceTable", {}).get("Races", []):
                rnd = int(race.get("round", 0))
                for q in race.get("QualifyingResults", []):
                    did = q.get("Driver", {}).get("driverId", "")
                    key = (rnd, did)
                    if key not in rows:
                        continue
                    rows[key]["quali_position"] = int(q.get("position", 0) or 0)
                    best = min(
                        (_time_to_ms(q.get(s)) for s in ("Q3", "Q2", "Q1") if q.get(s)),
                        default=0.0,
                    )
                    rows[key]["quali_best_ms"] = best
    except JolpicaError:
        log.warning(f"qualifying unavailable for {season}")

    # --- sprint -----------------------------------------------------------------------
    try:
        for payload in _paginate(f"{season}/sprint.json", base_url, timeout, sleep):
            for race in payload.get("MRData", {}).get("RaceTable", {}).get("Races", []):
                rnd = int(race.get("round", 0))
                for sp in race.get("SprintResults", []):
                    did = sp.get("Driver", {}).get("driverId", "")
                    key = (rnd, did)
                    if key not in rows:
                        continue
                    pt = sp.get("positionText", "")
                    rows[key]["sprint_position"] = int(pt) if pt.isdigit() else 0
                    rows[key]["sprint_points"] = float(sp.get("points", 0) or 0)
                    rows[key]["has_sprint"] = 1
    except JolpicaError:
        pass

    df = pd.DataFrame(list(rows.values()))
    # Fallback: if grid is 0 but we have a qualifying position, use it.
    mask = (df["grid"] == 0) & (df["quali_position"] > 0)
    df.loc[mask, "grid"] = df.loc[mask, "quali_position"]
    log.info(f"jolpica: {season} -> {len(df)} entries across {df['round'].nunique()} rounds")
    return _coerce(df)


def fetch_schedule(season: int, base_url: str = DEFAULT_BASE, timeout: int = 25) -> pd.DataFrame:
    """Fetch the race calendar (no results) — used for future-season prediction.

    Includes a data-driven ``has_sprint`` flag: the Ergast/jolpica schedule carries a
    ``Sprint`` session block on sprint weekends, so we can tell *before a weekend runs*
    whether it features a sprint (used to seed future-race prediction and to validate
    ``f1predict predict <race> sprint``).
    """
    payload = _get(f"{base_url}/{season}.json?limit=100", timeout)
    races = payload.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    out = []
    for race in races:
        circ = race.get("Circuit", {})
        loc = circ.get("Location", {})
        out.append({
            "season": season, "round": int(race.get("round", 0)),
            "race_name": race.get("raceName", ""), "circuit_id": circ.get("circuitId", ""),
            "country": loc.get("country", ""), "locality": loc.get("locality", ""),
            "date": race.get("date", ""),
            "has_sprint": 1 if race.get("Sprint") else 0,
        })
    return pd.DataFrame(out)


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    for col, dtype in RAW_COLUMNS.items():
        if col not in df:
            df[col] = 0 if dtype in ("int32", "float32") else ""
        if dtype == "int32":
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int32")
        elif dtype == "float32":
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype("float32")
        else:
            df[col] = df[col].astype("string").fillna("")
    return df[list(RAW_COLUMNS)]
