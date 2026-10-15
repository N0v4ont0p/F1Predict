"""Jolpica (the Ergast successor) — lap times, pit stops and car numbers per race.

Used where the F1 live-timing archive is unavailable (the whole 2022 season is not served).
Requests go through the shared :class:`~f1p.sources.http.HttpClient`: the host's published
limits (4/s, 500/h) are enforced and every page is cached, so a season costs about 300
requests once.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from f1p.sources.http import HttpClient

BASE = "https://api.jolpi.ca/ergast/f1"
PAGE = 100


def lap_seconds(value: str) -> float | None:
    if not value:
        return None
    try:
        parts = value.split(":")
        return int(parts[0]) * 60 + float(parts[1]) if len(parts) == 2 else float(parts[0])
    except ValueError:
        return None


def _pages(http: HttpClient, path: str) -> Iterator[dict[str, Any]]:
    offset = 0
    while True:
        data = http.get_json(f"{BASE}/{path}", {"limit": PAGE, "offset": offset})["MRData"]
        yield data
        offset += PAGE
        if offset >= int(data.get("total", 0)):
            return


def _races(data: dict[str, Any]) -> list[dict[str, Any]]:
    return data.get("RaceTable", {}).get("Races", [])


def laps(http: HttpClient, season: int, rnd: int) -> list[tuple[str, int, float]]:
    """``(driverId, lap, seconds)`` for every timed lap of the race."""
    out: dict[tuple[str, int], float] = {}
    for data in _pages(http, f"{season}/{rnd}/laps.json"):
        for race in _races(data):
            for lap in race.get("Laps", []):
                n = int(lap["number"])
                for t in lap.get("Timings", []):
                    secs = lap_seconds(t.get("time", ""))
                    if secs is not None:
                        out[(t["driverId"], n)] = secs
    return [(d, n, s) for (d, n), s in sorted(out.items(), key=lambda kv: (kv[0][1], kv[0][0]))]


def pit_laps(http: HttpClient, season: int, rnd: int) -> dict[str, set[int]]:
    """Laps on which each driver entered the pit lane (in-laps)."""
    out: dict[str, set[int]] = {}
    for data in _pages(http, f"{season}/{rnd}/pitstops.json"):
        for race in _races(data):
            for stop in race.get("PitStops", []):
                out.setdefault(stop["driverId"], set()).add(int(stop["lap"]))
    return out


def car_numbers(http: HttpClient, season: int, rnd: int) -> dict[str, str]:
    """Jolpica driverId → car number (to join with F1DB's results)."""
    out: dict[str, str] = {}
    for data in _pages(http, f"{season}/{rnd}/results.json"):
        for race in _races(data):
            for r in race.get("Results", []):
                out[r["Driver"]["driverId"]] = str(r["number"])
    return out
