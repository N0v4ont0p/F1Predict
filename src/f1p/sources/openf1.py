"""OpenF1 (api.openf1.org): tyre stints per race, 2023 →.

Used where the live-timing archive withholds its tyre feed (``TimingAppData`` answers 401 for
most seasons). Requests go through the shared rate-limited, caching
:class:`~f1p.sources.http.HttpClient`.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime

from f1p.sources.http import HttpClient

BASE = "https://api.openf1.org/v1"
FIRST_SEASON = 2023

Stint = tuple[int, int, str | None, int]  # (lap_start, lap_end, compound, tyre age at start)


def race_sessions(http: HttpClient, season: int) -> dict[date, int]:
    """Race-day date → OpenF1 session key for every Grand Prix of ``season``."""
    rows = http.get_json(f"{BASE}/sessions", {"year": season, "session_name": "Race"})
    out: dict[date, int] = {}
    for r in rows if isinstance(rows, list) else []:
        start = r.get("date_start")
        if start and r.get("session_key") is not None:
            out[datetime.fromisoformat(start).date()] = int(r["session_key"])
    return out


def stints(http: HttpClient, session_key: int) -> dict[str, list[Stint]]:
    """Car number → its stints in order."""
    rows = http.get_json(f"{BASE}/stints", {"session_key": session_key})
    out: dict[str, list[Stint]] = {}
    for r in sorted(
        rows if isinstance(rows, list) else [],
        key=lambda x: (x.get("driver_number") or 0, x.get("stint_number") or 0),
    ):
        if r.get("lap_start") is None or r.get("driver_number") is None:
            continue
        comp = r.get("compound")
        out.setdefault(str(r["driver_number"]), []).append(
            (
                int(r["lap_start"]),
                int(r.get("lap_end") or r["lap_start"]),
                comp.upper() if isinstance(comp, str) else None,
                int(r.get("tyre_age_at_start") or 0),
            )
        )
    return out


def tyre_on_lap(car_stints: Sequence[Stint], lap: int) -> tuple[int, str | None, int] | None:
    """``(stint number, compound, tyre age after the lap)`` for ``lap``.

    OpenF1 stints share their boundary lap (the pit lap ends one stint and starts the next);
    that lap is run on the old set, so a lap belongs to the latest stint that began before
    it (the first stint also owns lap 1)."""
    if not car_stints:
        return None
    k = 0
    for i, (start, _, _, _) in enumerate(car_stints):
        if start < lap or (i == 0 and start <= lap):
            k = i
    start, _, comp, age0 = car_stints[k]
    age = age0 + (lap - start) + (1 if k == 0 else 0)
    return k, comp, max(age, 1)
