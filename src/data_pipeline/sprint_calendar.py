"""Sprint-weekend awareness — *which* Grands Prix run a Saturday sprint.

The system must know whether a given weekend has a sprint so it can (a) seed future-race
prediction correctly and (b) refuse a ``predict <race> sprint`` request for a non-sprint
weekend. Knowledge is resolved in priority order, most-authoritative first:

1. **Master dataset** — completed weekends carry a data-driven ``has_sprint`` flag set by
   the jolpica client when the API returned sprint results.
2. **Live schedule** — :func:`~f1predict.data_pipeline.jolpica.fetch_schedule` reads the
   ``Sprint`` session block, so upcoming weekends are known the moment the calendar is out.
3. **Hardcoded fallback** — a small built-in set of confirmed sprint venues per season, used
   only when offline and the weekend isn't in the data yet. The schedule always wins.
"""
from __future__ import annotations

import pandas as pd

from ..config import Config, load_config
from ..utils.logging import get_logger

log = get_logger()

# Best-effort fallback of confirmed sprint venues (canonical circuit_id) per season. This is
# *only* consulted when neither the master dataset nor the live schedule can answer; the
# data-driven sources always take precedence, so a slightly stale list here is harmless.
_SPRINT_FALLBACK: dict[int, set[str]] = {
    2024: {"shanghai", "miami", "imola", "americas", "interlagos", "losail"},
    2025: {"shanghai", "miami", "spa", "americas", "interlagos", "losail"},
    2026: {"shanghai", "miami", "gilles_villeneuve", "silverstone", "zandvoort", "marina_bay"},
}

# In-process cache of fetched schedules so repeated lookups don't re-hit the network.
_SCHEDULE_CACHE: dict[int, pd.DataFrame] = {}


def clear_schedule_cache() -> None:
    """Drop the cached calendars (used by the shell ``/reload``)."""
    _SCHEDULE_CACHE.clear()


def _schedule(season: int, cfg: Config) -> pd.DataFrame | None:
    if season in _SCHEDULE_CACHE:
        return _SCHEDULE_CACHE[season]
    try:
        from .jolpica import fetch_schedule
        cal = fetch_schedule(season, cfg.get("data.jolpica_base_url"),
                             int(cfg.get("data.request_timeout")))
    except Exception as exc:  # pragma: no cover - offline / future-unknown
        log.warning(f"could not fetch {season} calendar for sprint lookup: {exc}")
        return None
    if cal is not None and len(cal):
        _SCHEDULE_CACHE[season] = cal
    return cal


def sprint_rounds(season: int, cfg: Config | None = None,
                  master: pd.DataFrame | None = None) -> list[dict]:
    """Return the sprint weekends for ``season`` as ``[{round, circuit_id, race_name}]``.

    Prefers real results in the master, then the live schedule, then the fallback set.
    Sorted by round where known. Used to power helpful error messages.
    """
    cfg = cfg or load_config()
    if master is None:
        try:
            from .build import load_master
            master = load_master(cfg)
        except Exception:  # pragma: no cover - master may be absent
            master = None

    rows: dict[int, dict] = {}
    if master is not None and len(master):
        sub = master[(master["season"] == season) & (master["has_sprint"] == 1)]
        for _, r in sub[["round", "circuit_id", "race_name"]].drop_duplicates().iterrows():
            rows[int(r["round"])] = {
                "round": int(r["round"]), "circuit_id": str(r["circuit_id"]),
                "race_name": str(r["race_name"]),
            }

    if not rows:
        cal = _schedule(season, cfg)
        if cal is not None and "has_sprint" in cal:
            sub = cal[cal["has_sprint"] == 1]
            for _, r in sub.iterrows():
                rows[int(r["round"])] = {
                    "round": int(r["round"]), "circuit_id": str(r["circuit_id"]),
                    "race_name": str(r.get("race_name", r["circuit_id"])),
                }

    if not rows:
        for cid in sorted(_SPRINT_FALLBACK.get(season, set())):
            rows[len(rows)] = {"round": None, "circuit_id": cid, "race_name": cid}

    return [rows[k] for k in sorted(rows, key=lambda x: (x is None, x))]


def weekend_has_sprint(
    season: int,
    rnd: int | None = None,
    circuit_id: str | None = None,
    cfg: Config | None = None,
    master: pd.DataFrame | None = None,
) -> bool:
    """Return ``True`` if the identified weekend runs a sprint.

    Identify the weekend by ``rnd`` and/or ``circuit_id``. Resolution order: master dataset
    (real, completed) → live schedule (upcoming) → hardcoded fallback (offline).
    """
    cfg = cfg or load_config()
    if rnd is None and circuit_id is None:
        return False

    if master is None:
        try:
            from .build import load_master
            master = load_master(cfg)
        except Exception:  # pragma: no cover
            master = None

    # 1) master dataset (completed weekends).
    if master is not None and len(master):
        sub = master[master["season"] == season]
        if rnd is not None:
            sub = sub[sub["round"] == rnd]
        if circuit_id is not None:
            sub = sub[sub["circuit_id"] == circuit_id]
        if len(sub):
            return bool((sub["has_sprint"] == 1).any())

    # 2) live schedule (upcoming weekends).
    cal = _schedule(season, cfg)
    if cal is not None and "has_sprint" in cal:
        sub = cal
        if rnd is not None:
            sub = sub[sub["round"] == rnd]
        if circuit_id is not None:
            sub = sub[sub["circuit_id"] == circuit_id]
        if len(sub):
            return bool((sub["has_sprint"] == 1).any())

    # 3) hardcoded fallback (offline, circuit-keyed only).
    if circuit_id is not None:
        return circuit_id in _SPRINT_FALLBACK.get(season, set())
    return False
