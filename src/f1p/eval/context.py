"""Event contexts: everything a model may know about an event at a checkpoint.

A context is built from the lake's curated tables, filtered through
:func:`f1p.core.clock.known_at` at the checkpoint's ``as_of`` — the single place where the
future is cut off for every model.

Checkpoints (plan §7.0):

* ``C1`` — Thursday of race week (pre-weekend): entry list, history, standings.
* ``C2`` — after the last practice session, just before qualifying: adds practice (and, on
  sprint weekends where they come first, sprint qualifying and the sprint).
* ``C4`` — after qualifying, before the race: adds qualifying and the grid.
* ``C5`` — in the race, after lap ``k`` (one third of the distance): the tables as at the
  start, plus the race's own laps ``1..k`` (:mod:`f1p.eval.in_race`). Only events with lap
  timing have a ``C5`` context, and it is asked for explicitly (not a default checkpoint).
  ``C5a``, ``C5b`` and ``C5c`` are the same at a quarter, half and two thirds of the
  distance (:data:`IN_RACE_FRACTIONS`).

Two **targets** are forecast: the race finishing order (at C1 and C4; the in-race
checkpoints on request) and the qualifying order (at C1 and C2).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, Protocol

import polars as pl

from f1p.core.clock import AsOf, known_at
from f1p.eval.metrics import Outcome

if TYPE_CHECKING:
    from f1p.eval.forecast import Forecast
from f1p.lake import timing

Tables = dict[str, pl.DataFrame]
CHECKPOINTS = ("C1", "C4")  # the race target's checkpoints (the default everywhere)
TARGETS = ("race", "quali")
TARGET_CHECKPOINTS: dict[str, tuple[str, ...]] = {"race": ("C1", "C4"), "quali": ("C1", "C2")}
# in-race checkpoints (race target) → the share of the race distance run at the cut; C5
# (one third) was the first and stays as it was, so its results remain comparable
IN_RACE_FRACTIONS: dict[str, float] = {
    "C5a": 1.0 / 4.0,
    "C5": 1.0 / 3.0,
    "C5b": 1.0 / 2.0,
    "C5c": 2.0 / 3.0,
}
IN_RACE = tuple(IN_RACE_FRACTIONS)
IN_RACE_FRACTION = IN_RACE_FRACTIONS["C5"]
MIN_RACE_LAPS = 10  # shorter races (abandoned, red-flagged early) have no in-race checkpoint


def is_in_race(checkpoint: str) -> bool:
    return checkpoint in IN_RACE_FRACTIONS


def cut_lap(race_laps: int, fraction: float = IN_RACE_FRACTION) -> int:
    """``k``: the lap after which an in-race forecast is made (at least lap 2, before the
    last lap)."""
    return int(min(max(2, math.ceil(fraction * race_laps)), max(race_laps - 1, 2)))


def checkpoint_lap(checkpoint: str, race_laps: int) -> int:
    """``k`` of an in-race ``checkpoint`` in a race of ``race_laps`` laps."""
    return cut_lap(race_laps, IN_RACE_FRACTIONS[checkpoint])


def checkpoint_time(
    checkpoint: str,
    race_date: date,
    race_start: datetime | None,
    quali_start: datetime | None = None,
) -> datetime:
    if checkpoint == "C1":
        return timing.pre_weekend_as_of(race_date)
    if checkpoint == "C2":
        return timing.pre_quali_as_of(race_date, quali_start)
    if checkpoint in ("C4", *IN_RACE):  # in the race: the tables as they were at the start
        return timing.pre_race_as_of(race_date, race_start)
    raise ValueError(
        f"unknown checkpoint {checkpoint!r}; choose from C1, C2, C4, {', '.join(IN_RACE)}"
    )


def target_checkpoints(target: str) -> tuple[str, ...]:
    if target not in TARGET_CHECKPOINTS:
        raise ValueError(f"unknown target {target!r}; choose from {TARGETS}")
    return TARGET_CHECKPOINTS[target]


def quali_starts(tables: Tables) -> dict[str, datetime]:
    """Exact qualifying start times (UTC) where F1DB has them."""
    if "sessions" not in tables:
        return {}
    s = tables["sessions"].filter(
        (pl.col("session") == "quali") & pl.col("start_utc").is_not_null()
    )
    return {str(e): t for e, t in s.select("event_id", "start_utc").iter_rows()}


@dataclass
class EventContext:
    event_id: str
    season: int
    round: int
    circuit_id: str
    race_date: date
    checkpoint: str
    as_of: datetime
    drivers: tuple[str, ...]
    constructors: tuple[str, ...]
    tables: Tables = field(repr=False)
    meta: dict[str, Any] = field(default_factory=dict)
    target: str = "race"
    _cache: dict[str, pl.DataFrame] = field(default_factory=dict, repr=False)

    @property
    def key(self) -> tuple[str, str]:
        return (self.event_id, self.checkpoint)

    @property
    def n(self) -> int:
        return len(self.drivers)

    def known(self, name: str) -> pl.DataFrame:
        """``name`` as it could be seen at ``as_of`` (tables without ``observed_at`` are static)."""
        if name not in self._cache:
            df = self.tables[name]
            if "observed_at" in df.columns:
                df = known_at(df, AsOf(self.as_of))
            self._cache[name] = df
        return self._cache[name]

    def index(self) -> dict[str, int]:
        return {d: i for i, d in enumerate(self.drivers)}


def outcome_for(
    tables: Tables, event_id: str, target: str = "race"
) -> tuple[Outcome, datetime | None]:
    """The order to score against: the race classification, or the qualifying order."""
    if target == "quali":
        q = (
            tables["qualifying"]
            .filter(
                (pl.col("event_id") == event_id)
                & (pl.col("session") == "quali")
                & pl.col("position").is_not_null()
            )
            .sort("position")
        )
        known = q["observed_at"].max() if q.height else None
        return Outcome(event_id, tuple(q["driver_id"].to_list())), known  # type: ignore[return-value]
    r = (
        tables["results"]
        .filter(
            (pl.col("event_id") == event_id) & (pl.col("session") == "race") & pl.col("classified")
        )
        .sort("position")
    )
    known_at_time = r["observed_at"].max() if r.height else None
    return Outcome(event_id, tuple(r["driver_id"].to_list())), known_at_time  # type: ignore[return-value]


def _field(tables: Tables, ev: dict[str, Any], checkpoint: str, as_of: datetime) -> pl.DataFrame:
    """Drivers expected to race, as knowable at ``as_of``."""
    starters = (
        tables["results"]
        .filter(
            (pl.col("event_id") == ev["event_id"])
            & (pl.col("session") == "race")
            & ~pl.col("status").is_in(["dns", "dnq", "dnpq"])
        )
        .select("driver_id", "constructor_id")
    )
    if checkpoint == "C4" or is_in_race(checkpoint):
        grid = known_at(
            tables["grid"].filter(
                (pl.col("event_id") == ev["event_id"]) & (pl.col("session") == "race")
            ),
            AsOf(as_of),
        ).select("driver_id", "constructor_id")
        if grid.height:
            return grid.unique("driver_id", keep="first", maintain_order=True)
    entries = (
        tables["entries"]
        .filter(
            (pl.col("season") == ev["season"])
            & (pl.col("round") == ev["round"])
            & ~pl.col("test_driver")
        )
        .select("driver_id", "constructor_id")
    )
    if entries.height:
        return entries.unique("driver_id", keep="first", maintain_order=True)
    if starters.height:
        return starters.unique("driver_id", keep="first", maintain_order=True)
    # A future round with no entry list yet: assume the latest line-up of the season.
    season = tables["entries"].filter(
        (pl.col("season") == ev["season"])
        & pl.col("round").is_not_null()
        & (pl.col("round") < ev["round"])
        & ~pl.col("test_driver")
    )
    if season.is_empty():
        return season.select("driver_id", "constructor_id")
    latest = season.filter(pl.col("round") == season["round"].max())
    return latest.select("driver_id", "constructor_id").unique(
        "driver_id", keep="first", maintain_order=True
    )


def live_context(
    tables: Tables,
    event_id: str,
    checkpoint: str,
    as_of: datetime | None = None,
    *,
    lineup: pl.DataFrame | None = None,
    target: str = "race",
    lap: int | None = None,
) -> EventContext:
    """Context for any event — including one not yet raced — at ``checkpoint``.

    ``as_of`` defaults to the checkpoint time, or now if that is still in the future (the
    model then sees exactly what is public today). ``lineup`` (``driver_id``,
    ``constructor_id``) overrides the field — a historical projection uses the line-up known
    at ``as_of`` rather than the entry list the round eventually had. In the race (``C5``
    and the others of :data:`IN_RACE_FRACTIONS`), ``lap`` is the last lap completed (default:
    the checkpoint's share of the distance).
    """
    from datetime import UTC

    rows = tables["events"].filter(pl.col("event_id") == event_id)
    if rows.is_empty():
        raise KeyError(f"unknown event {event_id!r}")
    ev = rows.row(0, named=True)
    cp_time = checkpoint_time(
        checkpoint, ev["race_date"], ev["race_start_utc"], quali_starts(tables).get(event_id)
    )
    if as_of is None:
        as_of = min(cp_time, datetime.now(UTC))
    fld = _field(tables, ev, checkpoint, as_of) if lineup is None else lineup
    if fld.is_empty():
        raise ValueError(f"no known field for {event_id}")
    meta: dict[str, Any] = {"live": 1, "checkpoint_time": cp_time.isoformat()}
    if is_in_race(checkpoint):
        race_laps = int(ev["laps"] or 0)
        k = lap if lap is not None else checkpoint_lap(checkpoint, race_laps)
        meta |= {"lap": k, "race_laps": race_laps}
    return EventContext(
        event_id=ev["event_id"],
        season=ev["season"],
        round=ev["round"],
        circuit_id=ev["circuit_id"],
        race_date=ev["race_date"],
        checkpoint=checkpoint,
        as_of=as_of,
        drivers=tuple(fld["driver_id"].to_list()),
        constructors=tuple(fld["constructor_id"].to_list()),
        tables=tables,
        meta=meta,
        target=target,
    )


def next_event(tables: Tables, now: datetime | None = None) -> str | None:
    """The first event whose race has no classification yet."""
    raced = set(
        tables["results"].filter(pl.col("session") == "race")["event_id"].unique().to_list()
    )
    upcoming = tables["events"].filter(~pl.col("event_id").is_in(list(raced)))
    if now is not None:
        upcoming = upcoming.filter(pl.col("race_date") >= now.date())
    if upcoming.is_empty():
        return None
    return str(upcoming.sort("season", "round")["event_id"][0])


def build_contexts(
    tables: Tables,
    *,
    seasons: tuple[int, int],
    checkpoints: tuple[str, ...] | None = None,
    target: str = "race",
) -> list[tuple[EventContext, Outcome, datetime | None]]:
    """Contexts (with their outcomes) for every event with a ``target`` result in
    ``seasons``, at ``checkpoints`` (default: the target's; an empty tuple gives none)."""
    checkpoints = target_checkpoints(target) if checkpoints is None else checkpoints
    lo, hi = seasons
    events = tables["events"].filter(pl.col("season").is_between(lo, hi)).sort("season", "round")
    starts = quali_starts(tables)
    timed = _timed_laps(tables, (lo, hi)) if any(map(is_in_race, checkpoints)) else {}
    out = []
    for ev in events.iter_rows(named=True):
        outcome, outcome_known = outcome_for(tables, ev["event_id"], target)
        if not outcome.classified:
            continue  # not yet raced (or no classification) — nothing to score
        for cp in checkpoints:
            meta: dict[str, Any] = {}
            if is_in_race(cp):
                if target != "race":
                    continue
                race_laps = int(ev["laps"] or 0)
                k = checkpoint_lap(cp, race_laps)
                # the race must have run long enough and have timing at lap k
                if race_laps < MIN_RACE_LAPS or timed.get(ev["event_id"], 0) < k:
                    continue
                meta = {"lap": k, "race_laps": race_laps}
            as_of = checkpoint_time(
                cp, ev["race_date"], ev["race_start_utc"], starts.get(ev["event_id"])
            )
            fld = _field(tables, ev, cp, as_of)
            drivers = fld["driver_id"].to_list()
            constructors = fld["constructor_id"].to_list()
            # Guarantee every classified finisher has a forecast (late substitutes, gaps in
            # entry lists); recorded so evaluation reports can show how often it happens.
            added = [d for d in outcome.classified if d not in set(drivers)]
            if added:
                src = tables["qualifying"] if target == "quali" else tables["results"]
                res = src.filter(
                    (pl.col("event_id") == ev["event_id"])
                    & pl.col("driver_id").is_in(added)
                    & (pl.col("session") == ("quali" if target == "quali" else "race"))
                )
                lookup = dict(res.select("driver_id", "constructor_id").iter_rows())
                drivers += added
                constructors += [lookup.get(d, "") for d in added]
            ctx = EventContext(
                event_id=ev["event_id"],
                season=ev["season"],
                round=ev["round"],
                circuit_id=ev["circuit_id"],
                race_date=ev["race_date"],
                checkpoint=cp,
                as_of=as_of,
                drivers=tuple(drivers),
                constructors=tuple(constructors),
                tables=tables,
                meta={"field_added": len(added), **meta},
                target=target,
            )
            out.append((ctx, outcome, outcome_known))
    return out


def _timed_laps(tables: Tables, seasons: tuple[int, int]) -> dict[str, int]:
    """Per event: the last lap for which at least half the starters have a timing record."""
    if "laps" not in tables:
        return {}
    lo, hi = seasons
    ids = tables["events"].filter(pl.col("season").is_between(lo, hi))["event_id"]
    laps = tables["laps"].filter(pl.col("event_id").is_in(ids.implode()))
    if laps.is_empty():
        return {}
    per_lap = laps.group_by("event_id", "lap").len()
    field_size = per_lap.group_by("event_id").agg(pl.col("len").max().alias("cars"))
    ok = per_lap.join(field_size, on="event_id").filter(pl.col("len") * 2 >= pl.col("cars"))
    return dict(ok.group_by("event_id").agg(pl.col("lap").max()).iter_rows())


class Forecaster(Protocol):
    name: str
    checkpoints: frozenset[str]

    def prepare(self, ctx: EventContext) -> Any: ...

    def forecast(
        self, ctx: EventContext, signal: Any, history: Sequence[tuple[EventContext, Any, Outcome]]
    ) -> Forecast | None: ...
