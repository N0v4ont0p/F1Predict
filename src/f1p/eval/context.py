"""Event contexts: everything a model may know about an event at a checkpoint.

A context is built from the lake's curated tables, filtered through
:func:`f1p.core.clock.known_at` at the checkpoint's ``as_of`` — the single place where the
future is cut off for every model.

Checkpoints (plan §7.0):

* ``C1`` — Thursday of race week (pre-weekend): entry list, history, standings.
* ``C4`` — after qualifying, before the race: adds qualifying and the grid.
"""

from __future__ import annotations

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
CHECKPOINTS = ("C1", "C4")


def checkpoint_time(checkpoint: str, race_date: date, race_start: datetime | None) -> datetime:
    if checkpoint == "C1":
        return timing.pre_weekend_as_of(race_date)
    if checkpoint == "C4":
        return timing.pre_race_as_of(race_date, race_start)
    raise ValueError(f"unknown checkpoint {checkpoint!r}; choose from {CHECKPOINTS}")


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


def outcome_for(tables: Tables, event_id: str) -> tuple[Outcome, datetime | None]:
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
    if checkpoint == "C4":
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
    tables: Tables, event_id: str, checkpoint: str, as_of: datetime | None = None
) -> EventContext:
    """Context for any event — including one not yet raced — at ``checkpoint``.

    ``as_of`` defaults to the checkpoint time, or now if that is still in the future (the
    model then sees exactly what is public today).
    """
    from datetime import UTC

    rows = tables["events"].filter(pl.col("event_id") == event_id)
    if rows.is_empty():
        raise KeyError(f"unknown event {event_id!r}")
    ev = rows.row(0, named=True)
    cp_time = checkpoint_time(checkpoint, ev["race_date"], ev["race_start_utc"])
    if as_of is None:
        as_of = min(cp_time, datetime.now(UTC))
    fld = _field(tables, ev, checkpoint, as_of)
    if fld.is_empty():
        raise ValueError(f"no known field for {event_id}")
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
        meta={"live": 1, "checkpoint_time": cp_time.isoformat()},
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
    checkpoints: tuple[str, ...] = CHECKPOINTS,
) -> list[tuple[EventContext, Outcome, datetime | None]]:
    """Contexts (with their outcomes) for every event with a race result in ``seasons``."""
    lo, hi = seasons
    events = tables["events"].filter(pl.col("season").is_between(lo, hi)).sort("season", "round")
    out = []
    for ev in events.iter_rows(named=True):
        outcome, outcome_known = outcome_for(tables, ev["event_id"])
        if not outcome.classified:
            continue  # not yet raced (or no classification) — nothing to score
        for cp in checkpoints:
            as_of = checkpoint_time(cp, ev["race_date"], ev["race_start_utc"])
            fld = _field(tables, ev, cp, as_of)
            drivers = fld["driver_id"].to_list()
            constructors = fld["constructor_id"].to_list()
            # Guarantee every classified finisher has a forecast (late substitutes, gaps in
            # entry lists); recorded so evaluation reports can show how often it happens.
            added = [d for d in outcome.classified if d not in set(drivers)]
            if added:
                res = tables["results"].filter(
                    (pl.col("event_id") == ev["event_id"])
                    & pl.col("driver_id").is_in(added)
                    & (pl.col("session") == "race")
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
                meta={"field_added": len(added)},
            )
            out.append((ctx, outcome, outcome_known))
    return out


class Forecaster(Protocol):
    name: str
    checkpoints: frozenset[str]

    def prepare(self, ctx: EventContext) -> Any: ...

    def forecast(
        self, ctx: EventContext, signal: Any, history: Sequence[tuple[EventContext, Any, Outcome]]
    ) -> Forecast | None: ...
