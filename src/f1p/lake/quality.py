"""Data-quality gates (plan §6.4). Gates fail the pipeline; they never warn silently.

Each gate returns a :class:`GateResult` with the offending rows, so a failure explains itself.
The suite runs on curated tables *before* they are committed to the lake.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import polars as pl

from f1p.core import points as P
from f1p.core.status import UnknownStatusError, classify_status

POINTS_FROM_SEASON = P.EARLIEST_SUPPORTED_SEASON
MIN_SEQUENTIAL_GAP_DAYS = 6


@dataclass(frozen=True, slots=True)
class GateResult:
    name: str
    passed: bool
    summary: str
    failures: pl.DataFrame = field(default_factory=pl.DataFrame)

    def __str__(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        return f"[{mark}] {self.name}: {self.summary}"


class QualityError(RuntimeError):
    def __init__(self, results: list[GateResult]):
        failed = [r for r in results if not r.passed]
        super().__init__("data-quality gates failed:\n" + "\n".join(str(r) for r in failed))
        self.results = results


Tables = dict[str, pl.DataFrame]


def _result(name: str, failures: pl.DataFrame, ok: str, bad: str) -> GateResult:
    passed = failures.height == 0
    return GateResult(name, passed, ok if passed else bad.format(n=failures.height), failures)


# --------------------------------------------------------------------------- gates
def classification_integrity(t: Tables) -> GateResult:
    """Classified positions in each race/sprint are unique and run 1..k.

    Shared drives (1950s) are excluded. A post-race disqualification can leave a position
    officially vacant (e.g. 1983 Brazil, P2), so gaps are allowed up to the number of
    disqualified cars in that classification — never more.
    """
    res = t["results"].filter(~pl.col("shared_car"))
    dsq = (
        res.filter(pl.col("status") == "dsq")
        .group_by("event_id", "session")
        .agg(pl.len().alias("dsq"))
    )
    r = res.filter(pl.col("classified"))
    per = (
        r.group_by("event_id", "session")
        .agg(
            pl.len().alias("n"),
            pl.col("position").n_unique().alias("unique"),
            pl.col("position").min().alias("lo"),
            pl.col("position").max().alias("hi"),
        )
        .join(dsq, on=["event_id", "session"], how="left")
        .with_columns(pl.col("dsq").fill_null(0))
    )
    bad = per.filter(
        (pl.col("unique") != pl.col("n"))
        | (pl.col("lo") != 1)
        | (pl.col("hi") > pl.col("n") + pl.col("dsq"))
    ).sort("event_id")
    return _result(
        "classification_integrity",
        bad,
        f"{per.height} classifications are contiguous 1..k",
        "{n} classification(s) have gaps or duplicate positions",
    )


def status_taxonomy(t: Tables) -> GateResult:
    """Every retirement reason maps to a status class in strict mode (plan Appendix B)."""
    reasons = t["results"]["reason_retired"].drop_nulls().unique().to_list()
    unknown = []
    for reason in reasons:
        try:
            classify_status(reason, strict=True)
        except UnknownStatusError:
            unknown.append(reason)
    failures = pl.DataFrame(
        {"reason_retired": sorted(unknown)}, schema={"reason_retired": pl.String}
    )
    return _result(
        "status_taxonomy",
        failures,
        f"all {len(reasons)} retirement reasons map to a status class",
        "{n} retirement reason(s) are not in the taxonomy",
    )


def points_reconciliation(t: Tables) -> GateResult:
    """Recompute every race and sprint score since 1991 and compare with the official points."""
    ev = t["events"].select("event_id", "circuit_id", "distance_fraction")
    r = t["results"].filter(pl.col("season") >= POINTS_FROM_SEASON).join(ev, on="event_id")

    def expected(row: dict[str, Any]) -> float:
        season: int = row["season"]
        if row["session"] == "sprint":
            session, cond = P.Session.SPRINT, P.FULL_RACE
        else:
            session = P.Session.RACE
            cond = P.race_conditions(season, row["circuit_id"], row["distance_fraction"])
        return P.points_for(
            row["position"],
            season,
            session,
            cond,
            classified=row["classified"],
            fastest_lap=row["fastest_lap"],
        )

    r = r.with_columns(
        pl.struct(
            "season",
            "session",
            "circuit_id",
            "distance_fraction",
            "position",
            "classified",
            "fastest_lap",
        )
        .map_elements(expected, return_dtype=pl.Float64)
        .alias("expected")
    )
    bad = r.filter((pl.col("expected") - pl.col("points")).abs() > 1e-9).select(
        "event_id", "session", "position", "driver_id", "points", "expected"
    )
    return _result(
        "points_reconciliation",
        bad,
        f"{r.height} official scores since {POINTS_FROM_SEASON} reproduced exactly",
        "{n} official score(s) differ from the points rules",
    )


# Championship tables that deliberately differ from summed race points.
STANDINGS_EXCEPTIONS: frozenset[tuple[int, str]] = frozenset(
    {(1997, "michael_schumacher")},  # excluded from the 1997 championship
)


def standings_reconciliation(t: Tables) -> GateResult:
    """Each round's driver standings equal cumulative official race + sprint points (1991+)."""
    pts = (
        t["results"]
        .filter(pl.col("season") >= POINTS_FROM_SEASON)
        .group_by("season", "round", "driver_id")
        .agg(pl.col("points").sum())
        .sort("season", "round")
    )
    st = t["driver_standings"].filter(pl.col("season") >= POINTS_FROM_SEASON)
    # Cumulative points per driver up to and including each round with a standings table.
    rounds = st.select("season", "round").unique()
    expanded = (
        pts.join(rounds, on="season", suffix="_at")
        .filter(pl.col("round") <= pl.col("round_at"))
        .group_by("season", "round_at", "driver_id")
        .agg(pl.col("points").sum().alias("expected"))
        .rename({"round_at": "round"})
    )
    joined = st.join(expanded, on=["season", "round", "driver_id"], how="left").with_columns(
        pl.col("expected").fill_null(0.0)
    )
    exceptions = pl.DataFrame(
        list(STANDINGS_EXCEPTIONS),
        schema={"season": pl.Int32, "driver_id": pl.String},
        orient="row",
    )
    joined = joined.join(exceptions, on=["season", "driver_id"], how="anti")
    bad = joined.filter((pl.col("expected") - pl.col("points")).abs() > 1e-9).select(
        "event_id", "driver_id", "points", "expected"
    )
    return _result(
        "standings_reconciliation",
        bad,
        f"{joined.height} standings rows equal cumulative race points",
        "{n} standings row(s) differ from cumulative race points",
    )


def results_coverage(t: Tables, now: datetime | None = None) -> GateResult:
    """Every event whose race day is more than two days past has a race classification."""
    now = now or datetime.now(UTC)
    cutoff = (now - timedelta(days=2)).date()
    with_results = t["results"].filter(pl.col("session") == "race").select("event_id").unique()
    past = t["events"].filter(pl.col("race_date") <= cutoff).select("event_id", "race_date")
    missing = past.join(with_results, on="event_id", how="anti").sort("event_id")
    return _result(
        "results_coverage",
        missing,
        f"all {past.height} past events have race results",
        "{n} past event(s) have no race results",
    )


def knowability_order(t: Tables) -> GateResult:
    """Qualifying is public before the race starts; race results before the next event's C1."""
    from f1p.lake import timing

    ev = (
        t["events"]
        .select("event_id", "season", "round", "race_date", "race_start_utc")
        .sort("season", "round")
    )
    pre_race = ev.with_columns(
        pl.struct("race_date", "race_start_utc")
        .map_elements(
            lambda r: timing.pre_race_as_of(r["race_date"], r["race_start_utc"]),
            return_dtype=pl.Datetime("us", "UTC"),
        )
        .alias("c4"),
        pl.col("race_date")
        .shift(-1)
        .map_elements(timing.pre_weekend_as_of, return_dtype=pl.Datetime("us", "UTC"))
        .alias("next_c1"),
        (pl.col("race_date").shift(-1) - pl.col("race_date")).dt.total_days().alias("gap_days"),
    )
    # The 1950-60 Indianapolis 500 counted for the championship but ran within days of other
    # Grands Prix with a separate field; events < 6 days apart are concurrent, not sequential.
    pre_race = pre_race.with_columns(
        pl.when(pl.col("gap_days") >= MIN_SEQUENTIAL_GAP_DAYS)
        .then(pl.col("next_c1"))
        .alias("next_c1")
    )
    q = (
        t["qualifying"]
        .filter(pl.col("session") == "quali")
        .group_by("event_id")
        .agg(pl.col("observed_at").max().alias("quali_obs"))
    )
    r = (
        t["results"]
        .filter(pl.col("session") == "race")
        .group_by("event_id")
        .agg(pl.col("observed_at").max().alias("race_obs"))
    )
    j = pre_race.join(q, on="event_id", how="left").join(r, on="event_id", how="left")
    bad = j.filter(
        (pl.col("quali_obs") > pl.col("c4"))
        | (pl.col("race_obs") <= pl.col("c4"))
        | (pl.col("next_c1").is_not_null() & (pl.col("race_obs") > pl.col("next_c1")))
    ).select("event_id", "quali_obs", "c4", "race_obs", "next_c1")
    return _result(
        "knowability_order",
        bad,
        f"{j.height} events: qualifying < race start < race results < next event",
        "{n} event(s) violate causal knowability order",
    )


GATES: tuple[Callable[[Tables], GateResult], ...] = (
    classification_integrity,
    status_taxonomy,
    points_reconciliation,
    standings_reconciliation,
    results_coverage,
    knowability_order,
)


def run_gates(t: Tables) -> list[GateResult]:
    return [gate(t) for gate in GATES]


def enforce(t: Tables) -> list[GateResult]:
    """Run every gate; raise :class:`QualityError` if any failed."""
    results = run_gates(t)
    if not all(r.passed for r in results):
        raise QualityError(results)
    return results
