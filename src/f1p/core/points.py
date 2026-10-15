"""Championship points, exact per season (plan Appendix D; fixes legacy defect D5).

Supported from 1991, the first season of the 10-6-4-3-2-1 system that every later system
descends from. Earlier seasons (dropped scores, shared drives) are reconciled from F1DB data in
Phase 1 rather than re-derived here.

Two kinds of special case exist:

* **Conditions** that follow from how a race ran — distance completed and green-flag laps.
  Before 2022 a race stopped short of 75% distance paid half points; from 2022 the
  shortened-race scales apply.
* **Event multipliers** that no condition explains — only 2014 Abu Dhabi (double points).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

EARLIEST_SUPPORTED_SEASON = 1991


class Session(StrEnum):
    RACE = "race"
    SPRINT = "sprint"


# (first season, last season or None, points for P1, P2, ...)
_RACE_SCALES: tuple[tuple[int, int | None, tuple[float, ...]], ...] = (
    (1991, 2002, (10, 6, 4, 3, 2, 1)),
    (2003, 2009, (10, 8, 6, 5, 4, 3, 2, 1)),
    (2010, None, (25, 18, 15, 12, 10, 8, 6, 4, 2, 1)),
)
_SPRINT_SCALES: tuple[tuple[int, int | None, tuple[float, ...]], ...] = (
    (2021, 2021, (3, 2, 1)),
    (2022, None, (8, 7, 6, 5, 4, 3, 2, 1)),
)

# Bonus point for the fastest lap, awarded only to a driver classified in the top ten.
FASTEST_LAP_FIRST_SEASON = 2019
FASTEST_LAP_LAST_SEASON = 2024  # scrapped for 2025 by the World Motor Sport Council

# Shortened-race scales from 2022, keyed by the upper bound of the completed-distance band.
SHORTENED_RACE_FIRST_SEASON = 2022
_SHORTENED_SCALES: tuple[tuple[float, tuple[float, ...]], ...] = (
    (0.25, (6, 4, 3, 2, 1)),
    (0.50, (13, 10, 8, 6, 5, 4, 3, 2, 1)),
    (0.75, (19, 14, 12, 9, 8, 6, 5, 3, 2, 1)),
)
FULL_POINTS_DISTANCE = 0.75
MIN_GREEN_FLAG_LAPS = 2  # 2022+: no points unless the leader completes two laps without SC/VSC

# Explicit multipliers keyed by (season, circuit_id). Half-points races are NOT listed here:
# they follow from ``RaceConditions.distance_fraction``.
EVENT_MULTIPLIERS: MappingProxyType[tuple[int, str], float] = MappingProxyType(
    {(2014, "yas_marina"): 2.0},
)

# Races that ran short of 75% yet paid full points. In 2022 the reduced scales applied only to
# a race suspended and *not resumed*; the 2022 Japanese GP resumed and ran to the time limit
# (28/53 laps), so full points were awarded — the rule was rewritten for 2023.
FULL_POINTS_EVENTS: frozenset[tuple[int, str]] = frozenset({(2022, "suzuka")})


@dataclass(frozen=True, slots=True)
class RaceConditions:
    """How a race actually ran, as far as scoring is concerned.

    ``distance_fraction`` is the share of the scheduled distance the leader completed.
    ``green_flag_laps`` is the number of laps the leader completed without SC/VSC (2022+
    minimum-laps rule); ``None`` means "not known, assume the minimum was met".
    """

    distance_fraction: float = 1.0
    green_flag_laps: int | None = None
    multiplier: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.distance_fraction <= 1.0:
            raise ValueError(f"distance_fraction must be in [0, 1], got {self.distance_fraction}")
        if self.green_flag_laps is not None and self.green_flag_laps < 0:
            raise ValueError("green_flag_laps cannot be negative")
        if self.multiplier < 0:
            raise ValueError("multiplier cannot be negative")

    @property
    def is_full_distance(self) -> bool:
        return self.distance_fraction >= FULL_POINTS_DISTANCE


FULL_RACE = RaceConditions()


def _lookup(
    scales: tuple[tuple[int, int | None, tuple[float, ...]], ...], season: int, session: Session
) -> tuple[float, ...]:
    for first, last, table in scales:
        if season >= first and (last is None or season <= last):
            return table
    if session is Session.SPRINT:
        raise ValueError(f"no sprint races were held in {season}")
    raise NotImplementedError(
        f"points for {season} are not derived here (supported from {EARLIEST_SUPPORTED_SEASON})"
    )


def base_scale(season: int, session: Session = Session.RACE) -> tuple[float, ...]:
    """Points for P1, P2, ... under the season's full-distance system."""
    scales = _SPRINT_SCALES if session is Session.SPRINT else _RACE_SCALES
    return _lookup(scales, season, session)


def scale(
    season: int,
    session: Session = Session.RACE,
    conditions: RaceConditions = FULL_RACE,
) -> tuple[float, ...]:
    """Points for P1, P2, ... after applying shortened-race rules and multipliers."""
    if session is Session.SPRINT:
        if not conditions.is_full_distance or conditions.multiplier != 1.0:
            # Shortened-sprint scoring is reconciled from official data in Phase 1.
            raise NotImplementedError("shortened or weighted sprints are not modelled yet")
        return base_scale(season, session)

    table = base_scale(season, session)
    if season >= SHORTENED_RACE_FIRST_SEASON:
        if conditions.green_flag_laps is not None and (
            conditions.green_flag_laps < MIN_GREEN_FLAG_LAPS
        ):
            table = ()
        elif not conditions.is_full_distance:
            table = next(
                band for limit, band in _SHORTENED_SCALES if conditions.distance_fraction < limit
            )
    elif not conditions.is_full_distance:
        table = tuple(p / 2 for p in table)
    return tuple(p * conditions.multiplier for p in table)


def event_multiplier(season: int, circuit_id: str) -> float:
    """Explicit multiplier for an event (1.0 unless it is a documented special case)."""
    return EVENT_MULTIPLIERS.get((season, circuit_id), 1.0)


def race_conditions(
    season: int,
    circuit_id: str,
    distance_fraction: float = 1.0,
    green_flag_laps: int | None = None,
) -> RaceConditions:
    """Scoring conditions for a real event, applying every documented special case.

    This is the one entry point data pipelines should use: it combines the completed
    distance with the event multipliers and full-points exceptions above. It reproduces the
    official points of every race and sprint since 1991 (checked by the points-reconciliation
    gate in ``f1p.lake.quality``).
    """
    if (season, circuit_id) in FULL_POINTS_EVENTS:
        distance_fraction = 1.0
    return RaceConditions(
        distance_fraction=distance_fraction,
        green_flag_laps=green_flag_laps,
        multiplier=event_multiplier(season, circuit_id),
    )


def fastest_lap_bonus_applies(
    season: int,
    session: Session = Session.RACE,
    conditions: RaceConditions = FULL_RACE,
) -> bool:
    """Whether the fastest-lap bonus point exists for this race.

    Only full-points races are treated as eligible. Shortened races carry no bonus here — true
    for 2021 Spa; the 2022+ shortened-race wording is confirmed by points reconciliation
    against official standings in Phase 1.
    """
    return (
        session is Session.RACE
        and FASTEST_LAP_FIRST_SEASON <= season <= FASTEST_LAP_LAST_SEASON
        and conditions.is_full_distance
        and conditions.multiplier == 1.0
    )


def points_for(
    position: int | None,
    season: int,
    session: Session = Session.RACE,
    conditions: RaceConditions = FULL_RACE,
    *,
    classified: bool = True,
    fastest_lap: bool = False,
) -> float:
    """Points scored by one driver.

    ``position`` is the classified finishing position (1-based); ``None`` or an unclassified
    driver scores nothing. ``fastest_lap`` marks the driver who set the race's fastest lap.
    """
    if position is None or not classified:
        return 0.0
    if position < 1:
        raise ValueError(f"position must be >= 1, got {position}")
    table = scale(season, session, conditions)
    pts = float(table[position - 1]) if position <= len(table) else 0.0
    if fastest_lap and position <= 10 and fastest_lap_bonus_applies(season, session, conditions):
        pts += 1.0
    return pts


def points_table(
    season: int,
    n_positions: int,
    session: Session = Session.RACE,
    conditions: RaceConditions = FULL_RACE,
) -> tuple[float, ...]:
    """Dense points vector for positions 1..``n_positions`` (zeros past the scoring places).

    Useful for vectorised expected-points calculations from a finishing-position distribution.
    """
    table = scale(season, session, conditions)
    return tuple(float(table[i]) if i < len(table) else 0.0 for i in range(n_positions))
