"""Regulation eras, resets and session formats.

Resets are listed as *data*: the world model learns how strongly each one scrambled the order
(plan §7.1) instead of hard-coding a regression constant as the legacy code did.
"""

from __future__ import annotations

from dataclasses import dataclass

# Seasons that began a major chassis and/or power-unit rule-set.
MAJOR_RESETS: tuple[int, ...] = (
    1961,
    1966,
    1983,
    1989,
    1995,
    1998,
    2009,
    2014,
    2017,
    2022,
    2026,
)

# Resets whose main change was the power unit, classified from the regulations themselves
# (fixed before any model was fitted to them; see docs/plan/RESET_PLAN.md). Engine changes
# re-order the field more than chassis/aero ones, so the world model may give them a larger
# reset. The others are chassis/aero resets:
#   1961  2.5 l → 1.5 l engines                 1983  flat bottoms, ground effect banned
#   1966  1.5 l → 3.0 l engines                 1998  narrow track, grooved tyres
#   1989  turbos banned, 3.5 l atmospheric      2009  aero overhaul, slicks (KERS optional)
#   1995  3.5 l → 3.0 l engines                 2017  wider cars and tyres
#   2014  1.6 l V6 turbo-hybrid power units     2022  ground-effect floors, 18" wheels
#   2026  new power units (half electric) with new chassis and active aero
POWER_UNIT_RESETS: tuple[int, ...] = (1961, 1966, 1989, 1995, 2014, 2026)

FIRST_CHAMPIONSHIP_SEASON = 1950
KNOCKOUT_QUALIFYING_FIRST_SEASON = 2006
Q3_SIZE = 10


@dataclass(frozen=True, slots=True)
class Era:
    """A stretch of seasons between two resets. ``last`` is ``None`` for the current era."""

    first: int
    last: int | None

    @property
    def name(self) -> str:
        return f"era_{self.first}"

    def __contains__(self, season: object) -> bool:
        return (
            isinstance(season, int)
            and season >= self.first
            and (self.last is None or season <= self.last)
        )


def _check(season: int) -> None:
    if season < FIRST_CHAMPIONSHIP_SEASON:
        raise ValueError(f"no world championship before {FIRST_CHAMPIONSHIP_SEASON}: {season}")


def era_start(season: int) -> int:
    """First season of the era containing ``season``."""
    _check(season)
    starts = [r for r in MAJOR_RESETS if r <= season]
    return starts[-1] if starts else FIRST_CHAMPIONSHIP_SEASON


def era(season: int) -> Era:
    start = era_start(season)
    later = [r for r in MAJOR_RESETS if r > start]
    return Era(first=start, last=later[0] - 1 if later else None)


def is_power_unit_reset(season: int) -> bool:
    _check(season)
    return season in POWER_UNIT_RESETS


def is_reset_season(season: int) -> bool:
    _check(season)
    return season in MAJOR_RESETS


def seasons_since_reset(season: int) -> int:
    """0 in a reset season, 1 the season after, and so on."""
    return season - era_start(season)


@dataclass(frozen=True, slots=True)
class KnockoutFormat:
    """Cars eliminated after Q1 and Q2; the remaining ``Q3_SIZE`` fight for pole."""

    field_size: int
    q1_eliminated: int
    q2_eliminated: int

    @property
    def q2_size(self) -> int:
        return self.field_size - self.q1_eliminated

    @property
    def q3_size(self) -> int:
        return self.q2_size - self.q2_eliminated


def knockout_format(season: int, field_size: int) -> KnockoutFormat:
    """Three-part knockout qualifying for a field of ``field_size`` cars.

    Matches every knockout season: 20 cars → 5 + 5, 22 → 6 + 6 (incl. 2026), 24 → 7 + 7. An odd
    field eliminates the extra car in Q1. Formats before 2006 differed and are not modelled.
    """
    if season < KNOCKOUT_QUALIFYING_FIRST_SEASON:
        raise NotImplementedError(
            f"knockout qualifying began in {KNOCKOUT_QUALIFYING_FIRST_SEASON}"
        )
    if field_size <= Q3_SIZE:
        raise ValueError(f"field of {field_size} is too small for knockout qualifying")
    surplus = field_size - Q3_SIZE
    q1 = (surplus + 1) // 2
    return KnockoutFormat(field_size=field_size, q1_eliminated=q1, q2_eliminated=surplus - q1)
