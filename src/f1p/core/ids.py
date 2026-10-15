"""Canonical identifiers shared by every layer.

Entity ids are lowercase ASCII slugs in the style the Ergast/Jolpica lineage uses
(``max_verstappen``, ``red_bull``, ``yas_marina``), so source ids usually pass through
unchanged and cross-source resolution (Phase 1) only has to handle the exceptions.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from typing import NewType

DriverId = NewType("DriverId", str)
TeamId = NewType("TeamId", str)
LineageId = NewType("LineageId", str)
CircuitId = NewType("CircuitId", str)
LayoutId = NewType("LayoutId", str)

_SLUG_SEP = re.compile(r"[^a-z0-9]+")


def slugify(name: str) -> str:
    """``"Kimi Räikkönen"`` → ``"kimi_raikkonen"``; ``"Red Bull"`` → ``"red_bull"``."""
    ascii_text = (
        unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii").lower()
    )
    slug = _SLUG_SEP.sub("_", ascii_text).strip("_")
    if not slug:
        raise ValueError(f"cannot build an id from {name!r}")
    return slug


class SessionType(StrEnum):
    """Every session a weekend (or test) can contain."""

    TEST = "test"
    FP1 = "fp1"
    FP2 = "fp2"
    FP3 = "fp3"
    SPRINT_QUALIFYING = "sq"
    SPRINT = "sprint"
    QUALIFYING = "quali"
    RACE = "race"

    @property
    def is_practice(self) -> bool:
        return self in (SessionType.TEST, SessionType.FP1, SessionType.FP2, SessionType.FP3)

    @property
    def scores_points(self) -> bool:
        return self in (SessionType.SPRINT, SessionType.RACE)


_EVENT_KEY = re.compile(r"^(?P<season>\d{4})-R(?P<round>\d{2})$")


@dataclass(frozen=True, slots=True, order=True)
class EventKey:
    """A championship round, rendered ``2026-R05``."""

    season: int
    round: int

    def __post_init__(self) -> None:
        if not 1950 <= self.season <= 2100:
            raise ValueError(f"implausible season {self.season}")
        if not 1 <= self.round <= 99:
            raise ValueError(f"round must be 1..99, got {self.round}")

    def __str__(self) -> str:
        return f"{self.season}-R{self.round:02d}"

    @classmethod
    def parse(cls, text: str) -> EventKey:
        m = _EVENT_KEY.match(text)
        if not m:
            raise ValueError(f"not an event key: {text!r} (expected e.g. '2026-R05')")
        return cls(int(m["season"]), int(m["round"]))


@dataclass(frozen=True, slots=True, order=True)
class SessionKey:
    """One session of one event, rendered ``2026-R05/quali``."""

    event: EventKey
    session: SessionType

    def __str__(self) -> str:
        return f"{self.event}/{self.session.value}"

    @classmethod
    def parse(cls, text: str) -> SessionKey:
        event, sep, session = text.partition("/")
        if not sep:
            raise ValueError(f"not a session key: {text!r} (expected e.g. '2026-R05/quali')")
        return cls(EventKey.parse(event), SessionType(session))
