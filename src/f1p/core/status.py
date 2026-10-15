"""Result-status taxonomy (plan Appendix B; fixes legacy defect D1).

The legacy code treated any status other than ``Finished`` / ``+N Lap(s)`` as a retirement, so
the ``Lapped`` status used by the source for 2024+ results turned 338 classified finishers into
DNFs. Here every raw status maps to an explicit class, and classification follows the sporting
rule (90% of the winner's laps, rounded down) rather than the status string alone.
"""

from __future__ import annotations

import math
import re
from enum import StrEnum


class StatusClass(StrEnum):
    FINISHED = "finished"
    LAPPED = "lapped"
    DNF_MECHANICAL = "dnf_mechanical"
    DNF_INCIDENT = "dnf_incident"
    DNF_OTHER = "dnf_other"
    NOT_CLASSIFIED = "not_classified"
    DSQ = "dsq"
    DNS = "dns"
    DNQ = "dnq"
    DNPQ = "dnpq"

    @property
    def took_finish(self) -> bool:
        """Running at the flag (whether or not on the lead lap)."""
        return self in (StatusClass.FINISHED, StatusClass.LAPPED)

    @property
    def retired(self) -> bool:
        return self in (StatusClass.DNF_MECHANICAL, StatusClass.DNF_INCIDENT, StatusClass.DNF_OTHER)

    @property
    def started(self) -> bool:
        return self not in (StatusClass.DNS, StatusClass.DNQ, StatusClass.DNPQ)


class UnknownStatusError(ValueError):
    """Raised in strict mode for a raw status the taxonomy does not recognise."""


_LAPPED = re.compile(r"^\+\s*\d+\s+laps?$", re.IGNORECASE)

# Order matters: the first matching rule wins, so specific phrases precede generic keywords.
_EXACT: dict[str, StatusClass] = {
    "finished": StatusClass.FINISHED,
    "lapped": StatusClass.LAPPED,
    "disqualified": StatusClass.DSQ,
    "excluded": StatusClass.DSQ,
    "did not start": StatusClass.DNS,
    "withdrew": StatusClass.DNS,
    "did not qualify": StatusClass.DNQ,
    "107% rule": StatusClass.DNQ,
    "did not prequalify": StatusClass.DNPQ,
    "not classified": StatusClass.NOT_CLASSIFIED,
    "retired": StatusClass.DNF_OTHER,
    "retirement": StatusClass.DNF_OTHER,
    "illness": StatusClass.DNF_OTHER,
    "physical": StatusClass.DNF_OTHER,
    "driver unwell": StatusClass.DNF_OTHER,
    "unwell": StatusClass.DNF_OTHER,
    "unfit": StatusClass.DNF_OTHER,
    "injured": StatusClass.DNF_OTHER,
    "injury": StatusClass.DNF_OTHER,
    "injury in practice": StatusClass.DNF_OTHER,
    "stalled": StatusClass.DNF_OTHER,  # driver error or mechanical — ambiguous
    # Rule breaches and administrative stops (never mechanical, even when they name a part).
    "technical infringements": StatusClass.DNF_OTHER,
    "additional brake fluid": StatusClass.DNF_OTHER,
    "fuel sample": StatusClass.DNF_OTHER,
    "wrong tyres": StatusClass.DNF_OTHER,
    "tyre dispute": StatusClass.DNF_OTHER,
    "tyre shortage": StatusClass.DNF_OTHER,
    "camera car": StatusClass.DNF_OTHER,
    "car seized by police": StatusClass.DNF_OTHER,
    "car underweight": StatusClass.DNF_OTHER,
    "driving too slowly": StatusClass.DNF_OTHER,
    "failed scrutineering": StatusClass.DNF_OTHER,
    "failed to serve stop-go penalty": StatusClass.DNF_OTHER,
    "misled stewards": StatusClass.DNF_OTHER,
    "missed weight check": StatusClass.DNF_OTHER,
    "money dispute": StatusClass.DNF_OTHER,
    "overtaking on formation lap": StatusClass.DNF_OTHER,
    "pit stop issue": StatusClass.DNF_OTHER,
    "practice only": StatusClass.DNF_OTHER,
    "push start": StatusClass.DNF_OTHER,
    "push start in pits": StatusClass.DNF_OTHER,
    "received outside assistance": StatusClass.DNF_OTHER,
    "rejoined track illegally": StatusClass.DNF_OTHER,
    "reserve driver": StatusClass.DNF_OTHER,
    "reversed in pits": StatusClass.DNF_OTHER,
    "safety concerns": StatusClass.DNF_OTHER,
}

# Prefixes of rule breaches / car hand-overs; checked before the part keywords below, so
# "Illegal fuel" or "Ignored black flag" never count as mechanical or incident retirements.
_OTHER_PREFIXES = ("car raced by", "illegal", "ignored", "incorrect")

_INCIDENT_WORDS = (
    "accident",
    "collision",
    "crash",
    "spun off",
    "spin",
    "damage",
    "puncture",
    "debris",
)

_MECHANICAL_WORDS = (
    "engine",
    "gearbox",
    "transmission",
    "clutch",
    "hydraulic",
    "electrical",
    "electronic",
    "power unit",
    "power loss",
    "ers",
    "turbo",
    "mgu",
    "battery",
    "brake",
    "suspension",
    "wheel",
    "driveshaft",
    "drivetrain",
    "differential",
    "axle",
    "overheating",
    "oil",
    "water",
    "fuel",
    "exhaust",
    "throttle",
    "steering",
    "radiator",
    "cooling",
    "pneumatic",
    "mechanical",
    "technical",
    "vibration",
    "tyre",
    "halfshaft",
    "chassis",
    "spark plugs",
    "alternator",
    "distributor",
    "ignition",
    "magneto",
    "injection",
    "pump",
    "valve",
    "piston",
    "crankshaft",
    "camshaft",
    "con-rod",
    "out of fuel",
    "seat",
    "undertray",
    "front wing",
    "rear wing",
    "wing",
    "body work",
    "bodywork",
    "refuelling",
    "fire",
    "airbox",
    "floor",
    "compressor",
    "coolant",
    "energy storage",
    "hybrid",
    "kers",
    "launch control",
    "mechenical",  # sic — a source typo
    "starter",
    "supercharger",
    "track rod",
    "traction control",
    "safety switch",
    "safety belt",
    "handling",
    "unraceworthy",
    "car not ready",
)


def _matches(text: str, words: tuple[str, ...]) -> bool:
    return any(re.search(rf"\b{re.escape(w)}", text) for w in words)


def classify_status(raw: str, *, strict: bool = False) -> StatusClass:
    """Map a raw source status (e.g. ``"+1 Lap"``, ``"Lapped"``, ``"Gearbox"``) to a class.

    Unrecognised statuses become ``DNF_OTHER``, or raise :class:`UnknownStatusError` when
    ``strict`` is set (the Phase 1 data-quality gate runs in strict mode).
    """
    text = " ".join(raw.strip().lower().split())
    if not text:
        raise UnknownStatusError("empty status")
    if text in _EXACT:
        return _EXACT[text]
    if _LAPPED.match(text):
        return StatusClass.LAPPED
    if text.startswith(_OTHER_PREFIXES):
        return StatusClass.DNF_OTHER
    if _matches(text, _INCIDENT_WORDS):
        return StatusClass.DNF_INCIDENT
    if _matches(text, _MECHANICAL_WORDS):
        return StatusClass.DNF_MECHANICAL
    if strict:
        raise UnknownStatusError(f"unrecognised status: {raw!r}")
    return StatusClass.DNF_OTHER


def classification_threshold(winner_laps: int) -> int:
    """Minimum laps for classification: 90% of the winner's laps, rounded down."""
    if winner_laps < 0:
        raise ValueError("winner_laps cannot be negative")
    return math.floor(0.9 * winner_laps)


def is_classified(status: StatusClass, laps_completed: int, winner_laps: int) -> bool:
    """Whether a driver appears in the official classification.

    Lead-lap finishers are always classified. Lapped finishers and retirements are classified
    if they covered at least 90% of the winner's laps (rounded down). Entries the source marks
    not classified, disqualified, non-starting or non-qualifying never are.
    """
    if status is StatusClass.FINISHED:
        return True
    if status is StatusClass.LAPPED or status.retired:
        return laps_completed >= classification_threshold(winner_laps)
    return False
