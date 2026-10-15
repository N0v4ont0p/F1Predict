from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from f1p.core.status import (
    StatusClass,
    UnknownStatusError,
    classification_threshold,
    classify_status,
    is_classified,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Finished", StatusClass.FINISHED),
        # The legacy bug (D1): "Lapped" is a classified finisher, not a retirement.
        ("Lapped", StatusClass.LAPPED),
        ("+1 Lap", StatusClass.LAPPED),
        ("+12 Laps", StatusClass.LAPPED),
        ("  +2   laps ", StatusClass.LAPPED),
        ("Gearbox", StatusClass.DNF_MECHANICAL),
        ("Power Unit", StatusClass.DNF_MECHANICAL),
        ("Hydraulics", StatusClass.DNF_MECHANICAL),
        ("Fuel pressure", StatusClass.DNF_MECHANICAL),
        ("Water leak", StatusClass.DNF_MECHANICAL),
        ("ERS", StatusClass.DNF_MECHANICAL),
        ("Tyre", StatusClass.DNF_MECHANICAL),
        ("Accident", StatusClass.DNF_INCIDENT),
        ("Collision", StatusClass.DNF_INCIDENT),
        ("Collision damage", StatusClass.DNF_INCIDENT),
        ("Spun off", StatusClass.DNF_INCIDENT),
        ("Puncture", StatusClass.DNF_INCIDENT),
        ("Damage", StatusClass.DNF_INCIDENT),
        ("Retired", StatusClass.DNF_OTHER),
        ("Illness", StatusClass.DNF_OTHER),
        ("Not classified", StatusClass.NOT_CLASSIFIED),
        ("Disqualified", StatusClass.DSQ),
        ("Excluded", StatusClass.DSQ),
        ("Did not start", StatusClass.DNS),
        ("Withdrew", StatusClass.DNS),
        ("Did not qualify", StatusClass.DNQ),
        ("107% Rule", StatusClass.DNQ),
        ("Did not prequalify", StatusClass.DNPQ),
    ],
)
def test_classify_known_statuses(raw: str, expected: StatusClass) -> None:
    assert classify_status(raw) is expected


def test_unknown_status_is_dnf_other_unless_strict() -> None:
    assert classify_status("Mystery") is StatusClass.DNF_OTHER
    with pytest.raises(UnknownStatusError):
        classify_status("Mystery", strict=True)


def test_empty_status_rejected() -> None:
    with pytest.raises(UnknownStatusError):
        classify_status("   ")


def test_status_properties() -> None:
    assert StatusClass.LAPPED.took_finish
    assert not StatusClass.LAPPED.retired
    assert StatusClass.DNF_MECHANICAL.retired
    assert not StatusClass.DNS.started
    assert StatusClass.DSQ.started


@pytest.mark.parametrize(("winner", "threshold"), [(57, 51), (44, 39), (70, 63), (1, 0), (0, 0)])
def test_classification_threshold_rounds_down(winner: int, threshold: int) -> None:
    assert classification_threshold(winner) == threshold


@pytest.mark.parametrize(
    ("status", "laps", "winner", "expected"),
    [
        (StatusClass.FINISHED, 57, 57, True),
        (StatusClass.LAPPED, 55, 57, True),
        (StatusClass.LAPPED, 45, 57, False),
        (StatusClass.DNF_MECHANICAL, 52, 57, True),  # retired late, still classified
        (StatusClass.DNF_INCIDENT, 50, 57, False),
        (StatusClass.NOT_CLASSIFIED, 57, 57, False),
        (StatusClass.DSQ, 57, 57, False),
        (StatusClass.DNS, 0, 57, False),
    ],
)
def test_is_classified(status: StatusClass, laps: int, winner: int, expected: bool) -> None:
    assert is_classified(status, laps, winner) is expected


@given(st.integers(min_value=0, max_value=200))
def test_threshold_is_at_most_ninety_percent(winner: int) -> None:
    t = classification_threshold(winner)
    assert t <= 0.9 * winner < t + 1


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Rule breaches name car parts but are not mechanical failures.
        ("Illegal fuel", StatusClass.DNF_OTHER),
        ("Illegal brake ducts", StatusClass.DNF_OTHER),
        ("Ignored black flag", StatusClass.DNF_OTHER),
        ("Car raced by Surtees", StatusClass.DNF_OTHER),
        ("Technical infringements", StatusClass.DNF_OTHER),
        ("Push start in pits", StatusClass.DNF_OTHER),
        ("Stalled", StatusClass.DNF_OTHER),
        ("Unwell", StatusClass.DNF_OTHER),
        ("Energy storage system failure", StatusClass.DNF_MECHANICAL),
        ("KERS failure", StatusClass.DNF_MECHANICAL),
        ("Mechenical", StatusClass.DNF_MECHANICAL),
        ("Broken floor", StatusClass.DNF_MECHANICAL),
        ("Fatal accident in practice", StatusClass.DNF_INCIDENT),
    ],
)
def test_classify_f1db_reasons(raw: str, expected: StatusClass) -> None:
    assert classify_status(raw, strict=True) is expected
