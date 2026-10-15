from __future__ import annotations

import numpy as np
import pytest

from f1p.lake.curate_f1db import Curated
from f1p.models.ratings import RaceObs, race_observations
from f1p.models.reliability import ReliabilityFilter, run_reliability


def _race(statuses: list[str], finish: list[int], teams: list[str] | None = None) -> RaceObs:
    n = len(statuses)
    drivers = tuple(f"d{i}" for i in range(n))
    return RaceObs(
        "x",
        2020,
        1,
        drivers,
        tuple(teams or [f"T{i}" for i in range(n)]),
        np.array(finish, dtype=np.intp),
        status=tuple(statuses),
    )


def test_mechanical_failures_hurt_the_team_incidents_the_driver() -> None:
    f = ReliabilityFilter()
    for _ in range(10):
        f.update(
            _race(
                ["finished", "dnf_mechanical", "dnf_incident", "finished"],
                [0, 3],
                ["A", "B", "C", "D"],
            )
        )
    p = f.p_finish(["d0", "d1", "d2", "d3"], ["A", "B", "C", "D"], f.step)
    assert p[1] < p[0]
    assert p[2] < p[0]
    # a new driver in the fragile car inherits the car's risk, not the crasher's
    p_new = f.p_finish(["new", "new2"], ["B", "A"], f.step)
    assert p_new[0] < p_new[1]
    # and the crash-prone driver carries the risk into a reliable car
    p_move = f.p_finish(["d2", "new3"], ["A", "A"], f.step)
    assert p_move[0] < p_move[1]


def test_state_before_step_is_causal() -> None:
    f = ReliabilityFilter()
    f.update(_race(["finished", "finished"], [0, 1]))
    before = f.p_finish(["d0"], ["T0"], 1)
    f.update(_race(["dnf_mechanical", "finished"], [1]))
    assert f.p_finish(["d0"], ["T0"], 1) == pytest.approx(before)
    assert f.p_finish(["d0"], ["T0"], 2)[0] < before[0]


def test_classified_retirement_is_not_a_dnf() -> None:
    f = ReliabilityFilter()
    for _ in range(5):  # retired on the last lap but classified (>90% distance)
        f.update(_race(["dnf_mechanical", "finished"], [0, 1]))
    assert f.field_rates(f.step)[0] == pytest.approx(0.0)


def test_rates_track_history(curated: Curated) -> None:
    races = race_observations(curated.tables)
    f = run_reliability(races)
    last = races[-1]
    p = f.p_finish(last.drivers, last.teams, len(races))
    assert np.all((p > 0.5) & (p < 1.0))
    rates = f.field_rates(len(races))
    assert 0.0 < rates.sum() < 0.5
