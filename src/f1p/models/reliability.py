"""Reliability: the probability that each car is classified (plan §7.1, retirements).

Retirements are split by cause and attributed to whoever is most plausibly responsible:

* **mechanical** failures to the team lineage (the car);
* **incidents** (accidents, collisions, spins) to the driver;
* everything else (disqualifications, "not classified", unknown) to a field-wide base rate.

Every rate is a Beta-Binomial belief with exponential forgetting per race, shrunk towards the
current field-wide rate for that cause, so a team's 1950s fragility does not haunt it in 2026
and a newcomer starts at the era norm. Like the rating filter, the state is recorded per
entity, so the belief at any past moment is exact and causal.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from f1p.models.ratings import BeliefHistory, RaceObs

FloatArray = NDArray[np.float64]

MECHANICAL = "dnf_mechanical"
INCIDENT = "dnf_incident"


@dataclass(frozen=True, slots=True)
class ReliabilityParams:
    decay_entity: float = 0.97  # per race the entity starts: weight kept by older evidence
    decay_field: float = 0.95  # per race, for the field-wide base rates
    prior_starts: float = 15.0  # pseudo-starts of the field rate an entity is shrunk towards

    BOUNDS: ClassVar[dict[str, tuple[float, float]]] = {
        "decay_entity": (0.8, 0.999),
        "decay_field": (0.7, 0.999),
        "prior_starts": (1.0, 200.0),
    }


@dataclass(slots=True)
class _Rate:
    events: float = 0.0
    exposure: float = 0.0

    def decayed(self, factor: float) -> _Rate:
        return _Rate(self.events * factor, self.exposure * factor)


class ReliabilityFilter:
    """Exponentially-forgetting cause-specific retirement rates."""

    def __init__(self, params: ReliabilityParams | None = None):
        self.p = params or ReliabilityParams()
        self.step = 0
        # field-wide (mech, incident, other) per car-start, recorded after each step
        self.field = np.zeros(3)
        self.field_exposure = 0.0
        self.field_hist: list[tuple[float, float, float, float]] = []
        self.team: dict[str, _Rate] = {}
        self.driver: dict[str, _Rate] = {}
        self.team_hist: dict[str, BeliefHistory] = {}
        self.driver_hist: dict[str, BeliefHistory] = {}

    def update(self, race: RaceObs) -> None:
        finished = np.zeros(len(race.drivers), dtype=bool)
        finished[race.finish_order] = True
        counts = np.zeros(3)
        mech_by_team: dict[str, list[float]] = {}
        for i, (d, t) in enumerate(zip(race.drivers, race.teams, strict=True)):
            status = race.status[i] if race.status else ""
            mech = float(not finished[i] and status == MECHANICAL)
            inc = float(not finished[i] and status == INCIDENT)
            other = float(not finished[i] and not mech and not inc)
            counts += (mech, inc, other)
            tm = mech_by_team.setdefault(t, [0.0, 0.0])
            tm[0] += mech
            tm[1] += 1.0
            self._bump(self.driver, self.driver_hist, d, inc, 1.0)
        for t, (events, exposure) in mech_by_team.items():
            self._bump(self.team, self.team_hist, t, events, exposure)
        f = self.p.decay_field
        self.field = self.field * f + counts
        self.field_exposure = self.field_exposure * f + len(race.drivers)
        self.field_hist.append((*self.field.tolist(), self.field_exposure))  # type: ignore[arg-type]
        self.step += 1

    def _bump(
        self, store: dict[str, _Rate], hist: dict[str, BeliefHistory], key: str, ev: float, n: float
    ) -> None:
        r = store.get(key) or _Rate()
        r = r.decayed(self.p.decay_entity)
        r.events += ev
        r.exposure += n
        store[key] = r
        h = hist.get(key)
        if h is None:
            h = hist[key] = BeliefHistory()
        h.steps.append(self.step)
        h.states.append((r.events, r.exposure, 0))

    def field_rates(self, before_step: int) -> FloatArray:
        """Field-wide (mechanical, incident, other) retirement rate per start."""
        if before_step <= 0 or not self.field_hist:
            return np.array([0.08, 0.06, 0.03])  # modern-era-ish defaults before any data
        m, i, o, n = self.field_hist[min(before_step, len(self.field_hist)) - 1]
        return np.array([m, i, o]) / max(n, 1e-9)

    def _entity_rate(
        self, hist: dict[str, BeliefHistory], key: str, step: int, base: float
    ) -> float:
        h = hist.get(key)
        state = h.at(step) if h is not None else None
        events, exposure = (state[0], state[1]) if state is not None else (0.0, 0.0)
        k = self.p.prior_starts
        return (events + k * base) / (exposure + k)

    def p_finish(
        self, drivers: Sequence[str], teams: Sequence[str], before_step: int
    ) -> FloatArray:
        """P(classified) for each car, from races before ``before_step``."""
        mech0, inc0, other0 = self.field_rates(before_step)
        out = np.empty(len(drivers))
        for j, (d, t) in enumerate(zip(drivers, teams, strict=True)):
            pm = self._entity_rate(self.team_hist, t, before_step, mech0)
            pi = self._entity_rate(self.driver_hist, d, before_step, inc0)
            out[j] = (1.0 - pm) * (1.0 - pi) * (1.0 - other0)
        return np.clip(out, 0.01, 0.999)


def run_reliability(
    races: Sequence[RaceObs], params: ReliabilityParams | None = None
) -> ReliabilityFilter:
    f = ReliabilityFilter(params)
    for race in races:
        f.update(race)
    return f
