"""Search spaces (plan M6 §C.3): what a trial varies, the model it builds, the sampler that
suits it, and how *simple* a configuration is (for the one-standard-error rule: among
configurations within 1 SE of the best on *select*, the simplest wins).

Simplicity is "closest to the hand-set configuration" measured on each parameter's own
scale (standardised by its search range), so an equally good configuration that moves
fewer knobs, or moves them less, is preferred.
"""

from __future__ import annotations

import importlib.util
import math
from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from typing import Any

from f1p.models.ratings import RatingParams
from f1p.models.world_v1 import WorldV1


@dataclass(frozen=True, slots=True)
class Param:
    name: str
    low: float
    high: float
    log: bool = False
    integer: bool = False

    def suggest(self, trial: Any) -> float | int:
        if self.integer:
            return int(trial.suggest_int(self.name, int(self.low), int(self.high), log=self.log))
        return float(trial.suggest_float(self.name, self.low, self.high, log=self.log))

    def unit(self, value: float) -> float:
        """Position in the search range, 0..1 (log scale for log parameters)."""
        if self.log:
            return (math.log(value) - math.log(self.low)) / (
                math.log(self.high) - math.log(self.low)
            )
        return (value - self.low) / (self.high - self.low)


@dataclass(frozen=True)
class Space:
    name: str
    target: str  # race or quali
    params: tuple[Param, ...]
    build: Callable[[dict[str, Any]], Any]  # params → model
    baseline: dict[str, Any]  # the hand-set configuration (trial 0)
    sampler: str  # tpe, cmaes, gp
    expensive: bool = False

    def suggest(self, trial: Any) -> dict[str, Any]:
        return {p.name: p.suggest(trial) for p in self.params}

    def complexity(self, values: dict[str, Any]) -> float:
        """Distance from the hand-set configuration in standardised units (lower = simpler)."""
        total = 0.0
        for p in self.params:
            if p.name in values and p.name in self.baseline:
                total += abs(p.unit(float(values[p.name])) - p.unit(float(self.baseline[p.name])))
        return total


# -- world_v1: combiner (cheap: the prepared signals are shared across trials) ------------------
def _combiner(values: dict[str, Any]) -> WorldV1:
    return WorldV1(l2=float(values["l2"]), training_seasons=int(values["training_seasons"]))


COMBINER = Space(
    "combiner",
    "race",
    (Param("l2", 0.05, 20.0, log=True), Param("training_seasons", 3, 10, integer=True)),
    _combiner,
    {"l2": WorldV1.l2, "training_seasons": WorldV1.training_seasons},
    "tpe",
)


# -- world_v1: the rating filter (every trial re-runs the filter: expensive) ----------------------
_FILTER_FIELDS = tuple(f.name for f in fields(RatingParams) if f.name in RatingParams.BOUNDS)


def _filter(values: dict[str, Any]) -> WorldV1:
    base = WorldV1().params
    return WorldV1(params=replace(base, **{k: float(v) for k, v in values.items()}))


def _filter_param(name: str) -> Param:
    lo, hi = RatingParams.BOUNDS[name]
    # scale parameters are searched on a log scale, shares and multipliers linearly
    log = lo > 0 and name.startswith(("tau", "beta", "driver_sigma", "team_sigma"))
    return Param(name, lo, hi, log=log)


FILTER = Space(
    "filter",
    "race",
    tuple(_filter_param(n) for n in _FILTER_FIELDS),
    _filter,
    {n: getattr(WorldV1().params, n) for n in _FILTER_FIELDS},
    "cmaes",  # > 10 continuous parameters (plan §C.1)
    expensive=True,
)


# -- race_sim_v2: the global knobs not covered by the moment calibration (plan §C.3) --------------
def _race_sim(values: dict[str, Any]) -> Any:
    from f1p.models.race_sim_v2 import RaceSimV2, load_knobs

    knobs = replace(load_knobs(), **{k: float(v) for k, v in values.items()})
    return RaceSimV2(knobs=knobs, n_sims=4_000)


RACE_SIM = Space(
    "race_sim",
    "race",
    (
        Param("pace_scale", 0.4, 2.0, log=True),
        Param("overtake_shift", -2.0, 2.0),
        Param("start_noise", 0.2, 1.5, log=True),
    ),
    _race_sim,
    {"pace_scale": 1.0, "overtake_shift": 0.0, "start_noise": 0.6},
    # Gaussian-process Bayesian optimisation needs PyTorch; multivariate TPE otherwise
    "gp" if importlib.util.find_spec("torch") is not None else "tpe",
    expensive=True,
)


# -- quali_v1: its combiner ---------------------------------------------------------------------
def _quali_defaults() -> Any:
    from f1p.models.quali_v1 import QualiV1

    return QualiV1()


_QUALI_DEFAULTS = _quali_defaults()


def _quali(values: dict[str, Any]) -> Any:
    from f1p.models.quali_v1 import QualiV1

    return QualiV1(l2=float(values["l2"]), training_seasons=int(values["training_seasons"]))


QUALI = Space(
    "quali",
    "quali",
    (Param("l2", 0.05, 20.0, log=True), Param("training_seasons", 3, 10, integer=True)),
    _quali,
    {"l2": _QUALI_DEFAULTS.l2, "training_seasons": _QUALI_DEFAULTS.training_seasons},
    "tpe",
)

SPACES: dict[str, Space] = {s.name: s for s in (COMBINER, FILTER, RACE_SIM, QUALI)}
