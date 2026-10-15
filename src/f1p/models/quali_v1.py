"""Qualifying model v1: the qualifying order at C1 (Thursday) and C2 (before qualifying).

The same machinery as :mod:`f1p.models.world_v1` aimed at a different target: a linear
Plackett-Luce combiner with stage-wise temperatures, refitted each season on earlier
seasons' *qualifying* orders. Its signals:

* ``rating_q`` — the rating filter with qualifying (and sprint) sessions assimilated,
  read at the checkpoint;
* ``rating`` — the race-only filter (race craft and one-lap pace differ);
* ``quali_pace`` — exponentially weighted best-lap deficit to pole in recent sessions;
* ``fp_pace`` (C2) — this weekend's best practice lap as a gap to the fastest;
* ``sq`` (C2) — this weekend's sprint-qualifying order, on sprint weekends where it comes
  first.

There are no retirements in qualifying and no weather input yet. The knockout (Q1/Q2/Q3)
probabilities follow from the full position distribution and the season's format.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from f1p.core.regulations import knockout_format
from f1p.eval.context import EventContext
from f1p.eval.forecast import Forecast
from f1p.models.features import (
    practice_pace,
    quali_pace_signal,
    sprint_quali_ranks,
)
from f1p.models.world_v1 import PRIORS, Signal, WorldV1, _log_rank, predictive_log_strength

QUALI_FEATURES = {
    "C1": ("rating_q", "rating", "quali_pace"),
    "C2": ("rating_q", "rating", "quali_pace", "fp_pace", "sq"),
}
QUALI_PRIORS = PRIORS | {"rating_q": 1.0, "rating": 0.0}


@dataclass
class QualiV1(WorldV1):
    name: str = "quali_v1"
    checkpoints: frozenset[str] = frozenset({"C1", "C2"})
    targets: frozenset[str] = frozenset({"quali"})
    retirements: bool = False
    weather: bool = False
    features: dict[str, tuple[str, ...]] = field(default_factory=lambda: dict(QUALI_FEATURES))
    priors: dict[str, float] = field(default_factory=lambda: dict(QUALI_PRIORS))

    def prepare(self, ctx: EventContext) -> Signal:
        wanted = self.features[ctx.checkpoint]
        cols: dict[str, np.ndarray] = {}
        if "rating" in wanted:
            mu, var = self.beliefs(ctx, quali=False)
            cols["rating"] = predictive_log_strength(mu, var, self.params.beta)
        if "rating_q" in wanted:
            mu, var = self.beliefs(ctx, quali=True)
            cols["rating_q"] = predictive_log_strength(mu, var, self.params.beta)
        if "quali_pace" in wanted:
            cols["quali_pace"] = quali_pace_signal(ctx, decay=self.pace_decay)
        if "fp_pace" in wanted:
            cols["fp_pace"] = practice_pace(ctx)
        if "sq" in wanted:
            cols["sq"] = _log_rank(sprint_quali_ranks(ctx), ctx.n)
        x = np.column_stack([cols[f] for f in wanted])
        return Signal(x, np.zeros(ctx.n), 0.0)


@dataclass(frozen=True, slots=True)
class Knockout:
    """P(reaches Q3), P(out in Q2), P(out in Q1) per car, from the position distribution."""

    q3: np.ndarray
    out_q2: np.ndarray
    out_q1: np.ndarray


def knockout_probs(fc: Forecast, season: int) -> Knockout | None:
    """Knockout-stage probabilities under the season's format (``None`` before 2006 or for
    fields too small for three parts)."""
    try:
        fmt = knockout_format(season, fc.n)
    except (NotImplementedError, ValueError):
        return None
    pos = fc.position_probs  # n x n: P(car i qualifies in place k+1)
    q3 = pos[:, : fc.n - fmt.q1_eliminated - fmt.q2_eliminated].sum(axis=1)
    out_q1 = pos[:, fc.n - fmt.q1_eliminated :].sum(axis=1)
    return Knockout(q3=q3, out_q2=1.0 - q3 - out_q1, out_q1=out_q1)
