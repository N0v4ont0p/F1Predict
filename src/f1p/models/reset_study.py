"""The regulation-reset study (docs/plan/RESET_PLAN.md): fit the filter's reset parameters,
score them race by race, and cross-validate them one reset at a time.

Only the reset parameters (:attr:`RatingParams.RESET_FIELDS`) are fitted; every other value
stays as in the base parameters (``FITTED_V1``), so an effect is the reset parameters' own.
The objective is the filter's prequential race log-likelihood, as for ``FITTED_V1``, plus a
shrinkage penalty that pulls each parameter towards "no change" — a Gaussian prior on its log
(sd :data:`PRIOR_SD`). Resets are rare (five between 1980 and 2013), so the penalty matters.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from f1p.core.regulations import MAJOR_RESETS
from f1p.models.ratings import RaceObs, RatingFilter, RatingParams

FloatArray = NDArray[np.float64]

# The pre-registered variants (no others are tried): which reset parameters each frees.
VARIANTS: dict[str, tuple[str, ...]] = {
    "H1": ("reset_pu_ratio",),
    "H2": ("reset_drift", "reset_drift_races"),
    "H3": ("reset_carryover_ratio",),
    "H1+H3": ("reset_pu_ratio", "reset_carryover_ratio"),
    "H2+H3": ("reset_drift", "reset_drift_races", "reset_carryover_ratio"),
}
# Simplicity order for the decision rule (fewer, then gentler, parameters first).
SIMPLICITY = ("H3", "H1", "H2", "H1+H3", "H2+H3")
# Prior sd of each parameter's log, around 1 ("no change"); the decay length has no prior
# of its own (it only matters when reset_drift > 1).
PRIOR_SD: dict[str, float] = {
    "reset_pu_ratio": 0.5,
    "reset_carryover_ratio": 0.5,
    "reset_drift": 1.0,
}
EARLY_ROUNDS = 8  # "early" in a reset season: the races the parameters act on most

# The adaptive response (docs/plan/CALIBRATION_ADAPTIVE_PLAN.md, ADR), pre-registered: one
# strength κ each, with a half-normal prior (sd 1) that pulls it towards 0, "no response".
ADR_VARIANTS: dict[str, tuple[str, ...]] = {
    "R1": ("adapt_reset",),
    "R2": ("adapt_all",),
}
ADR_PRIOR_SD = 1.0
ADR_GRID = 16  # log-spaced κ values tried before the 1-D refinement


def race_logliks(races: Sequence[RaceObs], params: RatingParams) -> FloatArray:
    """The prequential log-likelihood of every race (0 for non-race sessions)."""
    f = RatingFilter(params)
    return np.array([f.update(r) for r in races])


def penalty(params: RatingParams, names: Sequence[str]) -> float:
    total = 0.0
    for n in names:
        if n in RatingParams.ADAPT_FIELDS:  # half-normal on κ itself
            total += 0.5 * (getattr(params, n) / ADR_PRIOR_SD) ** 2
            continue
        sd = PRIOR_SD.get(n)
        if sd is not None:
            total += 0.5 * (math.log(max(getattr(params, n), 1e-9)) / sd) ** 2
    return total


def _to_unit(names: Sequence[str], params: RatingParams) -> FloatArray:
    out = []
    for n in names:
        lo, hi = RatingParams.BOUNDS[n]
        u = min(max((getattr(params, n) - lo) / (hi - lo), 1e-6), 1 - 1e-6)
        out.append(math.log(u / (1 - u)))
    return np.array(out)


def _from_unit(names: Sequence[str], x: FloatArray, base: RatingParams) -> RatingParams:
    vals = {}
    for n, xi in zip(names, x, strict=True):
        lo, hi = RatingParams.BOUNDS[n]
        vals[n] = lo + (hi - lo) / (1.0 + math.exp(-float(xi)))
    return replace(base, **vals)


@dataclass(frozen=True)
class Fit:
    variant: str
    params: RatingParams
    loglik: float  # scored races' prequential log-likelihood (no penalty)
    objective: float  # loglik - penalty


def scored_mask(
    races: Sequence[RaceObs],
    *,
    score_from: int,
    train_until: int,
    exclude: Sequence[int] = (),
) -> NDArray[np.bool_]:
    return np.array(
        [
            r.is_race and score_from <= r.season <= train_until and r.season not in exclude
            for r in races
        ]
    )


def fit_variant(
    races: Sequence[RaceObs],
    variant: str,
    base: RatingParams,
    *,
    score_from: int,
    train_until: int,
    exclude: Sequence[int] = (),
    max_iter: int = 600,
) -> Fit:
    """The variant's parameters maximising the penalised prequential log-likelihood of the
    races in ``[score_from, train_until]`` outside ``exclude`` (the filter still runs through
    every race, so no state is lost). Races after ``train_until`` are never run."""
    from scipy.optimize import minimize

    run = [r for r in races if r.season <= train_until]
    mask = scored_mask(run, score_from=score_from, train_until=train_until, exclude=exclude)
    if variant in ADR_VARIANTS:
        return _fit_strength(run, mask, variant, base)
    names = VARIANTS[variant]

    def objective(x: FloatArray) -> float:
        p = _from_unit(names, x, base)
        ll = float(race_logliks(run, p)[mask].sum())
        return -(ll - penalty(p, names)) if np.isfinite(ll) else 1e12

    # start at "no change" (the decay length at its default)
    res = minimize(
        objective,
        _to_unit(names, base),
        method="Nelder-Mead",
        options={"maxiter": max_iter, "xatol": 1e-3, "fatol": 1e-3, "adaptive": True},
    )
    p = _from_unit(names, res.x, base)
    ll = float(race_logliks(run, p)[mask].sum())
    return Fit(variant, p, ll, ll - penalty(p, names))


def _fit_strength(
    run: Sequence[RaceObs], mask: NDArray[np.bool_], variant: str, base: RatingParams
) -> Fit:
    """An ADR variant's one strength κ: a log-spaced grid over the bounds (0 included), then
    a bounded 1-D refinement between the best point's neighbours. A κ whose filter diverges
    (a non-finite likelihood: large κ compounds the widening) scores -inf. κ stays 0 unless
    a κ > 0 beats "no response" on the penalised objective."""
    from scipy.optimize import minimize_scalar

    (name,) = ADR_VARIANTS[variant]
    lo, hi = RatingParams.BOUNDS[name]

    def score(k: float) -> tuple[RatingParams, float, float]:
        p = replace(base, **{name: k})
        ll = float(race_logliks(run, p)[mask].sum())
        if not np.isfinite(ll):
            return p, ll, -math.inf
        return p, ll, ll - penalty(p, (name,))

    grid = [lo, *np.geomspace(1e-3, hi, ADR_GRID).tolist()]
    tried = [score(k) for k in grid]
    i = max(range(len(grid)), key=lambda j: tried[j][2])
    a, b = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]

    def negative(k: float) -> float:
        return -score(float(k))[2]

    res: Any = minimize_scalar(negative, bounds=(a, b), method="bounded", options={"xatol": 1e-4})
    best = max([tried[i], score(float(res.x))], key=lambda s: s[2])
    return Fit(variant, best[0], best[1], best[2])


@dataclass(frozen=True)
class Block:
    """Scores of one reset season, the variant against the base."""

    season: int
    early: float  # Δ log-likelihood over the first EARLY_ROUNDS races
    season_total: float  # Δ over the whole season
    next_season: float  # Δ over the season after (does the filter re-converge?)


def reset_blocks(
    races: Sequence[RaceObs],
    params: RatingParams,
    base: RatingParams,
    seasons: Sequence[int],
) -> list[Block]:
    """Δ prequential log-likelihood (params - base) in each reset season of ``seasons``."""
    last = max(seasons) + 1
    run = [r for r in races if r.season <= last]
    d = race_logliks(run, params) - race_logliks(run, base)
    out = []
    for s in seasons:
        season = np.array([r.is_race and r.season == s for r in run])
        early = season & np.array([r.round <= EARLY_ROUNDS for r in run])
        after = np.array([r.is_race and r.season == s + 1 for r in run])
        out.append(Block(s, float(d[early].sum()), float(d[season].sum()), float(d[after].sum())))
    return out


@dataclass(frozen=True)
class Fold:
    held_out: int
    params: RatingParams
    block: Block


def leave_one_reset_out(
    races: Sequence[RaceObs],
    variant: str,
    base: RatingParams,
    *,
    resets: Sequence[int],
    score_from: int,
    train_until: int,
) -> list[Fold]:
    """For each reset in ``resets``: fit the variant on every scored race except that reset
    season's, then score the held-out reset season against the base."""
    folds = []
    for s in resets:
        fit = fit_variant(
            races,
            variant,
            base,
            score_from=score_from,
            train_until=train_until,
            exclude=(s,),
        )
        folds.append(Fold(s, fit.params, reset_blocks(races, fit.params, base, [s])[0]))
    return folds


def resets_between(lo: int, hi: int) -> list[int]:
    return [s for s in MAJOR_RESETS if lo <= s <= hi]


# Eras for R2's cross-validation: from one reset to the season before the next (the last one
# ends at ``until``), starting 1983-2022.
def eras(until: int) -> list[tuple[int, int]]:
    starts = resets_between(1983, min(2022, until))
    ends = [s - 1 for s in starts[1:]] + [until]
    return list(zip(starts, ends, strict=True))


@dataclass(frozen=True)
class EraFold:
    era: tuple[int, int]
    params: RatingParams
    gain: float  # Δ prequential log-likelihood over the held-out era's races


def leave_one_era_out(
    races: Sequence[RaceObs],
    variant: str,
    base: RatingParams,
    *,
    score_from: int,
    train_until: int,
) -> list[EraFold]:
    """For each era: fit the variant on every scored race outside it, then score the era's
    races against the base."""
    run = [r for r in races if r.season <= train_until]
    base_ll = race_logliks(run, base)
    out = []
    for lo, hi in eras(train_until):
        held = tuple(range(lo, hi + 1))
        fit = fit_variant(
            run, variant, base, score_from=score_from, train_until=train_until, exclude=held
        )
        inside = np.array([r.is_race and lo <= r.season <= hi for r in run])
        d = race_logliks(run, fit.params) - base_ll
        out.append(EraFold((lo, hi), fit.params, float(d[inside].sum())))
    return out
