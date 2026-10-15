"""Stacking (plan item 8): a log-linear pool of the members' Plackett-Luce strengths.

For an event, member ``m`` gives centred log-strengths ``η_m``; the stack's log-strength is
``Σ_m w_m · η_m`` with smooth stage temperatures, i.e. the same linear Plackett-Luce combiner
as the world model, applied to models instead of signals. Weights are fitted **per season and
checkpoint on earlier seasons only** (by default the previous four), shrunk towards an equal
split, and bounded at zero. Retirements come from the first member that forecasts them (the
world model).

The stack runs on the forecasts a rolling-origin evaluation already produced
(:func:`stack_result`), so every member's forecast is itself causal and the stack adds no
look-ahead. For a live event, :func:`pool` applies the latest weights to the members' live
forecasts.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import NDArray

from f1p.eval.context import Tables, build_contexts
from f1p.eval.forecast import Forecast
from f1p.eval.harness import EvalResult
from f1p.eval.metrics import Outcome, calibration_pairs, score
from f1p.models import linear_pl

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PoolFit:
    members: tuple[str, ...]
    fit: linear_pl.PLFit

    @property
    def weights(self) -> dict[str, float]:
        return {m: float(w) for m, w in zip(self.members, self.fit.w, strict=True)}


def _centred(fc: Forecast, drivers: Sequence[str]) -> FloatArray:
    assert fc.log_strength is not None
    index = {d: i for i, d in enumerate(fc.drivers)}
    ls = np.array([fc.log_strength[index[d]] for d in drivers])
    return ls - ls.mean()


def design(forecasts: Sequence[Forecast], drivers: Sequence[str]) -> FloatArray:
    return np.column_stack([_centred(fc, drivers) for fc in forecasts])


def fit_pool(
    samples: list[tuple[FloatArray, NDArray[np.intp]]],
    members: Sequence[str],
    *,
    l2: float = 2.0,
) -> PoolFit:
    m = len(members)
    n_stages = linear_pl.MAX_STAGES
    fit = linear_pl.fit_model(
        samples,
        prior=np.full(m, 1.0 / m),
        n_stages=n_stages,
        l2=l2,
        bounds=[(0.0, None)] * m,
        basis=linear_pl.smooth_stage_basis(n_stages),
    )
    return PoolFit(tuple(members), fit)


def pool(
    fit: PoolFit, forecasts: Sequence[Forecast], *, name: str = "stack", seed: int | None = None
) -> Forecast:
    """The pooled forecast of one event (``forecasts`` in ``fit.members`` order)."""
    first = forecasts[0]
    drivers = first.drivers
    ls = fit.fit.log_strength(design(forecasts, drivers))
    p_finish = next((f.p_finish for f in forecasts if f.p_finish is not None), None)
    if p_finish is not None and first.p_finish is None:
        # align to the first member's driver order
        src = next(f for f in forecasts if f.p_finish is not None)
        idx = {d: i for i, d in enumerate(src.drivers)}
        p_finish = np.array([src.p_finish[idx[d]] for d in drivers])  # type: ignore[index]
    return Forecast.from_log_strength(
        first.event_id,
        first.checkpoint,
        name,
        drivers,
        ls,
        seed=seed,
        meta={f"w_{m}": w for m, w in fit.weights.items()},
        p_finish=p_finish,
        stage_scales=np.exp(fit.fit.theta),
    )


def stack_result(
    tables: Tables,
    result: EvalResult,
    members: Sequence[str],
    *,
    seasons: tuple[int, int],
    training_seasons: int = 4,
    name: str = "stack",
) -> tuple[EvalResult, dict[tuple[int, str], PoolFit]]:
    """Add the stack to ``result`` (which must hold every member's forecasts,
    ``keep_forecasts=True``, for the test seasons and the training seasons before them).

    Returns the per-event scores of the members and the stack on events every member
    forecast, and the fitted weights per (season, checkpoint)."""
    outcomes: dict[str, Outcome] = {}
    seasons_of: dict[str, int] = {}
    for ctx, o, _ in build_contexts(
        tables, seasons=(seasons[0] - training_seasons, seasons[1]), checkpoints=("C1",)
    ):
        outcomes[ctx.event_id] = o
        seasons_of[ctx.event_id] = ctx.season
    by_event: dict[tuple[str, str], list[Forecast]] = {}
    for (model, cp, eid), fc in result.forecasts.items():
        if model in members and fc.log_strength is not None:
            by_event.setdefault((eid, cp), []).append(fc)
    complete = {
        k: sorted(v, key=lambda f: list(members).index(f.model))
        for k, v in by_event.items()
        if {f.model for f in v} == set(members) and k[0] in outcomes
    }
    fits: dict[tuple[int, str], PoolFit] = {}
    rows: list[dict[str, object]] = []
    calibration: dict[tuple[str, str, int], tuple[list[np.ndarray], list[np.ndarray]]] = {}
    for (eid, cp), fcs in sorted(complete.items(), key=lambda kv: (seasons_of[kv[0][0]], kv[0])):
        season = seasons_of[eid]
        if not seasons[0] <= season <= seasons[1]:
            continue
        key = (season, cp)
        if key not in fits:
            samples = []
            for (e2, c2), f2 in complete.items():
                if c2 != cp or not season - training_seasons <= seasons_of[e2] < season:
                    continue
                drivers = f2[0].drivers
                idx = {d: i for i, d in enumerate(drivers)}
                fin = np.array([idx[d] for d in outcomes[e2].classified if d in idx], np.intp)
                samples.append((design(f2, drivers), fin))
            if len(samples) < 10:
                continue
            fits[key] = fit_pool(samples, members)
        stacked = pool(fits[key], fcs, name=name)
        o = outcomes[eid]
        for fc in [*fcs, stacked]:
            rows.append(
                {
                    "model": fc.model,
                    "target": "race",
                    "checkpoint": cp,
                    "event_id": eid,
                    "season": season,
                    "round": int(eid.split("-R")[-1]),
                    "field_added": 0,
                    **score(fc, o),
                }
            )
            for k in (3, 10):
                p, y = calibration_pairs(fc, o, k)
                bucket = calibration.setdefault((fc.model, cp, k), ([], []))
                bucket[0].append(p)
                bucket[1].append(y)
    per_event = pl.DataFrame(rows) if rows else pl.DataFrame()
    cal = {k: (np.concatenate(p), np.concatenate(y)) for k, (p, y) in calibration.items()}
    return EvalResult(per_event, cal, {}), fits
