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

**Stack v2** (plan M6 §E): the members of a checkpoint are those that forecast it (``lapseq``
has no C1 forecast; the in-race models only C5). Optional **recalibrations** of the pool,
each fitted on the same training seasons and reported next to the raw pool (checked by the
winner ECE, ``ece_top1``):

* ``isotonic`` (``name_iso``): an isotonic map of P(win), fitted in-sample. On the select
  split it made winner log-loss significantly worse (+0.134, 95% CI +0.028…+0.243 over 44
  races): two seasons hold ~44 winners and a step function fitted to them is over-confident
  at its plateaus.
* ``temperature`` (``name_temp``): P(win) ∝ P(win)^τ — one smooth parameter.
* ``temperature_all`` (``name_tempall``): every log-strength times τ, so the whole order
  (podium, points, the full likelihood) is tempered, fitted on the finishers' likelihood.

Both temperatures are bounded to [0.5, 2], shrunk towards 1 (a penalty on log τ), and fitted
on **out-of-sample** pooled forecasts: each training season forecast by the pool fitted on
the other training seasons (leave one season out; all of them before the forecast season, so
nothing is seen early). With a single training season they fall back to in-sample.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from f1p.eval.context import Tables, build_contexts
from f1p.eval.forecast import Forecast, pl_log_likelihood
from f1p.eval.harness import EvalResult
from f1p.eval.metrics import Outcome, calibration_pairs, score
from f1p.models import linear_pl

FloatArray = NDArray[np.float64]


RECALIBRATIONS: dict[str, str] = {  # mode → suffix of the recalibrated stack's name
    "isotonic": "iso",
    "temperature": "temp",
    "temperature_all": "tempall",
}
TEMPERATURE_BOUNDS = (0.5, 2.0)
TEMPERATURE_L2 = 1.0  # penalty on (log τ)², against log-likelihoods summed over races


@dataclass(frozen=True)
class PoolFit:
    members: tuple[str, ...]
    fit: linear_pl.PLFit
    # fitted recalibrations by mode: an isotonic map, or a temperature τ
    calibrations: dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def calibrator(self) -> Any:
        """The isotonic P(win) map, if fitted."""
        return self.calibrations.get("isotonic")

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
    fit: PoolFit,
    forecasts: Sequence[Forecast],
    *,
    name: str = "stack",
    seed: int | None = None,
    scale: float = 1.0,
) -> Forecast:
    """The pooled forecast of one event (``forecasts`` in ``fit.members`` order); ``scale``
    multiplies the pooled log-strengths (a temperature on the whole order)."""
    first = forecasts[0]
    drivers = first.drivers
    ls = scale * fit.fit.log_strength(design(forecasts, drivers))
    p_finish = next((f.p_finish for f in forecasts if f.p_finish is not None), None)
    if p_finish is not None and first.p_finish is None:
        # align to the first member's driver order
        src = next(f for f in forecasts if f.p_finish is not None)
        idx = {d: i for i, d in enumerate(src.drivers)}
        p_finish = np.array([src.p_finish[idx[d]] for d in drivers])  # type: ignore[index]
    meta = {f"w_{m}": w for m, w in fit.weights.items()}
    if scale != 1.0:
        meta["temperature"] = scale
    return Forecast.from_log_strength(
        first.event_id,
        first.checkpoint,
        name,
        drivers,
        ls,
        seed=seed,
        meta=meta,
        p_finish=p_finish,
        stage_scales=np.exp(fit.fit.theta),
    )


def recalibrated(
    fit: PoolFit,
    fc: Forecast,
    forecasts: Sequence[Forecast],
    *,
    mode: str,
    name: str,
) -> Forecast:
    """The pooled forecast ``fc`` (of the members' ``forecasts``) under the pool's ``mode``
    recalibration. P(win) maps renormalise over the field and keep the favourites' order."""
    cal = fit.calibrations.get(mode)
    if cal is None:
        return replace(fc, model=name)
    if mode == "isotonic":
        p = np.clip(np.asarray(cal.predict(fc.p_win), dtype=np.float64), 1e-6, None)
        return replace(fc, model=name, win_probs=p / p.sum())
    if mode == "temperature":
        return replace(fc, model=name, win_probs=temper(fc.p_win, float(cal)))
    if mode == "temperature_all":
        return pool(fit, forecasts, name=name, scale=float(cal))
    raise ValueError(f"unknown recalibration {mode!r}; choose from {list(RECALIBRATIONS)}")


def temper(p_win: FloatArray, tau: float) -> FloatArray:
    """P(win)^τ, renormalised (τ > 1 sharpens, τ < 1 flattens)."""
    logp = tau * np.log(np.clip(p_win, 1e-12, None))
    e = np.exp(logp - logp.max())
    return e / e.sum()


def fit_calibrator(p_win: FloatArray, won: FloatArray) -> Any:
    """Isotonic regression of "won" on P(win) (needs both outcomes in the data)."""
    from sklearn.isotonic import IsotonicRegression

    if won.min() == won.max():
        return None
    return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p_win, won)


def _fit_log_tau(objective: Any, l2: float, bounds: tuple[float, float]) -> float:
    """τ maximising ``objective(τ) - l2·(log τ)²`` over ``bounds`` (searched on log τ)."""
    from scipy.optimize import minimize_scalar

    lo, hi = np.log(bounds[0]), np.log(bounds[1])
    res: Any = minimize_scalar(
        lambda u: -(objective(float(np.exp(u))) - l2 * u * u),
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": 1e-4},
    )
    return float(np.exp(res.x))


def fit_temperature(
    p_wins: Sequence[FloatArray],
    winners: Sequence[int],
    *,
    l2: float = TEMPERATURE_L2,
    bounds: tuple[float, float] = TEMPERATURE_BOUNDS,
) -> float:
    """τ for P(win)^τ: the winners' penalised log-likelihood, over races."""
    logs = [np.log(np.clip(p, 1e-12, None)) for p in p_wins]

    def loglik(tau: float) -> float:
        total = 0.0
        for lp, w in zip(logs, winners, strict=True):
            z = tau * lp
            total += float(z[w] - np.logaddexp.reduce(z))
        return total

    return _fit_log_tau(loglik, l2, bounds)


def fit_temperature_all(
    races: Sequence[tuple[FloatArray, NDArray[np.intp], FloatArray | None]],
    *,
    l2: float = TEMPERATURE_L2,
    bounds: tuple[float, float] = TEMPERATURE_BOUNDS,
) -> float:
    """τ for the whole order: (log-strengths, finishers, stage scales) per race, the
    finishers' penalised Plackett-Luce log-likelihood under τ · log-strength."""

    def loglik(tau: float) -> float:
        return sum(pl_log_likelihood(tau * ls, fin, ss) for ls, fin, ss in races)

    return _fit_log_tau(loglik, l2, bounds)


def stack_result(
    tables: Tables,
    result: EvalResult,
    members: Sequence[str],
    *,
    seasons: tuple[int, int],
    training_seasons: int = 4,
    name: str = "stack",
    recalibrate: Sequence[str] = (),
) -> tuple[EvalResult, dict[tuple[int, str], PoolFit]]:
    """Add the stack to ``result`` (which must hold every member's forecasts,
    ``keep_forecasts=True``, for the test seasons and the training seasons before them).

    At each checkpoint the members are those of ``members`` with forecasts there. Returns
    the per-event scores of those members and the stack (``name``, and ``name_<suffix>`` for
    each ``recalibrate`` mode, see :data:`RECALIBRATIONS`) on events every one of them
    forecast, and the fitted pools per (season, checkpoint)."""
    unknown = [m for m in recalibrate if m not in RECALIBRATIONS]
    if unknown:
        raise ValueError(f"unknown recalibration {unknown}; choose from {list(RECALIBRATIONS)}")
    outcomes: dict[str, Outcome] = {}
    seasons_of: dict[str, int] = {}
    for ctx, o, _ in build_contexts(
        tables, seasons=(seasons[0] - training_seasons, seasons[1]), checkpoints=("C1",)
    ):
        outcomes[ctx.event_id] = o
        seasons_of[ctx.event_id] = ctx.season
    by_event: dict[tuple[str, str], list[Forecast]] = {}
    present: dict[str, set[str]] = {}
    for (model, cp, eid), fc in result.forecasts.items():
        if model in members and fc.log_strength is not None:
            by_event.setdefault((eid, cp), []).append(fc)
            present.setdefault(cp, set()).add(model)
    cp_members = {cp: [m for m in members if m in got] for cp, got in present.items()}
    complete = {
        k: sorted(v, key=lambda f: list(members).index(f.model))
        for k, v in by_event.items()
        if {f.model for f in v} == set(cp_members[k[1]]) and k[0] in outcomes
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
                samples.append((design(f2, drivers), fin, f2))
            if len(samples) < 10:
                continue
            fitted = fit_pool([(x, f) for x, f, _ in samples], cp_members[cp])
            if recalibrate:
                seasons_in = [seasons_of[f2[0].event_id] for _, _, f2 in samples]
                cals = _fit_recalibrations(
                    fitted,
                    samples,
                    seasons_in,
                    cp_members[cp],
                    outcomes,
                    recalibrate,
                    name,
                )
                fitted = PoolFit(fitted.members, fitted.fit, cals)
            fits[key] = fitted
        stacked = pool(fits[key], fcs, name=name)
        out = [*fcs, stacked]
        for mode in recalibrate:
            out.append(
                recalibrated(
                    fits[key], stacked, fcs, mode=mode, name=f"{name}_{RECALIBRATIONS[mode]}"
                )
            )
        o = outcomes[eid]
        for fc in out:
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
            bucket = calibration.setdefault((fc.model, cp, 1), ([], []))
            bucket[0].append(fc.p_win)
            bucket[1].append(_won(fc, o))
    per_event = pl.DataFrame(rows) if rows else pl.DataFrame()
    cal = {k: (np.concatenate(p), np.concatenate(y)) for k, (p, y) in calibration.items()}
    return EvalResult(per_event, cal, {}), fits


def _won(fc: Forecast, o: Outcome) -> FloatArray:
    y = np.zeros(fc.n)
    if o.winner in fc.drivers:
        y[fc.drivers.index(o.winner)] = 1.0
    return y


def _fit_recalibrations(
    fitted: PoolFit,
    samples: list[tuple[FloatArray, NDArray[np.intp], list[Forecast]]],
    sample_seasons: list[int],
    members: Sequence[str],
    outcomes: dict[str, Outcome],
    modes: Sequence[str],
    name: str,
) -> dict[str, Any]:
    """Each recalibration of ``modes``, fitted on the pool's training races: in-sample for
    the isotonic map (as first specified), out-of-sample (leave one training season out) for
    the temperatures when there are at least two training seasons."""
    cals: dict[str, Any] = {}
    in_sample = [pool(fitted, f2, name=name) for _, _, f2 in samples]
    if "isotonic" in modes:
        p = np.concatenate([fc.p_win for fc in in_sample])
        y = np.concatenate([_won(fc, outcomes[fc.event_id]) for fc in in_sample])
        cals["isotonic"] = fit_calibrator(p, y)
    temps = [m for m in modes if m.startswith("temperature")]
    if not temps:
        return cals
    held_out = _held_out_pools(samples, sample_seasons, members, name) or in_sample
    if "temperature" in temps:
        won = [(fc.p_win, _winner(fc, outcomes[fc.event_id])) for fc in held_out]
        won = [(p, w) for p, w in won if w is not None]
        cals["temperature"] = (
            fit_temperature([p for p, _ in won], [w for _, w in won]) if won else None
        )
    if "temperature_all" in temps:
        races = []
        for fc in held_out:
            idx = {d: i for i, d in enumerate(fc.drivers)}
            fin = np.array([idx[d] for d in outcomes[fc.event_id].classified if d in idx], np.intp)
            if len(fin) >= 2 and fc.log_strength is not None:
                races.append((fc.log_strength, fin, fc.stage_scales))
        cals["temperature_all"] = fit_temperature_all(races) if races else None
    return cals


def _held_out_pools(
    samples: list[tuple[FloatArray, NDArray[np.intp], list[Forecast]]],
    sample_seasons: list[int],
    members: Sequence[str],
    name: str,
) -> list[Forecast]:
    """Each training race forecast by the pool fitted on the *other* training seasons (none
    with fewer than two seasons, or too few races outside a season)."""
    distinct = sorted(set(sample_seasons))
    if len(distinct) < 2:
        return []
    out: list[Forecast] = []
    for s in distinct:
        rest = [(x, f) for (x, f, _), si in zip(samples, sample_seasons, strict=True) if si != s]
        if len(rest) < 10:
            return []
        fit_s = fit_pool(rest, members)
        out += [
            pool(fit_s, f2, name=name)
            for (_, _, f2), si in zip(samples, sample_seasons, strict=True)
            if si == s
        ]
    return out


def _winner(fc: Forecast, o: Outcome) -> int | None:
    return fc.drivers.index(o.winner) if o.winner in fc.drivers else None
