"""Proper scoring rules and ranking metrics for one forecast (plan §10.2).

All per-event metrics are oriented so that **lower is better**, except ``winner_hit`` and
``spearman`` (higher is better) — :data:`HIGHER_IS_BETTER` records this for reports.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from f1p.eval.forecast import Forecast, pl_log_likelihood

EPS = 1e-6
HIGHER_IS_BETTER = frozenset({"winner_hit", "spearman", "finisher_pl_loglik"})


@dataclass(frozen=True, slots=True)
class Outcome:
    """What happened: the classified finishers in order (best first)."""

    event_id: str
    classified: tuple[str, ...]

    @property
    def winner(self) -> str | None:
        return self.classified[0] if self.classified else None


def _rank(values: NDArray[np.float64]) -> NDArray[np.float64]:
    order = np.argsort(values, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1)
    # average ties
    uniq, inv, counts = np.unique(values, return_inverse=True, return_counts=True)
    if len(uniq) != len(values):
        sums = np.zeros(len(uniq))
        np.add.at(sums, inv, ranks)
        ranks = (sums / counts)[inv]
    return ranks


def spearman(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    if len(a) < 2:
        return float("nan")
    if np.all(a == a[0]) or np.all(b == b[0]):
        return 0.0  # a constant ranking carries no ordinal information
    ra, rb = _rank(a), _rank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def score(forecast: Forecast, outcome: Outcome) -> dict[str, float]:
    """Score one forecast. Drivers in the outcome but not the forecast are an error."""
    idx = {d: i for i, d in enumerate(forecast.drivers)}
    missing = [d for d in outcome.classified if d not in idx]
    if missing:
        raise ValueError(f"{forecast.event_id}: forecast has no entry for {missing}")
    n = forecast.n
    finishers = np.array([idx[d] for d in outcome.classified], dtype=np.intp)
    y_podium = np.zeros(n)
    y_podium[finishers[:3]] = 1.0
    y_points = np.zeros(n)
    y_points[finishers[:10]] = 1.0

    p_win = forecast.p_win
    p_podium = forecast.p_top(3)
    p_points = forecast.p_top(10)
    metrics: dict[str, float] = {
        "n_drivers": float(n),
        "n_classified": float(len(finishers)),
        "winner_logloss": float("nan"),
        "winner_hit": float("nan"),
        "podium_brier": float(np.mean((p_podium - y_podium) ** 2)),
        "points_brier": float(np.mean((p_points - y_points) ** 2)),
        "spearman": float("nan"),
        "finisher_pl_loglik": float("nan"),
    }
    if len(finishers):
        w = finishers[0]
        metrics["winner_logloss"] = float(-np.log(max(p_win[w], EPS)))
        # Ties for favourite share the credit, so a flat forecast cannot "hit" by list order.
        favourites = np.flatnonzero(np.isclose(p_win, p_win.max(), rtol=0.0, atol=1e-12))
        metrics["winner_hit"] = float(w in favourites) / len(favourites)
    if len(finishers) >= 2:
        # Rank by exact strengths when the model has them (no Monte Carlo noise), else by
        # expected position.
        predicted = (
            -forecast.log_strength
            if forecast.log_strength is not None
            else forecast.expected_position
        )
        metrics["spearman"] = spearman(
            predicted[finishers], np.arange(1, len(finishers) + 1, dtype=np.float64)
        )
        if forecast.log_strength is not None:
            metrics["finisher_pl_loglik"] = pl_log_likelihood(
                forecast.log_strength, finishers
            ) / len(finishers)
    return metrics


def calibration_pairs(
    forecast: Forecast, outcome: Outcome, k: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """(probability, outcome) pairs for "finishes in the top ``k``" — for reliability/ECE."""
    idx = {d: i for i, d in enumerate(forecast.drivers)}
    y = np.zeros(forecast.n)
    for d in outcome.classified[:k]:
        y[idx[d]] = 1.0
    return forecast.p_top(k), y


def expected_calibration_error(
    probs: NDArray[np.float64], outcomes: NDArray[np.float64], bins: int = 10
) -> float:
    """Weighted mean |observed frequency - mean predicted probability| over equal-width bins."""
    if probs.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(probs, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = which == b
        if mask.any():
            total += mask.sum() * abs(outcomes[mask].mean() - probs[mask].mean())
    return float(total / probs.size)


def reliability_table(
    probs: NDArray[np.float64], outcomes: NDArray[np.float64], bins: int = 10
) -> list[tuple[float, float, int]]:
    """(mean predicted, observed frequency, count) per non-empty bin."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(probs, edges[1:-1]), 0, bins - 1)
    out = []
    for b in range(bins):
        mask = which == b
        if mask.any():
            out.append((float(probs[mask].mean()), float(outcomes[mask].mean()), int(mask.sum())))
    return out
