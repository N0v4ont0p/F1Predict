"""Ranking & regression metrics for race-result prediction.

All metrics operate per-race then average across races, because absolute position error
is only meaningful within a single classification.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return 0.0
    rho, _ = spearmanr(a, b)
    return 0.0 if np.isnan(rho) else float(rho)


def race_metrics(pred: pd.DataFrame, actual_col: str = "target_position",
                 pred_col: str = "pred_position") -> dict[str, float]:
    """Metrics for a single race prediction frame."""
    a = pred[actual_col].to_numpy()
    p = pred[pred_col].to_numpy()
    pred_rank = pred[pred_col].rank(method="first").to_numpy()
    winner_actual = pred.loc[pred[actual_col].idxmin(), "driver_id"]
    winner_pred = pred.loc[pred[pred_col].idxmin(), "driver_id"]
    podium_actual = set(pred.nsmallest(3, actual_col)["driver_id"])
    podium_pred = set(pred.nsmallest(3, pred_col)["driver_id"])
    return {
        "mae": float(np.mean(np.abs(pred_rank - a))),
        "spearman": _safe_spearman(p, a),
        "winner_hit": float(winner_actual == winner_pred),
        "podium_overlap": len(podium_actual & podium_pred) / 3.0,
    }


def aggregate_metrics(per_race: list[dict[str, float]]) -> dict[str, float]:
    if not per_race:
        return {"mae": float("nan"), "spearman": float("nan"),
                "winner_accuracy": float("nan"), "podium_accuracy": float("nan"),
                "n_races": 0}
    df = pd.DataFrame(per_race)
    return {
        "mae": float(df["mae"].mean()),
        "spearman": float(df["spearman"].mean()),
        "winner_accuracy": float(df["winner_hit"].mean()),
        "podium_accuracy": float(df["podium_overlap"].mean()),
        "n_races": int(len(df)),
    }


# --------------------------------------------------------------------------- calibration
def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean squared error between predicted probabilities and binary outcomes (0 == perfect).

    A proper scoring rule for probabilistic forecasts (e.g. P(win) vs. did-win). Lower is
    better; 0.0 is a perfect, perfectly-confident forecaster.
    """
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if p.size == 0:
        return float("nan")
    return float(np.mean((p - y) ** 2))


def top_n_accuracy(pred_rank: np.ndarray, actual_rank: np.ndarray, n: int = 3) -> float:
    """Fraction of the predicted top-``n`` that actually finished in the top-``n``."""
    pr = np.asarray(pred_rank)
    ar = np.asarray(actual_rank)
    pred_set = set(np.argsort(pr)[:n])
    actual_set = set(np.argsort(ar)[:n])
    return len(pred_set & actual_set) / float(n)


def reliability_curve(probs: np.ndarray, outcomes: np.ndarray, bins: int = 10):
    """Calibration (reliability) curve: mean predicted prob vs. observed frequency per bin.

    Returns ``(bin_centres, observed_freq, counts)`` for plotting a calibration diagram.
    """
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    centres, obs, counts = [], [], []
    for b in range(bins):
        m = idx == b
        counts.append(int(m.sum()))
        centres.append(float(p[m].mean()) if m.any() else (edges[b] + edges[b + 1]) / 2)
        obs.append(float(y[m].mean()) if m.any() else float("nan"))
    return np.array(centres), np.array(obs), np.array(counts)
