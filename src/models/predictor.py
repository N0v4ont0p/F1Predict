"""The :class:`RacePredictor` — a fitted model plus the metadata needed to turn raw
position predictions into ranked race results and calibrated probabilities.

It persists as a single joblib artifact alongside a JSON model card.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class RacePredictor:
    model: Any
    feature_cols: list[str]
    family: str
    residual_std: float = 3.0          # std of position residuals — drives probabilities
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ prediction
    def predict_positions(self, X: pd.DataFrame) -> np.ndarray:
        """Raw predicted finishing position (continuous, lower == better)."""
        return self.model.predict(X[self.feature_cols].to_numpy())

    def predict_race(self, race_df: pd.DataFrame, season: int | None = None,
                     session: str | None = None) -> pd.DataFrame:
        """Return a ranked prediction table for a single session.

        Adds ``pred_position`` (continuous), ``predicted_rank`` (1..n) and probabilistic
        columns ``p_win``/``p_podium``/``p_points`` estimated analytically from the model's
        residual spread (a fast approximation; the simulation module produces full
        Monte-Carlo distributions). ``expected_points`` uses the points system appropriate to
        ``session``: race points for a race, the sprint table for a sprint, and zero for
        qualifying (grid order scores no championship points). ``session`` defaults to the
        model's own trained session.
        """
        session = (session or self.metadata.get("session") or "race").lower()
        df = race_df.copy().reset_index(drop=True)
        df["pred_position"] = self.predict_positions(df)
        df = df.sort_values("pred_position").reset_index(drop=True)
        df["predicted_rank"] = np.arange(1, len(df) + 1)
        probs = self._analytic_probabilities(df["pred_position"].to_numpy())
        df["p_win"] = probs["win"]
        df["p_podium"] = probs["podium"]
        df["p_points"] = probs["points"]
        df["expected_points"] = self._expected_points(probs, season, session)
        return df

    def _analytic_probabilities(self, mu: np.ndarray) -> dict[str, np.ndarray]:
        """Approximate P(win/podium/points) via pairwise Gaussian race-order sampling."""
        n = len(mu)
        sigma = max(self.residual_std, 0.5)
        rng = np.random.default_rng(0)
        draws = rng.normal(mu[None, :], sigma, size=(4000, n))
        order = np.argsort(draws, axis=1)
        ranks = np.empty_like(order)
        rows = np.arange(4000)[:, None]
        ranks[rows, order] = np.arange(n)[None, :]
        ranks = ranks + 1  # 1-based finishing position
        return {
            "win": (ranks == 1).mean(axis=0),
            "podium": (ranks <= 3).mean(axis=0),
            "points": (ranks <= 10).mean(axis=0),
        }

    def _expected_points(self, probs: dict[str, np.ndarray], season: int | None,
                         session: str = "race") -> np.ndarray:
        from ..utils.points import points_for_position
        season = season or 2025
        if session == "qualifying":
            # Qualifying decides the grid, not championship points.
            return np.zeros_like(probs["win"])
        pts_session = "S" if session == "sprint" else "R"
        # Expected points ~ P(win)*win_pts + marginal podium/points contributions.
        win_pts = points_for_position(1, season, session=pts_session)
        p3 = points_for_position(3, season, session=pts_session)
        p8 = points_for_position(8, season, session=pts_session)
        return (
            probs["win"] * win_pts
            + (probs["podium"] - probs["win"]) * p3
            + (probs["points"] - probs["podium"]) * p8
        )

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> Path:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        card = {
            "family": self.family,
            "n_features": len(self.feature_cols),
            "feature_cols": self.feature_cols,
            "residual_std": self.residual_std,
            **self.metadata,
        }
        path.with_suffix(".card.json").write_text(json.dumps(card, indent=2))
        return path

    @staticmethod
    def load(path: str | Path) -> "RacePredictor":
        import joblib

        return joblib.load(path)
