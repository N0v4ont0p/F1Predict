from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import LogisticRegression, Ridge

from f1p.eval import harness
from f1p.lake.curate_f1db import Curated
from f1p.models import tfm


def test_bradley_terry_recovers_strengths() -> None:
    s = np.array([1.2, 0.4, 0.0, -0.3, -1.3])
    s -= s.mean()
    p = 1.0 / (1.0 + np.exp(-(s[:, None] - s[None, :])))
    np.fill_diagonal(p, 0.0)
    np.testing.assert_allclose(tfm.bradley_terry(p), s, atol=1e-8)
    # noisy pair probabilities: the order survives
    rng = np.random.default_rng(0)
    noisy = np.clip(p + rng.normal(0, 0.03, p.shape), 0.01, 0.99)
    noisy = 0.5 * (noisy + (1 - noisy.T))
    np.fill_diagonal(noisy, 0.0)
    assert np.array_equal(np.argsort(-tfm.bradley_terry(noisy)), np.argsort(-s))


class _Fake(tfm.TFMRanker):
    """TabICL replaced by small scikit-learn models: the framing, context and mapping logic
    are what is tested (the TabICL checkpoint is not downloaded in CI)."""

    def _estimator(self, task: str) -> Any:
        return Ridge(alpha=1.0) if task == "regression" else LogisticRegression(max_iter=500)


def test_unknown_framing() -> None:
    with pytest.raises(ValueError, match="framing"):
        tfm.TFMRanker(framing="listwise")
    assert tfm.TFMRanker(framing="pairwise").name == "tabicl_pairwise"


@pytest.mark.parametrize("framing", ["pointwise", "pairwise"])
def test_framings_run_through_the_harness(curated: Curated, framing: str) -> None:
    model = _Fake(framing=framing, name=f"fake_{framing}", training_seasons=8)
    res = harness.run(curated.tables, [model], seasons=(2026, 2026), history_from=2017)
    df = res.per_event
    assert df.height > 0
    assert df["model"].unique().to_list() == [f"fake_{framing}"]
    assert df["winner_logloss"].is_finite().all()
    # better than a coin toss over the field: the features carry the order
    rho = df.filter(pl.col("checkpoint") == "C4")["spearman"].to_numpy()
    assert float(np.mean(rho)) > 0.3
