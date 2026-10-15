from __future__ import annotations

import numpy as np
import pytest

from f1p.eval.forecast import pl_sample_orders
from f1p.models import linear_pl


def _synthetic(w: np.ndarray, n_races: int, n: int, seed: int) -> list[linear_pl.Sample]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_races):
        x = rng.normal(size=(n, w.size))
        order = pl_sample_orders(x @ w, 1, rng)[0]
        # retirements: a random subset drops out; the rest keep their relative order
        finished = rng.random(n) > 0.25
        keep = order[finished[order]]
        out.append((x, keep.astype(np.intp)))
    return out


def test_gradient_is_exact() -> None:
    samples = _synthetic(np.array([1.0, -0.5]), 5, 8, 1)
    w = np.array([0.3, 0.2])
    _, g = linear_pl.loglik_and_grad(w, samples)
    eps = 1e-6
    for i in range(2):
        e = np.zeros(2)
        e[i] = eps
        hi, _ = linear_pl.loglik_and_grad(w + e, samples)
        lo, _ = linear_pl.loglik_and_grad(w - e, samples)
        assert g[i] == pytest.approx((hi - lo) / (2 * eps), rel=1e-5, abs=1e-6)


def test_fit_recovers_true_weights() -> None:
    truth = np.array([1.2, 0.4])
    samples = _synthetic(truth, 600, 12, 7)
    w = linear_pl.fit(samples, prior=np.zeros(2), l2=1e-3)
    np.testing.assert_allclose(w, truth, atol=0.12)


def test_no_data_returns_prior() -> None:
    prior = np.array([1.0, 0.5])
    np.testing.assert_array_equal(linear_pl.fit([], prior=prior), prior)


def test_bounds_respected() -> None:
    samples = _synthetic(np.array([-1.0]), 200, 8, 3)
    w = linear_pl.fit(samples, prior=np.array([0.5]), l2=1e-3, bounds=[(0.0, None)])
    assert w[0] == pytest.approx(0.0, abs=1e-8)
