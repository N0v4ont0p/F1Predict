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


def _staged_synthetic(
    w: np.ndarray, scales: np.ndarray, n_races: int, n: int, seed: int
) -> list[linear_pl.Sample]:
    from f1p.eval.forecast import sample_orders

    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_races):
        x = rng.normal(size=(n, w.size))
        orders, _ = sample_orders(x @ w, 1, rng, stage_scales=scales)
        out.append((x, orders[0].astype(np.intp)))
    return out


def test_staged_with_zero_theta_is_plain_pl() -> None:
    from f1p.eval.forecast import pl_log_likelihood

    samples = _synthetic(np.array([1.0, -0.5]), 20, 10, 2)
    w = np.array([0.7, 0.1])
    data = linear_pl.Padded.from_samples(samples)
    ll, _, _ = linear_pl.staged_loglik_grad(w, np.zeros(3), data)
    expect = sum(pl_log_likelihood(x @ w, o) for x, o in samples if o.size >= 2)
    assert ll == pytest.approx(expect, rel=1e-10)


def test_staged_loglik_matches_forecast_scoring() -> None:
    from f1p.eval.forecast import pl_log_likelihood

    samples = _synthetic(np.array([1.0, -0.5]), 15, 9, 4)
    w, theta = np.array([0.8, -0.2]), np.array([0.6, 0.3, -0.2])
    ll, _, _ = linear_pl.staged_loglik_grad(w, theta, linear_pl.Padded.from_samples(samples))
    expect = sum(pl_log_likelihood(x @ w, o, np.exp(theta)) for x, o in samples if o.size >= 2)
    assert ll == pytest.approx(expect, rel=1e-10)


def test_staged_gradients_are_exact() -> None:
    samples = _synthetic(np.array([1.0, -0.5]), 12, 9, 5)
    data = linear_pl.Padded.from_samples(samples)
    w, theta = np.array([0.4, 0.3]), np.array([0.5, -0.3, 0.2])
    _, gw, gt = linear_pl.staged_loglik_grad(w, theta, data)
    eps = 1e-6
    for i in range(w.size):
        e = np.zeros_like(w)
        e[i] = eps
        hi = linear_pl.staged_loglik_grad(w + e, theta, data)[0]
        lo = linear_pl.staged_loglik_grad(w - e, theta, data)[0]
        assert gw[i] == pytest.approx((hi - lo) / (2 * eps), rel=1e-5, abs=1e-6)
    for k in range(theta.size):
        e = np.zeros_like(theta)
        e[k] = eps
        hi = linear_pl.staged_loglik_grad(w, theta + e, data)[0]
        lo = linear_pl.staged_loglik_grad(w, theta - e, data)[0]
        assert gt[k] == pytest.approx((hi - lo) / (2 * eps), rel=1e-5, abs=1e-6)


def test_fit_staged_recovers_front_temperatures() -> None:
    truth_w, truth_s = np.array([1.0]), np.array([2.5, 1.6])
    samples = _staged_synthetic(truth_w, truth_s, 1500, 12, 11)
    w, theta = linear_pl.fit_staged(
        samples, prior=np.array([0.5]), n_stages=2, l2=1e-3, l2_theta=1e-3
    )
    np.testing.assert_allclose(w, truth_w, atol=0.1)
    np.testing.assert_allclose(np.exp(theta), truth_s, rtol=0.15)


def test_smooth_basis_fixes_the_winner_scale() -> None:
    phi = linear_pl.smooth_stage_basis(5)
    assert phi.shape == (5, 2)
    np.testing.assert_array_equal(phi[0], [0.0, 0.0])
    samples = _staged_synthetic(np.array([1.0]), np.exp(-0.3 * np.arange(10)), 400, 10, 3)
    _, theta = linear_pl.fit_staged(
        samples, prior=np.array([0.5]), n_stages=10, basis=linear_pl.smooth_stage_basis(10)
    )
    assert theta[0] == 0.0
    assert theta[5] < theta[1] < 0.0  # temperature falls down the order


def test_race_covariate_gradient_is_exact() -> None:
    samples = _synthetic(np.array([1.0, -0.5]), 10, 8, 9)
    z = np.linspace(0.0, 1.0, len(samples))
    data = linear_pl.Padded.from_samples(samples, z)
    w, theta, gamma = np.array([0.6, 0.2]), np.array([0.3, -0.1]), -0.4
    _, gw, _, gg = linear_pl.loglik_grads(w, theta, gamma, data)
    eps = 1e-6
    hi = linear_pl.loglik_grads(w, theta, gamma + eps, data)[0]
    lo = linear_pl.loglik_grads(w, theta, gamma - eps, data)[0]
    assert gg == pytest.approx((hi - lo) / (2 * eps), rel=1e-5, abs=1e-6)
    for i in range(w.size):
        e = np.zeros_like(w)
        e[i] = eps
        hi = linear_pl.loglik_grads(w + e, theta, gamma, data)[0]
        lo = linear_pl.loglik_grads(w - e, theta, gamma, data)[0]
        assert gw[i] == pytest.approx((hi - lo) / (2 * eps), rel=1e-5, abs=1e-6)


def test_fit_model_recovers_wet_race_randomness() -> None:
    rng = np.random.default_rng(21)
    truth = np.array([1.5])
    samples, z = [], []
    for _ in range(1200):
        wet = float(rng.random() < 0.3)
        x = rng.normal(size=(12, 1))
        order = pl_sample_orders((x @ truth) * np.exp(-0.7 * wet), 1, rng)[0]
        samples.append((x, order.astype(np.intp)))
        z.append(wet)
    fit = linear_pl.fit_model(
        samples, prior=np.array([1.0]), n_stages=0, race_z=z, l2=1e-3, l2_gamma=1e-3
    )
    assert fit.w[0] == pytest.approx(1.5, abs=0.12)
    assert fit.gamma == pytest.approx(-0.7, abs=0.15)
    np.testing.assert_allclose(fit.log_strength(np.ones((1, 1)), 1.0), 1.5 * np.exp(-0.7), atol=0.2)
