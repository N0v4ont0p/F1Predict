"""Linear Plackett-Luce with stage-wise temperatures.

``η_i = x_i · w`` is each car's log-strength. The finishing order is chosen one place at a time;
at stage ``k`` (who takes P(k+1) among those left) the choice is a softmax of ``s_k · η`` with
``s_k = exp(θ_k)`` for the first ``K`` stages and ``s = 1`` after. Plain Plackett-Luce is
``θ = 0``. In F1 the front of the field is far more predictable than one temperature allows:
fitted per stage, the winner's temperature is about twice the midfield's.

Only classified finishers enter the likelihood (retirements are censored). Gradients are
exact and the computation is vectorised over races, so an L-BFGS fit over hundreds of races
takes well under a second.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
Sample = tuple[FloatArray, NDArray[np.intp]]  # (features n x p, finisher indices best-first)


@dataclass(frozen=True, slots=True)
class Padded:
    """Finishers' features, best first, padded to a common length."""

    x: FloatArray  # (R, M, p)
    mask: NDArray[np.bool_]  # (R, M)
    z: FloatArray  # (R,) race-level covariate that scales the temperature (e.g. rain)

    @classmethod
    def from_samples(
        cls, samples: Sequence[Sample], race_z: Sequence[float] | FloatArray | None = None
    ) -> Padded:
        keep = [i for i, (_, o) in enumerate(samples) if o.size >= 2]
        usable = [samples[i] for i in keep]
        p = usable[0][0].shape[1]
        m = max(o.size for _, o in usable)
        xs = np.zeros((len(usable), m, p))
        mask = np.zeros((len(usable), m), dtype=bool)
        for r, (x, o) in enumerate(usable):
            xs[r, : o.size] = x[o]
            mask[r, : o.size] = True
        z = np.zeros(len(usable)) if race_z is None else np.asarray(race_z, float)[keep]
        return cls(xs, mask, z)


@dataclass(frozen=True, slots=True)
class PLFit:
    """A fitted combiner: weights, per-place log-temperatures and the race-covariate effect.

    A race with covariate ``z`` has log-strengths ``exp(gamma · z) · (x · w)``.
    """

    w: FloatArray
    theta: FloatArray
    gamma: float = 0.0

    def log_strength(self, x: FloatArray, z: float = 0.0) -> FloatArray:
        return float(np.exp(self.gamma * z)) * (x @ self.w)


def staged_loglik_grad(
    w: FloatArray, theta: FloatArray, data: Padded
) -> tuple[float, FloatArray, FloatArray]:
    """Log-likelihood and its gradients in ``w`` and ``θ`` (``len(θ) = K`` staged places)."""
    ll, gw, gt, _ = loglik_grads(w, theta, 0.0, data)
    return ll, gw, gt


def loglik_grads(
    w: FloatArray, theta: FloatArray, gamma: float, data: Padded
) -> tuple[float, FloatArray, FloatArray, float]:
    """Log-likelihood and gradients in ``w``, ``θ`` and ``gamma``."""
    x, mask = data.x, data.mask
    scale = np.exp(gamma * data.z)[:, None]  # (R, 1)
    base = np.where(mask, x @ w, 0.0)  # (R, M)
    eta = base * scale
    neg = np.where(mask, eta, -np.inf)
    n_stages = theta.size
    total = 0.0
    g_eta = np.zeros_like(eta)
    g_theta = np.zeros_like(theta)
    cols = np.arange(eta.shape[1])

    for k in range(min(n_stages, eta.shape[1])):
        rows = mask[:, k] & (mask.sum(axis=1) > k + 1)  # a real choice remains at stage k
        if not rows.any():
            continue
        s = float(np.exp(theta[k]))
        z = np.where(cols[None, :] >= k, s * neg, -np.inf)[rows]
        zmax = z.max(axis=1, keepdims=True)
        e = np.exp(z - zmax)
        tot = e.sum(axis=1, keepdims=True)
        pi = e / tot
        lse = (zmax + np.log(tot))[:, 0]
        total += float(np.sum(s * eta[rows, k] - lse))
        contrib = -s * pi
        contrib[:, k] += s
        g_eta[rows] += contrib
        g_theta[k] += float(np.sum(s * (eta[rows, k] - (pi * eta[rows]).sum(axis=1))))

    # Plain Plackett-Luce for the places after the staged ones.
    rest = mask & (cols[None, :] >= n_stages)
    c = np.where(rest, eta, -np.inf).max(axis=1, keepdims=True)
    c = np.where(np.isfinite(c), c, 0.0)
    e = np.where(rest, np.exp(eta - c), 0.0)
    suffix = np.cumsum(e[:, ::-1], axis=1)[:, ::-1]
    live = rest & (suffix > 0)
    log_s = np.log(np.where(live, suffix, 1.0))
    total += float(np.sum(np.where(live, eta - c - log_s, 0.0)))
    inv = np.cumsum(np.where(live, 1.0 / np.where(live, suffix, 1.0), 0.0), axis=1)
    g_eta += np.where(live, 1.0 - e * inv, 0.0)

    g_w = np.einsum("rmp,rm->p", x, g_eta * scale)
    g_gamma = float(np.sum(data.z[:, None] * eta * g_eta))
    return total, g_w, g_theta, g_gamma


def loglik_and_grad(w: FloatArray, samples: Sequence[Sample]) -> tuple[float, FloatArray]:
    """Plain Plackett-Luce log-likelihood and gradient in ``w``."""
    usable = [(x, o) for x, o in samples if o.size >= 2]
    if not usable:
        return 0.0, np.zeros_like(w)
    ll, g, _ = staged_loglik_grad(w, np.zeros(0), Padded.from_samples(usable))
    return ll, g


def fit(
    samples: Sequence[Sample],
    *,
    prior: FloatArray,
    l2: float = 1.0,
    bounds: Sequence[tuple[float | None, float | None]] | None = None,
) -> FloatArray:
    """MAP weights under a Gaussian prior ``N(prior, 1/l2)`` (prior returned if no data)."""
    w, _ = fit_staged(samples, prior=prior, l2=l2, bounds=bounds, n_stages=0)
    return w


def fit_staged(
    samples: Sequence[Sample],
    *,
    prior: FloatArray,
    n_stages: int,
    l2: float = 1.0,
    l2_theta: float = 4.0,
    bounds: Sequence[tuple[float | None, float | None]] | None = None,
    basis: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray]:
    """MAP ``(w, θ)`` with ``θ = basis @ b`` (``basis`` defaults to one free ``θ`` per stage);
    ``b`` has a ``N(0, 1/l2_theta)`` prior, i.e. plain Plackett-Luce."""
    f = fit_model(
        samples,
        prior=prior,
        n_stages=n_stages,
        l2=l2,
        l2_theta=l2_theta,
        bounds=bounds,
        basis=basis,
    )
    return f.w, f.theta


def fit_model(
    samples: Sequence[Sample],
    *,
    prior: FloatArray,
    n_stages: int,
    l2: float = 1.0,
    l2_theta: float = 4.0,
    l2_gamma: float = 4.0,
    bounds: Sequence[tuple[float | None, float | None]] | None = None,
    basis: FloatArray | None = None,
    race_z: Sequence[float] | FloatArray | None = None,
) -> PLFit:
    """MAP combiner. With ``race_z`` (one value per sample) the effect ``gamma`` of that
    race-level covariate on the temperature is fitted too (prior ``N(0, 1/l2_gamma)``)."""
    phi = np.eye(n_stages) if basis is None else np.asarray(basis, dtype=np.float64)
    if phi.shape[0] != n_stages:
        raise ValueError("basis must have one row per stage")
    if not any(o.size >= 2 for _, o in samples):
        return PLFit(prior.copy(), np.zeros(n_stages))
    data = Padded.from_samples(samples, race_z)
    p, q = prior.size, phi.shape[1]
    use_gamma = race_z is not None
    from scipy.optimize import minimize

    def objective(v: FloatArray) -> tuple[float, FloatArray]:
        w, b = v[:p], v[p : p + q]
        gamma = float(v[-1]) if use_gamma else 0.0
        ll, gw, gt, gg = loglik_grads(w, phi @ b, gamma, data)
        dw = w - prior
        obj = -ll + 0.5 * l2 * float(dw @ dw) + 0.5 * l2_theta * float(b @ b)
        grad = [-gw + l2 * dw, -(phi.T @ gt) + l2_theta * b]
        if use_gamma:
            obj += 0.5 * l2_gamma * gamma**2
            grad.append(np.array([-gg + l2_gamma * gamma]))
        return obj, np.concatenate(grad)

    b_bounds = [(-2.0, 2.0)] * q if basis is None else [(None, None)] * q
    all_bounds = list(bounds or [(None, None)] * p) + b_bounds
    v0 = np.concatenate([prior, np.zeros(q)])
    if use_gamma:
        all_bounds.append((-3.0, 3.0))
        v0 = np.append(v0, 0.0)
    res = minimize(objective, v0, jac=True, method="L-BFGS-B", bounds=all_bounds)
    v = np.asarray(res.x, dtype=np.float64)
    return PLFit(v[:p], phi @ v[p : p + q], float(v[-1]) if use_gamma else 0.0)


MAX_STAGES = 26  # more places than any modern field has finishers


def smooth_stage_basis(n_stages: int = MAX_STAGES) -> FloatArray:
    """Log-temperature quadratic in the place: ``θ_k = b1·k/10 + b2·(k/10)²`` (``θ_0 = 0``,
    so the combiner's own scale is the winner's temperature)."""
    k = np.arange(n_stages, dtype=np.float64) / 10.0
    return np.column_stack([k, k**2])
