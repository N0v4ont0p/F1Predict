"""Accelerated kernels with a NumPy reference (plan §11a.2).

``staged_orders_mlx`` draws finishing orders from the stage-wise-temperature
Plackett-Luce with retirements on the GPU through MLX (unified memory: inputs and outputs
are shared with NumPy, no device copies on Apple silicon). It is distribution-identical to
the NumPy sampler in :mod:`f1p.eval.forecast` — tested statistically — but uses MLX's own
counter-based RNG, so individual draws differ between back ends.

Back-end choice: ``F1P_SAMPLER=numpy|mlx|auto`` (default ``auto`` = MLX on Apple silicon
when it is installed, NumPy otherwise).
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from numpy.typing import NDArray

from f1p.silicon.device import detect

FloatArray = NDArray[np.float64]
DNF_OFFSET = 1e4  # retirements sort behind every classified car (exact in float32)
PLACED = 1e6  # placed cars sort first, in placing order (integers exact in float32)


def mlx_available() -> bool:
    try:
        import mlx.core  # noqa: F401
    except ImportError:
        return False
    return True


def preferred_sampler() -> str:
    choice = os.environ.get("F1P_SAMPLER", "auto").lower()
    if choice in ("numpy", "mlx"):
        return choice if choice == "numpy" or mlx_available() else "numpy"
    return "mlx" if detect().apple_silicon and mlx_available() else "numpy"


def staged_orders_mlx(
    log_strength: FloatArray,
    n_samples: int,
    seed: int,
    *,
    p_finish: FloatArray | None = None,
    stage_scales: FloatArray | None = None,
) -> tuple[NDArray[np.intp], NDArray[np.bool_]]:
    """MLX twin of :func:`f1p.eval.forecast.sample_orders`."""
    import mlx.core as mx

    n = int(log_strength.size)
    ls = mx.array(np.asarray(log_strength, dtype=np.float32))
    key = mx.random.key(seed % (2**32))
    k_fin, k_rest, k_stage = mx.random.split(key, 3)
    if p_finish is None:
        finished = mx.ones((n_samples, n), dtype=mx.bool_)
    else:
        pf = mx.array(np.asarray(p_finish, dtype=np.float32))
        finished = mx.random.uniform(shape=(n_samples, n), key=k_fin) < pf[None, :]
    gumbel = mx.random.gumbel(shape=(n_samples, n), key=k_rest)
    order_key = ls[None, :] + gumbel - mx.where(finished, 0.0, DNF_OFFSET)
    if stage_scales is not None and stage_scales.size:
        avail = finished
        cols = mx.arange(n)[None, :]
        centred = ls - mx.max(ls)
        keys = mx.random.split(k_stage, int(min(stage_scales.size, n)))
        for k in range(int(min(stage_scales.size, n))):
            inv_w = mx.exp(-float(stage_scales[k]) * centred)
            u = mx.random.uniform(low=1e-7, high=1.0, shape=(n_samples, n), key=keys[k])
            arrival = mx.where(avail, -mx.log(u) * inv_w[None, :], mx.inf)
            pick = mx.argmin(arrival, axis=1)
            chosen = (cols == pick[:, None]) & avail
            avail = avail & ~chosen
            order_key = mx.where(chosen, PLACED - k, order_key)
    orders = mx.argsort(-order_key, axis=1)
    mx.eval(orders, finished)
    return np.array(orders).astype(np.intp), np.array(finished).astype(bool)


def describe() -> dict[str, Any]:
    return {"sampler": preferred_sampler(), "mlx": mlx_available()}
