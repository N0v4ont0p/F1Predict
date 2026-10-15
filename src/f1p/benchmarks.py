"""``f1p silicon bench`` — the Phase 0.5 benchmark suite (plan §11a.6).

Each benchmark times one workload the pipeline actually runs, on every back end available
on this machine, and reports a throughput. Results are written as JSON next to a device
description so baselines from the M6 can be compared across dependency upgrades.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from f1p.silicon.device import detect
from f1p.silicon.kernels import mlx_available, staged_orders_mlx


@dataclass(frozen=True, slots=True)
class BenchResult:
    name: str
    backend: str
    seconds: float  # best of the repeats
    throughput: float
    unit: str
    detail: str = ""


def _best(fn: Callable[[], Any], repeats: int) -> float:
    fn()  # warm-up (JIT, caches, allocations)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def _field(n: int = 22, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ls = rng.normal(size=n) * 1.5
    return ls, np.full(n, 0.9), np.exp(-0.12 * np.arange(26))


def bench_sampler(samples: int, repeats: int) -> list[BenchResult]:
    from f1p.eval.forecast import sample_orders

    ls, pf, sc = _field()
    out = []
    rng = np.random.default_rng(1)
    t = _best(lambda: sample_orders(ls, samples, rng, p_finish=pf, stage_scales=sc), repeats)
    out.append(BenchResult("race_sampler", "numpy", t, samples / t, "races/s", f"{samples} x 22"))
    if mlx_available():
        t = _best(lambda: staged_orders_mlx(ls, samples, 7, p_finish=pf, stage_scales=sc), repeats)
        out.append(BenchResult("race_sampler", "mlx", t, samples / t, "races/s", f"{samples} x 22"))
    return out


def bench_gemm(n: int, repeats: int) -> list[BenchResult]:
    a = np.random.default_rng(0).standard_normal((n, n)).astype(np.float32)
    b = np.random.default_rng(1).standard_normal((n, n)).astype(np.float32)
    t = _best(lambda: a @ b, repeats)
    out = [BenchResult("gemm_fp32", "numpy", t, 2 * n**3 / t / 1e9, "GFLOP/s", f"{n}x{n}")]
    if mlx_available():
        import mlx.core as mx

        am, bm = mx.array(a), mx.array(b)

        def run() -> None:
            mx.eval(am @ bm)

        t = _best(run, repeats)
        out.append(BenchResult("gemm_fp32", "mlx", t, 2 * n**3 / t / 1e9, "GFLOP/s", f"{n}x{n}"))
    return out


def bench_cholesky(n: int, repeats: int) -> list[BenchResult]:
    rng = np.random.default_rng(0)
    m = rng.standard_normal((n, n))
    spd = m @ m.T + n * np.eye(n)
    t = _best(lambda: np.linalg.cholesky(spd), repeats)
    return [BenchResult("cholesky_fp64", "numpy", t, n**3 / 3 / t / 1e9, "GFLOP/s", f"n={n}")]


def bench_world_model(repeats: int) -> list[BenchResult]:
    """Rating filter pass and combiner fit on synthetic seasons the size of real history."""
    from f1p.models import linear_pl
    from f1p.models.ratings import RaceObs, RatingParams, run_filter

    rng = np.random.default_rng(3)
    drivers = [f"d{i}" for i in range(40)]
    races = []
    for r in range(1000):
        pick = rng.choice(40, size=22, replace=False)
        races.append(
            RaceObs(
                f"e{r}",
                1990 + r // 20,
                r % 20 + 1,
                tuple(drivers[i] for i in pick),
                tuple(f"t{i // 2}" for i in pick),
                rng.permutation(22)[: rng.integers(12, 23)].astype(np.intp),
            )
        )
    t = _best(lambda: run_filter(races, RatingParams()), repeats)
    out = [BenchResult("rating_filter", "numpy", t, len(races) / t, "races/s", "1000 races")]
    samples = [
        (rng.normal(size=(22, 3)), rng.permutation(22)[:16].astype(np.intp)) for _ in range(160)
    ]
    basis = linear_pl.smooth_stage_basis()
    t = _best(
        lambda: linear_pl.fit_model(
            samples, prior=np.ones(3), n_stages=linear_pl.MAX_STAGES, basis=basis
        ),
        repeats,
    )
    out.append(BenchResult("combiner_fit", "scipy", t, 1 / t, "fits/s", "160 races, 26 stages"))
    return out


def run(*, quick: bool = False) -> list[BenchResult]:
    repeats = 2 if quick else 5
    results: list[BenchResult] = []
    results += bench_sampler(10_000 if quick else 200_000, repeats)
    results += bench_gemm(512 if quick else 4096, repeats)
    results += bench_cholesky(300 if quick else 2000, repeats)
    results += bench_world_model(1 if quick else 3)
    return results


def save(results: list[BenchResult], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = directory / f"bench-{stamp}.json"
    payload = {
        "at": stamp,
        "device": detect().to_dict(),
        "results": [asdict(r) for r in results],
    }
    path.write_text(json.dumps(payload, indent=2))
    return path
