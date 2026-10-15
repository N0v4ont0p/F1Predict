"""``f1p silicon bench`` — the Phase 0.5 benchmark suite (plan §11a.6).

Each benchmark times one workload the pipeline actually runs, on every back end available
on this machine, and reports a throughput. Two modes:

* **burst** (default, ``--quick`` for CI sizes): best of a few repeats;
* **sustained** (``--sustained [minutes]``): every workload loops for the given time and is
  reported per minute (p50 / p95 / worst, and the last-minute / first-minute ratio), with the
  thermal state sampled alongside — the numbers the M6 targets are locked from (§11a.7).

Results are written as JSON next to a device description so baselines can be compared
across machines and dependency upgrades.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
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
    seconds: float  # best (burst) or median (sustained) time per call
    throughput: float  # best (burst) or p50 per-minute (sustained)
    unit: str
    detail: str = ""
    sustained: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Workload:
    """One timed call: ``work`` units of ``unit`` per call of ``fn``."""

    name: str
    backend: str
    unit: str
    work: float
    detail: str
    fn: Callable[[], Any]


# --------------------------------------------------------------------------- workloads
def _field(n: int = 22, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ls = rng.normal(size=n) * 1.5
    return ls, np.full(n, 0.9), np.exp(-0.12 * np.arange(26))


def sampler_workloads(samples: int) -> list[Workload]:
    from f1p.eval.forecast import sample_orders

    ls, pf, sc = _field()
    rng = np.random.default_rng(1)
    detail = f"{samples} x 22"
    out = [
        Workload(
            "race_sampler",
            "numpy",
            "races/s",
            samples,
            detail,
            lambda: sample_orders(ls, samples, rng, p_finish=pf, stage_scales=sc),
        )
    ]
    if mlx_available():
        out.append(
            Workload(
                "race_sampler",
                "mlx",
                "races/s",
                samples,
                detail,
                lambda: staged_orders_mlx(ls, samples, 7, p_finish=pf, stage_scales=sc),
            )
        )
    return out


def gemm_workloads(n: int) -> list[Workload]:
    a = np.random.default_rng(0).standard_normal((n, n)).astype(np.float32)
    b = np.random.default_rng(1).standard_normal((n, n)).astype(np.float32)
    flop = 2 * n**3 / 1e9
    out = [Workload("gemm_fp32", "numpy", "GFLOP/s", flop, f"{n}x{n}", lambda: a @ b)]
    if mlx_available():
        import mlx.core as mx

        am, bm = mx.array(a), mx.array(b)
        out.append(
            Workload("gemm_fp32", "mlx", "GFLOP/s", flop, f"{n}x{n}", lambda: mx.eval(am @ bm))
        )
        ah, bh = am.astype(mx.bfloat16), bm.astype(mx.bfloat16)
        out.append(
            Workload("gemm_bf16", "mlx", "GFLOP/s", flop, f"{n}x{n}", lambda: mx.eval(ah @ bh))
        )
    if importlib.util.find_spec("jax") is not None:
        jax: Any = importlib.import_module("jax")
        jnp: Any = importlib.import_module("jax.numpy")

        aj, bj = jnp.asarray(a), jnp.asarray(b)
        mm = jax.jit(lambda x, y: x @ y)
        out.append(
            Workload(
                "gemm_fp32",
                "jax-cpu",
                "GFLOP/s",
                flop,
                f"{n}x{n}",
                lambda: mm(aj, bj).block_until_ready(),
            )
        )
    return out


def cholesky_workloads(n: int) -> list[Workload]:
    rng = np.random.default_rng(0)
    m = rng.standard_normal((n, n))
    spd = m @ m.T + n * np.eye(n)
    return [
        Workload(
            "cholesky_fp64",
            "numpy",
            "GFLOP/s",
            n**3 / 3 / 1e9,
            f"n={n}",
            lambda: np.linalg.cholesky(spd),
        )
    ]


def world_model_workloads() -> list[Workload]:
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
    samples = [
        (rng.normal(size=(22, 3)), rng.permutation(22)[:16].astype(np.intp)) for _ in range(160)
    ]
    basis = linear_pl.smooth_stage_basis()
    return [
        Workload(
            "rating_filter",
            "numpy",
            "races/s",
            len(races),
            "1000 races",
            lambda: run_filter(races, RatingParams()),
        ),
        Workload(
            "combiner_fit",
            "scipy",
            "fits/s",
            1,
            "160 races, 26 stages",
            lambda: linear_pl.fit_model(
                samples, prior=np.ones(3), n_stages=linear_pl.MAX_STAGES, basis=basis
            ),
        ),
    ]


def neural_engine_workloads() -> tuple[list[Workload], dict[str, Any]]:
    """A toy fixed-shape FP16 network compiled for the Neural Engine via Core ML.

    Only on macOS with ``coremltools`` installed; the returned info records where Core ML
    placed the operations (the compute plan), because "compiled and loaded" alone does not
    prove the model runs on the Neural Engine (plan §11a.3).
    """
    info: dict[str, Any] = {}
    if sys.platform != "darwin" or importlib.util.find_spec("coremltools") is None:
        info["neural_engine"] = "skipped (needs macOS + coremltools)"
        return [], info
    try:
        ct: Any = importlib.import_module("coremltools")
        mb: Any = importlib.import_module("coremltools.converters.mil").Builder

        n = 512
        w1 = np.random.default_rng(0).standard_normal((n, n)).astype(np.float16) * 0.05
        w2 = np.random.default_rng(1).standard_normal((n, n)).astype(np.float16) * 0.05

        @mb.program(
            input_specs=[mb.TensorSpec(shape=(64, n), dtype=ct.converters.mil.mil.types.fp16)]
        )
        def prog(x: Any) -> Any:
            h = mb.relu(x=mb.matmul(x=x, y=w1))
            return mb.matmul(x=h, y=w2)

        model = ct.convert(
            prog,
            convert_to="mlprogram",
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.macOS14,
        )
        name = next(iter(model.get_spec().description.input)).name
        x = {name: np.random.default_rng(2).standard_normal((64, n)).astype(np.float16)}
        flop = 2 * 2 * 64 * n * n / 1e9
        try:
            MLComputePlan: Any = importlib.import_module(  # noqa: N806
                "coremltools.models.compute_plan"
            ).MLComputePlan

            path = Path(model.package_path)  # type: ignore[attr-defined]
            plan = MLComputePlan.load_from_path(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
            devices: dict[str, int] = {}
            program = plan.model_structure.program
            if program is not None:
                for fn in program.functions.values():
                    for op in fn.block.operations:
                        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
                        dev = type(usage.preferred_compute_device).__name__ if usage else "none"
                        devices[dev] = devices.get(dev, 0) + 1
            info["neural_engine_placement"] = devices
        except Exception as exc:  # pragma: no cover - API differs across coremltools versions
            info["neural_engine_placement"] = f"unavailable: {exc}"
        return [
            Workload(
                "mlp_fp16", "coreml-ane", "GFLOP/s", flop, "64x512x512 x2", lambda: model.predict(x)
            )
        ], info
    except Exception as exc:  # pragma: no cover - only on macOS
        info["neural_engine"] = f"failed: {exc}"
        return [], info


def workloads(*, quick: bool) -> tuple[list[Workload], dict[str, Any]]:
    ws: list[Workload] = []
    ws += sampler_workloads(10_000 if quick else 200_000)
    ws += gemm_workloads(512 if quick else 4096)
    ws += cholesky_workloads(300 if quick else 2000)
    ws += world_model_workloads()
    ane, info = neural_engine_workloads() if not quick else ([], {})
    return ws + ane, info


# --------------------------------------------------------------------------- timing
def time_burst(w: Workload, repeats: int) -> BenchResult:
    w.fn()  # warm-up (JIT, caches, allocations)
    best = min(_timed(w.fn) for _ in range(repeats))
    return BenchResult(w.name, w.backend, best, w.work / best, w.unit, w.detail)


def _timed(fn: Callable[[], Any]) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def thermal_state() -> str:
    """macOS thermal pressure (``pmset -g therm``); ``n/a`` elsewhere."""
    if sys.platform != "darwin":
        return "n/a"
    try:
        out = subprocess.run(
            ["pmset", "-g", "therm"], capture_output=True, text=True, timeout=5, check=False
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    for line in out.splitlines():
        if "CPU_Speed_Limit" in line or "Thermal" in line:
            return line.strip()
    return out.strip().splitlines()[-1] if out.strip() else "unknown"


def time_sustained(
    w: Workload,
    seconds: float,
    *,
    window: float = 60.0,
    clock: Callable[[], float] = time.perf_counter,
) -> BenchResult:
    """Loop ``w`` for ``seconds``; report throughput per ``window`` (p50 / p95 / worst) and the
    last-window / first-window ratio, which exposes throttling."""
    w.fn()
    start = clock()
    windows: list[float] = []
    thermal: list[str] = []
    calls_total = 0
    win_start, win_work = start, 0.0
    while clock() - start < seconds:
        w.fn()
        calls_total += 1
        win_work += w.work
        now = clock()
        if now - win_start >= window:
            windows.append(win_work / (now - win_start))
            thermal.append(thermal_state())
            win_start, win_work = now, 0.0
    if win_work and clock() - win_start > 0:
        windows.append(win_work / (clock() - win_start))
    arr = np.asarray(windows) if windows else np.asarray([0.0])
    elapsed = clock() - start
    stats = {
        "minutes": round(elapsed / 60.0, 2),
        "windows": float(arr.size),
        "p50": float(np.median(arr)),
        "p95": float(np.quantile(arr, 0.95)),
        "worst": float(arr.min()),
        "last_over_first": float(arr[-1] / arr[0]) if arr[0] > 0 else 0.0,
    }
    if thermal:
        stats["thermal_final"] = thermal[-1]  # type: ignore[assignment]
    per_call = elapsed / max(calls_total, 1)
    return BenchResult(w.name, w.backend, per_call, stats["p50"], w.unit, w.detail, stats)


def run(
    *, quick: bool = False, sustained_minutes: float = 0.0
) -> tuple[list[BenchResult], dict[str, Any]]:
    ws, info = workloads(quick=quick)
    if sustained_minutes > 0:
        results = [time_sustained(w, sustained_minutes * 60.0) for w in ws]
    else:
        results = [time_burst(w, 2 if quick else 5) for w in ws]
    return results, info


def save(results: list[BenchResult], directory: Path, info: dict[str, Any] | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    mode = "sustained" if any(r.sustained for r in results) else "burst"
    path = directory / f"bench-{mode}-{stamp}.json"
    payload = {
        "at": stamp,
        "mode": mode,
        "device": detect().to_dict(),
        "info": info or {},
        "results": [asdict(r) for r in results],
    }
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def compare(
    current: list[dict[str, Any]], baseline: list[dict[str, Any]], *, tolerance: float = 0.15
) -> list[str]:
    """Regressions: workloads whose throughput fell more than ``tolerance`` below baseline."""
    base = {(r["name"], r["backend"], r["detail"]): r["throughput"] for r in baseline}
    out = []
    for r in current:
        ref = base.get((r["name"], r["backend"], r["detail"]))
        if ref and r["throughput"] < (1.0 - tolerance) * ref:
            out.append(
                f"{r['name']} [{r['backend']}, {r['detail']}]: "
                f"{r['throughput']:.1f} vs baseline {ref:.1f} {r['unit']}"
            )
    return out
