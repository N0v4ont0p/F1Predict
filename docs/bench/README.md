# Benchmarks on the M6

These runs replace the plan's estimated performance targets (§2.2, §11a.9) with sustained
numbers measured on the target machine.

## Run it (about 45 minutes, plugged in, nothing else heavy running)

```bash
git pull
just setup                                   # includes the `apple` extra (MLX on Metal)
uv pip install jax coremltools               # optional: JAX-CPU and Neural Engine workloads
uv run f1p silicon info                      # expect: clusters, GPU cores, race sampler = mlx
just bench                                   # burst run (≈1 min) — sanity check
just bench --sustained 5                     # every workload looped 5 min, reported per minute
```

Both runs write JSON to `artifacts/bench/`.

**Commit the sustained file as `docs/bench/m6/<date>.json`.** Then any later run can be
checked against it:

```bash
just bench --baseline docs/bench/m6/<date>.json   # exits 1 if anything is >15% slower
```

## Reading the table

| Column | Meaning |
|---|---|
| throughput | Median per-minute throughput (sustained) or best repeat (burst) |
| p95 / worst | Spread across minutes; a low *worst* means stalls (thermal, memory pressure, background jobs) |
| last/first | Last minute ÷ first minute; below about 0.9 means the machine throttled |
| `neural_engine_placement` | Which compute device Core ML put each operation on. It must show the Neural Engine, or the model is not running there (§11a.3) |

## Workloads

| Name | What it stands for |
|---|---|
| `race_sampler` (numpy, mlx) | Monte Carlo race orders for forecasts and season simulation |
| `gemm_fp32` / `gemm_bf16` (numpy = Accelerate/SME, mlx = GPU, jax-cpu) | Dense linear algebra and future neural training |
| `cholesky_fp64` | Gaussian-process and Kalman blocks |
| `rating_filter`, `combiner_fit` | The world model's hot loops |
| `mlp_fp16` (coreml-ane) | Neural Engine inference path for the amortised posterior (Phase 9) |
