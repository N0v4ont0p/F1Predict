# M6 follow-up plan — three experiments to run now, and the Apple-hardware checklist

This follows `M6_COMPUTE_PLAN.md`, whose steps 1–10 are built and merged. Part A plans three
experiments that can run on any machine; the select split (2022–2023) decides each one. Part B
lists everything that needs the M6 (Apple GPU, memory, long runs), so nothing is missed. It
is a plan, not code.

**Rules that still hold:**

- Every choice is made on *select* (2022–23).
- The holdout (2024–26) is scored once, last, through the ledger.
- Every result goes into the build log of `M6_COMPUTE_PLAN.md`.

---

## A. To do now

### A1. Temperature recalibration of the stack's P(win)

**Why.** The isotonic map overfit two seasons of ~44 winners. It made winner log-loss worse by
+0.134 (significant) and is now off by default. A single smooth parameter cannot overfit that
way.

**Method.**

- Multiply the pool's log-strengths by a temperature τ, fitted per (season, checkpoint) by
  maximum likelihood of the winners in the pool's training races.
- Bound τ to [0.5, 2] and shrink it towards 1 (an L2 penalty on log τ).
- To avoid in-sample optimism, fit τ on *out-of-sample* pooled forecasts: for each training
  season, the pool fitted on the seasons before it (cross-fitting). If that leaves too few
  races, fall back to in-sample with the shrinkage.
- Variant to compare: scale only P(win) (as isotonic did), or the whole Plackett-Luce
  strength vector. Scaling the whole vector also changes podium and points.

**Work.**

- `stacking.py`: a `fit_temperature` and a `recalibrate="none" | "isotonic" | "temperature"`
  option.
- `f1p eval stack --recalibrate temperature`.
- Tests: τ = 1 on a calibrated synthetic pool; τ > 1 recovered on an under-confident one;
  bounds respected.

**Experiment.** Re-run the select stack (C4, world_v1 + TabICL + LightGBM, two training
seasons) with each option. Prepared signals are cached, so this takes minutes.

**Decision gate.** Adopt it as the default only if winner log-loss is not significantly worse
than the raw pool and ECE top-1 improves. Otherwise keep `none` and record the result.

**Effort:** ~half a day.

### A2. C5 at several points in the race

**Why.** C5 today is one point (one third of the distance). The curve of accuracy against
race progress shows where the sequence model's advantage is largest, and where the simple
baselines catch up.

**Method.**

- Add in-race checkpoints `C5a` (1/4), `C5` (1/3, unchanged), `C5b` (1/2) and `C5c` (2/3).
  Each checkpoint maps to its fraction in one table in `eval/context.py`.
- Everything else (`race_so_far`, the baselines, lapseq) already reads the lap `k` from the
  context and needs no change.
- Keep `C5` unchanged so existing results and the ledger stay comparable.

**Work.**

- The checkpoint table; `checkpoint_time` and `IN_RACE` from it.
- The baselines' and lapseq's checkpoint sets.
- Tests: `k` per checkpoint; the no-look-ahead test parametrised over all four.

**Experiment.**

- Run `lapseq`, `pit_cycle`, `running_order` and `grid` on select at all four checkpoints.
- The 2022/2023 lapseq networks are cached, so no retraining is needed: one network serves
  every checkpoint.
- Report a table and a winner-log-loss curve against race fraction, with paired CIs against
  `pit_cycle`.

**Decision gate.** None (it is descriptive). Record which checkpoints are worth publishing
live.

**Effort:** ~half a day, plus ~15 min of runs.

### A3. A state-space (Mamba-style) variant of lapseq

**Why.** The plan's §D.2 ablation. The attention over laps costs O(T²) per car. A selective
state-space layer is O(T), carries a running state (natural for live forecasting: one
update per lap, no re-reading the race) and may generalise better on 200k laps.

**Method.**

- Replace each block's *attention over laps* with a selective state-space layer: Mamba-style
  input-dependent gating, diagonal state, causal by construction. Keep the attention
  *across cars* (within a lap there are only ~20 cars, and their interaction is the point).
- Same tokens, heads, losses, training driver and weights format; a `temporal="attention" |
  "ssm"` setting in `LapSeqConfig`.
- PyTorch first (fast CPU here); an MLX twin with the same parameter names so it runs on the
  Apple GPU later, under the same parity tests.
- Implement the scan directly (a loop or parallel scan over ≤ 80 laps), with no external
  `mamba-ssm` package, which needs CUDA.

**Work.**

- The SSM layer in both back ends.
- The config switch.
- Tests: causality, padding invariance, MLX↔PyTorch parity on CPU, and a step-by-step
  update equal to the full-sequence pass (the live-forecast property).

**Experiment.**

- Train `small` networks for 2022 and 2023, matched on parameter count and on the same
  pretraining corpus and seed.
- Compare with the attention lapseq on select at C5 (and the A2 checkpoints), on the
  next-lap log score (gate 1), and on training time.
- Paired bootstrap per race.
- Cost here: ~25 min per season network, so about 1 h.

**Decision gate.** Adopt the SSM as the default only if it is at least as good (no metric
significantly worse) and cheaper, or significantly better on order/winner at C5. Otherwise
keep it as a documented ablation.

**Effort:** ~1–1.5 days, plus ~1 h of runs.

### Order and delivery

Do A1 first (small, independent), then A2 (it gives A3 its evaluation grid), then A3. Each
experiment is one PR with code, tests, the select-split results and its build-log entry.

---

## B. Needs the M6 (Apple silicon) — checklist

Run these in order. Each line gives what it is, why it needs the M6, the command, and what to
record. Any item that changes a model must be finished before **B12**.

| # | Item | Why the M6 | Command / action | Record |
|---|---|---|---|---|
| B1 | Sustained benchmark baseline (plan step 0) | Real thermals, GPU and memory numbers | `just bench --sustained` | Throughput and throttling: the timing basis for everything below |
| B2 | MLX simulator on the GPU: timing and parity at scale | Metal | `f1p silicon bench` (race-sim workloads); parity test at large S | 100k races per X s (plan target < 2 s); NumPy↔MLX parity |
| B3 | lapseq GPU parity on real hardware | Metal | `pytest tests/models/test_lapseq.py` (`test_gpu_matches_cpu` runs only there) | Passes. Also re-check PyTorch MPS: if MPS now agrees, it can be re-enabled |
| B4 | lap_v2 posteriors for every race, sequential warm starts | Hours of 4-chain MCMC per season | `f1p lap validate --split tune --sequential --store default`, then select; per new race `f1p lap fit <event>` | Store under `artifacts/posteriors/lap_v2/default`; R-hat / ESS / E-BFMI per race |
| B5 | lap_v2 gates (plan §A.7) on select | Needs B4 | `f1p eval run --models world_v1,world_v1_lapv2 --split select` | Gate 3: does Bayesian pace help ranking? |
| B6 | Full simulator calibration (SMM) and predictive checks | CPU-hours of simulation | `f1p sim calibrate --split tune --every 1 --sims 2000`, then `f1p sim ppc --split select` | `artifacts/sim_v2/knobs.json`; KS with Benjamini–Hochberg |
| B7 | Tuning campaigns | Every core, overnight | `just tune filter 300`, `just tune combiner 100`, `just tune quali 100`, `just tune race_sim 60` | Selections with one-SE choice and null p-values; adopt only selections that pass |
| B8 | Full-size lapseq | GPU training at 6.3M parameters, 256 sims per race | `just lapseq <season> full` for 2019–2026 (or let eval train lazily) | Training logs; gate 1 with `lap_v2`'s predictive as the reference (needs B4) |
| B9 | Pairwise TabICL | MPS, contexts of ~16k rows | `f1p eval run --models tabicl_pairwise --split select --workers 1` | Compare with pointwise |
| B10 | TabICL fine-tuning (plan §D.1: positive stack weight now justifies it) | MPS, overnight, early stopping on select | New code needed: LoRA/full fine-tune on tune seasons | Only kept if it beats zero-shot on select |
| B11 | Stack v2 with all members on select | Needs B4–B10 | `just stack --split select --checkpoints C1,C4,C5 --members world_v1,world_v1_lapv2,race_sim_v2,tabicl_pointwise,lapseq,lgbm_rank` (+ A1's recalibration) | Weights and the gate against world_v1; freeze the member list and settings |
| B12 | **Holdout, once** | Final members exist only after B4–B11 | `just stack --split holdout --checkpoints C1,C4` (and the C5 models via `f1p eval run --split holdout --checkpoints C5`) | Ledger entry; the plan §H verdict. No changes afterwards without `--rescore` (recorded) |
| B13 | Overnight automation | macOS `launchd` | Write the plist running `just overnight <next event>` after each race weekend, background QoS, thermal check from B1 | Installed and dry-run once |
| B14 | Optional: custom Metal kernel for the simulator (plan §B.3) | Metal | New code, parity-tested against NumPy | Only if B2 misses the < 2 s target |
| B15 | Optional: neural posterior estimation (sbi) for simulator calibration | GPU training of the estimator | New code | Only if B6's predictive checks fail on many circuits |
| B16 | Optional: telemetry features (plan step 11) | 20–50 GB download and processing | New code | Gate: held-out lap elpd improves in lap_v2 |
| B17 | Optional: A3's SSM lapseq at full size | GPU | After A3 decides | Only if A3 adopts the SSM |

**Order on the M6:**

1. B1–B3 (one evening: hardware truth).
2. B4 (nights), and B6/B7 in parallel on spare cores.
3. B5.
4. B8–B10.
5. B11.
6. Freeze.
7. B12.
8. B13.

The optional items B14–B17 come only if their trigger condition is met.
