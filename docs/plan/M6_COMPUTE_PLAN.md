# M6 compute plan — full-Bayesian laps, GPU simulator, large-scale tuning, learned models

Status: **plan only, nothing built.** Follows `NEXT_PLAN.md` (all nine items merged). Target machine: the
M6 Mac mini, 24 GB unified memory, performance + efficiency CPU cores, GPU, Neural Engine.

## 0. Why, and what "maximum accuracy" means here

Today every model trains in seconds to minutes. That fits the data: about 22 races a year, about 205k
laps since 2018. Accuracy is limited by **information per race**, not by FLOPs. So compute is only worth
spending where it does one of the following:

1. **Removes an approximation that costs accuracy.**
   - Laplace → full posterior: uncertainty is currently Gaussian and too narrow, and the coarse 3×3
     simulator calibration is one example of the same problem.
2. **Adds information the current models throw away.**
   - Tyre cliffs, traffic, undercuts and safety-car dynamics.
   - Telemetry.
   - The order of laps within a race.
3. **Searches settings we fixed by hand.**
   - Priors, decays and windows, all of which were tuned on one season or not at all.
4. **Pools models whose errors differ.**
   - The stack nearly passed at C4 (−0.026, CI −0.058…+0.004).

### Rules that do not change

Every item below goes through the same gate as before:
- the rolling-origin backtest;
- paired bootstrap against the current best;
- a result ships only if its 95% interval excludes 0;
- negative results are written up in `docs/experiments.md`.

Two new rules, because large searches and big models make overfitting the test window easy:

- **Three-way time split, fixed now:**
  - *tune*: 2014–2021;
  - *select*: 2022–2023;
  - *holdout*: 2024–2026.

  The holdout is scored **once per item**, at the end, and the result is recorded even if it is bad.
  Every search and every architecture choice uses tune/select only.
- **Compute is budgeted and logged.** Each run records:
  - wall time;
  - energy (`powermetrics`);
  - peak memory;
  - thermal state (from `f1p silicon bench`).

  Long jobs run overnight under `launchd` at the background QoS lane, so the machine stays usable.

### Prerequisite (week 0)

Run `just bench --sustained 30 --save` on the M6. Every throughput estimate below is a guess until that
baseline exists. In particular:
- sustained GPU fp32 throughput;
- MLX vs NumPy for the sampler;
- how the performance cores and efficiency cores split work.

---

## A. Full Bayesian lap model (MCMC over all laps since 2018)

### A.1 Model — `lap_v2`, extending today's `lap_model`

The response is log lap time per clean lap, with a Student-t likelihood. Changes against today's model:

| Term | Today | v2 |
|---|---|---|
| Car pace | Random walk per team over races (season step) | **Gaussian process over race index** per team, with a Matérn-3/2 kernel and a changepoint at regulation resets and at known upgrade events (a feature, learned). Hilbert-space GP approximation (HSGP, ~20 basis functions) keeps it linear-time. |
| Circuit-type interaction | — | team × circuit-cluster effect: clusters from layout features (corner speed mix, straight fraction, altitude, surface abrasiveness), partially pooled |
| Driver | Constant | Driver skill as a slow random walk across seasons, plus a driver × wet effect, plus a rookie offset |
| Tyres | Linear wear per compound × event | **Monotone spline in tyre age** per compound × event, with hierarchical shrinkage to compound-global curves. This captures warm-up, linear wear and the "cliff". Monotonicity via the I-spline basis with positive weights. |
| Fuel | Linear | Linear with a per-season prior (fuel loads and car weights changed) |
| Track evolution | — | Smooth in race lap (rubbering in), per event |
| Traffic | 1{gap < 1 s} | Spline in the gap to the car ahead, plus a DRS indicator (gap < 1 s at the detection point, available in live timing) |
| Laps near pit stops | Excluded | Modelled: in-lap and out-lap get their own offsets. This feeds pit loss directly instead of the heuristic. |
| Noise | One σ | σ per driver × event, from a log-normal hierarchy. Student-t ν is learned. |
| Wet sessions | Mostly excluded | Included, with a wetness covariate (intermediate/wet compound share, rain from Open-Meteo) |

Parameter count is about 3–6k per 24-race window, and about 40k for a 2018–now joint fit.

### A.2 Inference stack

The frontier option comes first and a robust fallback second. Choose by benchmark on the M6 (A.5).

1. **nutpie (Rust NUTS) with normalising-flow mass-matrix adaptation**, on a PyMC model.
   - Fastest NUTS on CPU today.
   - Its flow adaptation fixes the funnel geometry that hierarchical models create.
   - One chain per performance core; 4–8 chains.
2. **NumPyro NUTS on JAX-CPU**, with chains vectorised (`chain_method="vectorized"`).
   - The fallback if the PyMC graph compiles slowly.
   - Also used for SVI (B.3).
3. **BlackJAX** is the escape hatch, and is used only if 1–2 struggle. It offers:
   - MCLMC, the microcanonical Langevin sampler, often 2–5× more efficient per gradient on large
     hierarchical models;
   - Pathfinder for initialisation.

GPU MCMC on Metal is **not** planned. jax-metal is experimental, and NUTS on these model sizes is
gradient-latency bound, so the performance cores beat the GPU. Re-check once a year.

Standard practice applies throughout:
- non-centred parameterisations;
- Pathfinder or Laplace (today's solver) for initialisation;
- `float64` for the likelihood.

### A.3 Making MCMC causal and affordable

The backtest needs a posterior **as of each race**, about 190 refits since 2019. There are three tiers:

| Tier | What | Cost (estimate, to be re-measured) |
|---|---|---|
| Exact | Full NUTS refit on a trailing window before every race | ~2–5 min × 190 ≈ 6–16 h per backtest (one or two nights) |
| Sequential | **Sequential Monte Carlo (SMC) over races.** Particles from race k are reweighted by race k+1's laps, resampled, and moved by a few NUTS/MCLMC steps (tempered SMC, as in BlackJAX). | Roughly 5–10× cheaper than the exact tier; still exact in the limit |
| Live | The weekly forecast: one fresh NUTS run on the latest window | Minutes |

Plan:
1. Build *Exact* first; it is the ground truth.
2. Build *Sequential*.
3. Validate that Sequential matches Exact (posterior means within 0.1 sd; credible intervals overlap
   ≥ 95%).
4. Use Sequential for all tuning.

### A.4 Validation (before any forecasting gate)

- **Diagnostics on every fit:** R-hat < 1.01, bulk/tail ESS > 400, zero divergences, E-BFMI > 0.3.
  Failures are logged and block publishing.
- **Simulation-based calibration (SBC):** about 200 synthetic datasets drawn from the prior. Rank
  histograms must be uniform. This proves the sampler recovers the model's own parameters.
- **Posterior predictive checks:**
  - lap-time distributions per stint;
  - stint degradation curves;
  - the gap-to-leader distribution.
- **Held-out lap score:**
  - expected log predictive density (elpd) of the next race's laps, against Laplace v1;
  - PSIS-LOO within the window, for model comparison between v2 variants.
- **Interval calibration:** the coverage of 50/80/95% predictive intervals on next-race car pace.
  Today's Laplace intervals are expected to be too narrow; this is the main payoff.

### A.5 Benchmarks to run first (go/no-go for A)

Measured on the M6, for one 24-race window:
- nutpie vs NumPyro vs BlackJAX-MCLMC;
- ESS per second, with 4 and 8 chains;
- peak memory.

If no sampler reaches ≥ 400 ESS on car-pace parameters in under 10 minutes, cut the model before
cutting the window. Remove terms in this order: the track-evolution spline, then the per-driver σ.

### A.6 Outputs

- `artifacts/posteriors/lap_v2/<as_of_event>.parquet`: 1,000 thinned draws of every parameter the
  simulator or features need.
- `lap_pace` v2 feature: the posterior mean **and sd** for each car. The combiner gets the sd as a new
  input, so uncertain pace is down-weighted.
- Pit-loss and degradation posteriors for B.

### A.7 Gates

1. Held-out lap elpd beats Laplace v1 (paired over races, significant).
2. Car-pace 80% interval coverage within 75–85% (Laplace v1 measured for comparison).
3. In `world_v1`, replacing `lap_pace` v1 with v2 (mean + sd) improves C1 winner log-loss or order
   log-likelihood significantly on *select*, then confirmed on *holdout*.

---

## B. GPU race simulator (`race_sim_v2`) on MLX/Metal

### B.1 Why v1 failed — the diagnosis this plan must fix

| Weakness in v1 | Consequence | v2 fix |
|---|---|---|
| Pace is lap-model mean ± Laplace sd only | Under-dispersed; ignores qualifying pace at C4 | Each simulated race draws one **posterior sample** from A (pace, degradation, pit loss). At C4, pace is also conditioned on qualifying and long-run practice through a pace/one-lap correlation learned in A. |
| One global start model | Ignores clean/dirty grid side, run to turn 1, launch differences | Hierarchical start model (B.2) |
| One overtaking curve, 3×3 calibration | Monaco ≈ Monza | Per-circuit hierarchical overtaking with DRS and tyre-delta terms (B.2) |
| SC as constant per-lap hazard | No link to incidents; wrong timing | Incident process (B.2) |
| Strategy from stop-count frequencies | No undercut and no reaction to an SC | Strategy policy (B.2) |
| 4k sims, CPU, Python loop over positions | Monte Carlo noise; no room for calibration | 100k–1M sims on GPU; calibration becomes affordable |

### B.2 Sub-models

Each sub-model is fitted from the `laps`, `pit_stops` and `results` tables; the Bayesian ones reuse A's
stack.

- **Start (lap 1).**
  - Inputs: grid slot, clean/dirty side, the distance from the grid to turn 1 (from circuit geometry),
    and a team launch effect.
  - Output: positions gained.
  - Model: an ordinal model with a first-lap incident probability per circuit.
  - Fit: on 2018+ lap-1/lap-2 positions; Jolpica covers 1996+ for longer start history.
- **Overtaking.** The probability that the car behind passes, per lap where it is within 1 s:
  `logit p = a_circuit + b·Δpace + c·DRS + d·Δtyre_age + e·Δcompound`.
  - `a_circuit` is partially pooled by circuit cluster.
  - Fitted on observed position swaps between consecutive laps.
- **Defending and dirty air.** The time lost while following is the traffic spline from A, so the
  simulator and the lap model agree.
- **Incidents → SC/VSC/red flag.**
  - Per-lap incident hazard per circuit, higher on lap 1 and on restart laps.
  - Each incident becomes a DNF, a VSC, an SC or a red flag, with circuit-dependent probabilities.
  - Durations come from history.
  - This ties safety cars to retirements, the dependency v1 lacked. It also yields correlated outcomes:
    an SC re-bunches the field and reshuffles strategies.
- **Reliability.** A per-lap DNF hazard per car from the existing reliability filter, split into
  mechanical (pace-independent) and incident (position-dependent: more in the midfield pack).
- **Strategy policy.**
  - A stochastic policy: pit decision per lap = f(tyre age vs the degradation-posterior cliff, gap to
    the car behind vs pit loss, SC active, laps left, compounds still owed).
  - Fitted by **inverse modelling**: a logistic or gradient-boosted classifier on the actual pit
    decisions in the `laps` table (about 8k stops).
  - Includes undercut/overcut by construction, because the gap terms are in the features.
- **Weather.** A rain-onset hazard from Open-Meteo forecasts for C4 and climatology for C1. On rain,
  cars switch to intermediates, which carry their own pace offset from A.

### B.3 GPU implementation (MLX)

- **State tensors:** (S sims × n cars) in fp32, covering cumulative time, tyre age, compound, stint,
  alive/retired, laps done, and SC state. At S = 1M, n = 22 the state is about 0.5 GB; this is where
  the RAM is used.
- **Lap step:** fully vectorised.
  - Lap times are drawn from the posterior sample of each sim.
  - The road order is found with one `argsort` per lap.
  - Overtakes are resolved with an **odd–even transposition network** over adjacent pairs, n − 1
    synchronous passes with no Python loop over positions. This is the core change from v1, whose
    per-position Python loop was the bottleneck.
- **Custom Metal kernel** (`mx.fast.metal_kernel`) for the pair-resolution pass, if profiling shows
  the composed-op version is memory-bound.
  - One thread per sim; the n ≤ 24 cars live in registers or thread-group memory.
  - Expected 5–20× over composed ops; to be measured.
- **`mx.compile`** for the whole lap step; the outer loop over laps stays in Python, at about 70
  iterations.
- **RNG:** counter-based (Threefry/Philox via `mx.random` keys split per lap). Results are bit-
  reproducible from (seed, event, checkpoint).
- **Precision:** times in fp32 relative to the leader; re-centre every lap, so cumulative fp32 error
  stays under 1 ms.
- **Parity:** the NumPy v1 reference stays as the oracle. A test runs both with matched settings at
  S = 200k and requires KS p > 0.01 on finishing-position, points and SC-count distributions.
- **Throughput target, to be confirmed on the M6:**
  - ≥ 1M race-laps-per-car per second, i.e. 100k races of 57 laps × 20 cars in under 2 s;
  - CPU NumPy as the fallback on other machines.

### B.4 Calibration — simulation-based, per circuit

v1's 3×3 grid search is replaced by two stages.

1. **Simulation-based inference** for the parameters that are not directly fittable: overtaking-curve
   scaling, start variance, incident hazard per circuit.
   - **Neural posterior estimation** with a flow-matching density estimator (the `sbi` package, PyTorch
     on MPS).
   - Summary statistics per circuit and season:
     - positions-gained histogram on lap 1;
     - overtakes per racing lap;
     - SC/VSC counts and timing;
     - |finish − grid| distribution;
     - stop-count mix;
     - gap to the winner at the finish.
   - A cheap simulator is what makes this work. Training needs about 10⁵–10⁶ simulations per circuit
     cluster, which is hours on the GPU (overnight).
2. **Direct proper-score tuning** of the few global knobs, using Bayesian optimisation (C) on the
   ranked probability score of the finishing order over *tune*.

Checks:
- SBC on the neural-posterior estimator;
- posterior predictive KS tests of every summary statistic per circuit, each required at p > 0.05
  (with a Benjamini–Hochberg correction across circuits).

### B.5 Outputs

- `race_sim_v2` as a `Forecaster`:
  - 100k sims at C1/C4;
  - exact-ish P(win) from the sample counts, with the Monte Carlo error reported in the meta;
  - points distributions;
  - correlated pairs, such as P(both cars of a team finish).
- A **season simulator v2**: whole remaining seasons as chained race sims (10k seasons × remaining
  races), replacing today's PL-based championship odds if the gate passes.
- Theatre/What-if: the Lab can request a server-side or pre-computed v2 run; WASM stays later work.

### B.6 Gates

1. Every per-circuit summary statistic passes the PPC/KS checks.
2. Standalone, it is within the error band of `world_v1`. The v1 failure margin to close is +0.53
   winner log-loss at C4.
3. **The stack (`world_v1` + `race_sim_v2` [+ D models]) beats `world_v1` significantly** on *select*,
   confirmed on *holdout*. This is the real objective; standalone parity is not required to ship it as
   a stack member.
4. Season simulator v2 title log-loss beats v1 on 2010–2025 (start at 2018 if laps are required).

---

## C. Fine-grained tuning across all models

### C.1 Engine

- **Optuna**, with storage in a local SQLite or JournalFile so runs are resumable and parallel-safe.
- Samplers:
  - **TPE (multivariate)** for mixed spaces;
  - **CMA-ES** for continuous ones with more than 10 parameters;
  - **NSGA-III** when two metrics are optimised jointly (winner log-loss and order log-likelihood).
- Pruning: **Hyperband/ASHA, with the season as the budget unit.** A trial is evaluated on 2014–2016
  first and killed early if it is worse than the median. That is how a few thousand trials fit into
  overnight runs.

### C.2 Parallelism on the M6

- One worker process per performance core, pinned to user-initiated QoS. Efficiency cores handle data
  loading and Optuna storage.
- BLAS is pinned to one thread inside workers; the 60× oversubscription bug from the last plan makes
  this rule mandatory.
- **Shared, zero-copy inputs:**
  - Contexts and prepared signals for every event are precomputed once into Arrow/Parquet and
    memory-mapped by all workers.
  - Rating-filter states are cached per hyperparameter set (a content hash), so trials that share
    filter settings and differ only in combiner settings skip the filter.
  - Today a backtest rebuilds contexts per run, which is most of its 10–25 minutes. The target is
    under 60 s per full-window trial for the combiner-only spaces.

### C.3 Search spaces, in order of expected value

| Model | Parameters | Notes |
|---|---|---|
| Rating filter (`world_v1`) | process noise, season-reset multiplier, debut variance, team/driver split, β | Was fitted once on ≤ 2013; re-fit on *tune* |
| Combiner | L2, stage-temperature basis (2 → 3 params), training-window seasons, pace decay | Cheap; many trials |
| Lap model v2 | Prior scales, GP length-scale priors, window length | Expensive; few trials on the SMC path with a reduced draw count |
| Quali model | Same as combiner, plus practice-pace weighting by session | |
| Race sim v2 | Global knobs not covered by B.4 | Bayesian optimisation with a GP surrogate; each trial is a GPU backtest |
| Challengers | Tree depth, learning rate, leaves, label gains | Standard |
| Stack | Pool training window, shrinkage, per-checkpoint/era weights | Last, on *select* only |

### C.4 Overfitting control

- Only *tune* is used inside the search.
- The top 5 configurations are re-scored on *select*. The winner is chosen by *select* with a
  one-standard-error rule: the simplest configuration within 1 SE of the best.
- The *holdout* is scored once.
- Search effort is reported: the number of trials, and the expected-max inflation on *tune* measured
  with a shuffled-label null run.

### C.5 Gate

The tuned configuration beats the hand-set one on *holdout* (paired bootstrap). If it doesn't, the
hand-set one stays, and the search documents how flat the optimum is.

---

## D. Larger learned models (where the RAM and GPU are needed)

### D.1 Tabular foundation models — in-context learning on the shared feature table

- **Candidates:**
  - TabICLv2;
  - TabPFN-2.5;
  - Mitra or LimiX, if they are available and their licences fit.

  A **licence check comes first**: some TabPFN releases restrict commercial use. Record the decision in
  the docs.
- **Runtime:** PyTorch on MPS, fp16/bf16. Peak memory is 8–16 GB for contexts of 10–50k rows, which is
  what the 24 GB is for.
- **Problem framing.** TFMs predict per row, so the ranking has to be expressed per row:
  - (a) per-car regression of the normalised finishing score → PL strengths through the out-of-sample
    scale mapping already used for the GBMs;
  - (b) **pairwise rows**: "does car A beat car B", 231 pairs a race → a Bradley–Terry fit per event.

  Both are evaluated; (b) suits classification-native TFMs.
- **Context selection.** Contexts are limited, so the context is the most relevant past rows: same
  circuit cluster, recent seasons, similar weather. Use the same causal cut as everything else.
- **Fine-tuning:** fine-tuning these models on F1 history (LoRA or full) is the frontier step.
  - Try it only if zero-shot ranks among the members that the stack weights above zero.
  - Run on MPS overnight, with early stopping on *select*.
- **Gate:** positive stack weight and a significant stack improvement on *select*/*holdout*.

### D.2 Sequence model over laps — `lapseq`

- **Goal:** model the race as a sequence. It improves pace and degradation and learns strategy and
  traffic interactions jointly. It also enables **in-race (live) forecasting**, which nothing else can.
- **Data:**
  - per-lap tokens per car: lap time residual, position, gaps, compound, tyre age, pit flags, track
    status, weather;
  - cross-car attention within a lap.

  That is about 205k laps today; with 2014–2017 from Jolpica, about 400k.
- **Architecture:** a factorised spatio-temporal transformer.
  - Attention over laps per car (causal) alternates with attention across cars within a lap.
  - About 5–30M parameters, which is the right size for this data; anything bigger overfits.
  - A Mamba-2/state-space variant is an ablation for long races.
  - Heads:
    - next-lap time (Student-t);
    - P(pit next lap);
    - P(incident);
    - final classification (Plackett–Luce over the last hidden states).
- **Training:**
  - MLX on the GPU (native and unified-memory friendly), or PyTorch MPS as the fallback.
  - Mixed precision.
  - Hours per run.
  - Causal season splits as everywhere.
- **Pretraining on simulated races** — the frontier idea that makes the small dataset work:
  - Pretrain on 10⁷ synthetic races from `race_sim_v2` drawn across the posterior, as a
    prior-fitted network: the simulator is the prior.
  - Then fine-tune on real laps.
  - This is amortised Bayesian inference. The network learns to condition on partial races the way
    the simulator would, and corrects the simulator where reality differs.
  - Generation is GPU-bound, about 1–3 hours for 10⁷ races at the B.3 target.
- **Gates:**
  - (1) next-lap time log-score beats the lap model v2's predictive;
  - (2) at C1/C4, its final-order head enters the stack with positive weight and a significant gain;
  - (3) new capability: an **in-race checkpoint (C5, after lap k)** evaluated on held-out races, as a
    new target in the harness with its own baselines (current running order; running order plus
    pit-cycle adjustment).

### D.3 Telemetry (optional, last)

- Live timing `CarData.z` (speed, throttle, brake, gear at about 4 Hz) and `Position.z`: roughly
  20–50 GB raw for 2018+.
- Distil per event and car into corner-level features: minimum speed per corner, traction and top
  speed. Keep features only (about 100 MB); stream-process the raw data and delete it.
- These feed A (car × circuit-cluster effects via corner-speed profiles) and D.2 as covariates.
- **Gate:** the corner-profile features improve held-out lap elpd in A.

---

## E. Integration and operations

- **Stack v2.** Members:
  - `world_v1`;
  - `world_v1` with lap v2;
  - `race_sim_v2`;
  - the TFM;
  - `lapseq`;
  - the GBMs.

  It uses the log-linear pool with weights per checkpoint and era, fitted on the previous four seasons
  (unchanged). Final isotonic recalibration of P(win), checked by ECE.
- **Pipelines:**
  - `just overnight` runs, in order: (1) sequential lap-posterior update, (2) the simulator calibration
    refresh for the next circuit, (3) the forecast publish.
  - A `launchd` plist runs it after each race weekend at the background QoS lane, with thermal
    throttling detection from the bench telemetry.
  - Search runs are separate (`just tune <space>`), always resumable.
- **Artifact registry:** `artifacts/<model>/<version>/<as_of>/`.
  - Each entry records the lake snapshot ID, the code git SHA, the seed and the config hash.
  - The forecast ledger records which model versions produced each published forecast.
- **CI:**
  - The heavy jobs never run in CI.
  - CI keeps tiny-config smoke tests: one chain, 50 draws; S = 1k sims; 2 trials; a 1-epoch model.
  - CI keeps the NumPy ↔ MLX simulator parity test at a small S.
- **Dependencies, all optional extras so the base install stays light:**
  - `bayes` (pymc, nutpie, numpyro, jax, blackjax, arviz);
  - `gpu` (mlx);
  - `tune` (optuna);
  - `learned` (torch, sbi, tabicl or tabpfn).

---

## F. Order of work and rough effort

| # | Step | Depends on | Effort | Compute |
|---|---|---|---|---|
| 0 | M6 sustained benchmark baseline | — | S | 30 min |
| 1 | Context/signal cache + parallel backtest runner (C.2 foundation) | — | M | — |
| 2 | Lap model v2 + sampler bake-off + SBC (A.1–A.5) | 0 | L | nights |
| 3 | Sequential lap posterior (A.3) + `lap_pace` v2 gate | 2 | M | 1–2 nights |
| 4 | MLX simulator core + parity (B.3) | 0 | L | — |
| 5 | Sub-models (B.2) fitted from data | 2 | L | hours |
| 6 | Simulation-based calibration (B.4) + `race_sim_v2` gates | 4, 5 | L | nights |
| 7 | Tuning campaign (C) for filter, combiner, quali | 1 | M | nights |
| 8 | TFM member (D.1) | 1 | M | hours |
| 9 | `lapseq` + simulator pretraining + C5 in-race target (D.2) | 6 | XL | days |
| 10 | Stack v2 + holdout scoring + docs | all | M | night |
| 11 | Telemetry features (D.3) | 2 | L | 1 day download |

Steps 2/3, 4/5 and 7/8 can run in parallel: CPU-bound work (MCMC, tuning) at night, GPU-bound work
(simulation, training) by day.

## G. Risks and how each is caught

| Risk | Early signal | Response |
|---|---|---|
| MCMC too slow for causal refits | A.5 benchmark | SMC path; smaller window; drop spline terms |
| Bayesian pace no better than Laplace for *ranking* | A.7 gate 3 | Keep v2 for uncertainty only (sd feature, simulator draws) |
| Simulator still worse | B.6 gate 2 | Ship as stack member only if gate 3 passes, else theatre-only |
| Search overfits | *select* vs *tune* gap; null-run inflation | One-SE rule; keep hand-set config |
| TFM licence or MPS op gaps | First import / smoke test | Use the alternative TFM; CPU fallback |
| Sequence model overfits 200k laps | Validation loss vs simulator-only prior | Smaller model; more synthetic pretraining |
| Thermal throttling invalidates timing | Bench telemetry | Background QoS, schedule at night, report sustained not burst |
| Data source access changes (OpenF1, live timing) | Ingest failures | Raw caches are immutable; fall back to Jolpica |

## H. What success looks like

- **Accuracy:** a stack that beats today's `world_v1` on the holdout with an interval excluding 0, at
  C1 and C4.
- **Calibration:** predictive intervals that cover what they claim.
- **Capability:** a new in-race forecast (C5).

All of it is reproducible on one Mac overnight, from a clean lake snapshot.

---

## I. Build log

What has been built against this plan, what was measured, and where the build deviates from the
plan (with the reason).

### Step 1 — parallel, cached backtest runner (done)

- `f1p.eval.runner` splits a backtest into a **prepare** stage (signals) and a **score** stage
  (forecasts and metrics).
  - Both run on one lazily spawned worker pool, single-threaded per worker.
  - Scoring is split by (season, checkpoint), the key of every per-season fit. Results are
    bit-identical to the serial path; this was checked on the real lake.
  - Prepared signals are cached by table content plus the model's prepare signature.
- `f1p.eval.splits` fixes tune / select / holdout.
  - The holdout ledger (`artifacts/ledger/holdout.jsonl`) allows one scoring per item and
    logs rescoring and ad-hoc exposure.
- `f1p.silicon.compute` records wall/CPU time, peak memory and thermal state with every run.
- Measured on a 4-vCPU Linux container:

  | Run (select split) | Serial | 4 workers | Cached |
  |---|---|---|---|
  | `world_v1` | 27.6 s | 17.4 s | 8.5 s |
  | `world_v1` + `race_sim_v1` | 195 s | 71 s | — |

### Steps 2–3 — Bayesian lap model `lap_v2` (built; gates pending M6 runs)

Modules: `models/lap_data.py` (design), `models/lap_v2.py` (model, samplers, persistence, feature),
`models/lap_validate.py` (held-out score, coverage, SBC, sequential check), `cli/lap.py`.

**Model, as built** — every A.1 term, with these decisions:

- **Car pace** is the sum of four terms, then centred per race:
  1. a team-season level walking across seasons;
  2. a Matérn-3/2 HSGP over race order (m = 30, c = 2.5);
  3. team slopes on circuit layout features;
  4. an i.i.d. team × race effect.
- **Why the boundary factor is c = 2.5.** The forecast race sits at the window edge, where
  c = 1.6 under-states the GP variance by 8%. The first real fit also showed the GP collapsing to
  a 1.2-race lengthscale to act as a per-race effect. With the explicit team × race term the
  lengthscale is about 14 races.
- **Circuit features** instead of clusters: straight share and slow-corner share (from the
  circuit geometry), a street-circuit flag, and km per corner.
- **Compound offsets** are contrasts *within* a race, against its most-used compound. A
  single-compound race (all of 2018–2021, or a full-wet race) has no offset to confuse with its
  base.
- **Tyre wear** is a monotone I-spline per compound, scaled per race by a wear rate.
  - The early I-spline column is dropped: it saturates within 4 laps and would act as a second,
    positive-only compound offset.
  - The out-lap offset carries warm-up.
- **Track evolution** is curvature only (orthogonal to fuel burn). It is behind a switch
  (§A.5's first cut).
- **Geometry fixes** found by the first real fits (R-hat 1.9 → 1.04):
  - zero-sum (Helmert) car and driver innovations, so the per-race centring leaves no ridge;
  - per-race centring of the within-race predictor, an exact reparameterisation of the base;
  - zero-sum season fuel deviations (a learned variance over 1–3 seasons is a funnel).
- **Inference:** one PyMC model, sampled by nutpie (diag, low-rank, flow or MCLMC), NumPyro,
  BlackJAX or PyMC.
- **Diagnostics** follow §A.4, plus `key_rhat` over the quantities forecasts read.

**First real result** (2024-R03 held out, 24-race window, 300 draws × 4 chains, 4 vCPU):

| | |
|---|---|
| Held-out elpd, v2 − Laplace v1 | **+0.68 nats per lap** (se 0.024, 863 laps) |
| Convergence | R-hat 1.04, 0 divergences, E-BFMI 0.68 |
| Car-pace 50/80/95% coverage (18 cars, one race) | 7 / 11 / 16 of 18 |
| Sampling time | ~20 min (the M6 must be benchmarked: §A.5) |

**Deviation — sequential tier (§A.3).** Each race adds parameters (base, team × race, pit and
compound terms), so tempered SMC over a fixed parameter space does not apply as written. The
sequential tier is instead a **warm-started refit**:

- the previous race's posterior means, in the sampler's unconstrained space, are mapped onto the
  next window by coordinate label;
- tuning is then short.

§A.3's acceptance test is unchanged and is run by `f1p lap sequential-check`. It compares
warm-vs-exact against exact-vs-exact (two seeds), which is the Monte-Carlo floor.

First check: 2024-R04, 12-race window, 300 draws × 4 chains, warm tuning 200 vs cold tuning 600.
The forecast quantities are next-race pace, the car scales, fuel, noise and compounds.

| | p95 \|Δmean\| / sd | Mean 90%-interval overlap | Sampling |
|---|---|---|---|
| warm vs exact | 0.19 | 0.956 | 104 s |
| exact vs exact (MC floor) | 0.15 | 0.958 | 220–278 s |

The warm fit is close to the Monte-Carlo floor at about 2.4× less sampling. At this run length the
floor itself exceeds §A.3's 0.1 sd, so that criterion is checked at production length on the M6.

**Feature.** `world_v1_lapv2` reads `artifacts/posteriors/lap_v2/default`. It uses two columns:

- `lap_pace_v2`: the posterior mean;
- `lap_pace_v2_unc`: mean × sd, so the linear combiner can down-weight uncertain pace.

The posterior store's fingerprint is part of the prepare signature.

**Runs for the M6** (all resumable; holdout scoring is guarded by the ledger):

```bash
uv sync --extra bayes
f1p lap bench 2024-R03 --chains 4,8                      # §A.5 go/no-go
f1p lap sequential-check 2024-R04                        # §A.3 acceptance
f1p lap sbc 2024-R03 --sims 200                          # §A.4
f1p lap validate --split tune --sequential               # posteriors for history + gate 1-2
f1p lap validate --split select --sequential             # gates 1-2 on select
f1p eval run --models world_v1,world_v1_lapv2 --split select   # gate 3
```

### Step 4 — simulator core on MLX, with parity (done; GPU timing pending M6)

`models/sim_core.py` is v1's race, vectorised over simulations, written once against a small
array shim. A NumPy twin and an MLX twin run identical code.

- **Overtaking is resolved in road order.** The cars are gathered into running order once per lap,
  resolved front to back column by column (one attempt per car per lap, as in v1), and scattered
  back.
- **On MLX the whole lap step is one `mx.compile`d graph.** The column loop is unrolled into it.
- **Per-lap random numbers are inputs.** In production they are drawn on the device from
  counter-based keys per (seed, lap), so nothing crosses to the host and runs are reproducible.
- **Times are relative to the leader**, re-centred every lap. This is exact in float32, and every
  decision depends only on differences.
- **Retirement keys are exact in float32** (v1's `1e9 - laps` collapses in float32).

**Deviation: no odd–even transposition network.** Repeated adjacent compare-and-swap passes
would give a car several overtaking attempts per lap. That changes the race v1 is calibrated on,
and it doesn't reduce the sequential depth (n − 1 rounds either way). The Metal kernel
(one thread per simulation, cars in registers) remains the next speed lever. It has to be
written and verified on the M6; this Linux build cannot run Metal.

**Parity tests** (`tests/models/test_sim_core.py`):

- On shared random numbers, NumPy and MLX run the same races: ≥ 99% identical orders
  (99.95% measured at 100k), with identical finishers and safety-car counts.
- Against v1's `simulate`, the core matches in distribution:
  - per-car finishing positions, by KS test with Bonferroni correction;
  - mean positions, within 4 standard errors;
  - finisher share and safety-car counts.
- The MLX device-RNG path is reproducible from its seed and matches NumPy in distribution.

`f1p silicon bench` now times `race_sim_v1` and `race_sim_v2` (NumPy, MLX) in car-laps/s, which
measures the §B.3 throughput target on the M6.

### Step 5 — simulator sub-models, fitted from data (done)

`models/sim_fit.py` holds one fitting tool, a MAP hierarchical logistic regression (L2 slopes,
partially pooled circuit intercepts, exact L-BFGS), and five sub-models. All of them are fitted
causally: `race_sim_v2` refits them once a season, on earlier seasons only.

| Sub-model | Fitted on | Notable estimates (2018–2026) |
|---|---|---|
| Overtaking: logit p = a_circuit + b·Δpace + g·gap + d·Δage + e·Δsoft | 26k close pairs (within 1 s at the line, green-flag, no pit laps) | +1.18 per s/lap of pace; −5.6 per s of gap; Monaco −2.4, Hockenheim/Austin +1.0 |
| Strategy: per-lap pit decision | 174k car-laps of finishers | neutralised lap +2.3; calibrated in-sample across probability bands |
| Start: Plackett–Luce ranking of start utilities | lap-1 orders | slot −0.60, clean side +0.29, run to turn 1 +0.21 |
| Incidents (a crash plus the neutralisation it causes, or an unexplained neutralisation) | per-lap hazard | lap 1 +2.2; P(neutral \| crash) 0.50; totals reconcile to the 245 observed periods |
| Compound choice | 2022+ dry races | start S/M/H 20/66/14% |

`models/sim_v2.py` runs them lap by lap over many simulated races, with one `lap_v2` posterior draw
per race (Laplace fallback). Cars queue in the pit lane. Each lap phase is a function over a typed
state. Tests check limiting cases with known answers:

- with passing, incidents and optional stops off, the finishing order is exactly the lap-1 order;
- with no start noise, the lap-1 order is the grid.

### Step 6 — calibration and checks (built; full runs pending M6)

`models/sim_calibrate.py`:

- **Summary statistics,** computed identically from simulated and real races:
  - lap-1 |gain| and hold share;
  - overtakes and close pairs per racing lap;
  - neutralisations;
  - stops and the one-stop share;
  - |finish − grid|;
  - the finisher share.
- **Simulated method of moments** over the global knobs (start noise, min gap, overtaking shift,
  incident scale), with common random numbers. A recovery test confirms it finds known knobs.
- **Per-circuit KS posterior-predictive checks**, Benjamini–Hochberg adjusted. This is gate 1.

Commands: `f1p sim calibrate --split tune` writes `artifacts/sim_v2/knobs.json`, which the
registered `race_sim_v2` loads; `f1p sim ppc --split select`.

**Deviation: no neural posterior estimation yet.** The plan names NPE (`sbi`, PyTorch on MPS).
The calibrated knobs are five global scalars, because the circuit-specific effects are fitted
directly in the sub-models, so the moment calibration provides the point estimate. NPE, a posterior
over the knobs, is M6 follow-up behind the same summaries. It needs a ~1 GB PyTorch dependency that
CI would otherwise install and run on every push.

**Structural findings during the build:**

- The Plackett–Luce start fit alone shuffles lap 1 too much: mean |gain| 2.2 vs 1.48 observed.
  The `start_noise` knob exists for this (0.6 gives 1.41).
- Holding cars at 0.3 s inflated close pairs and passes. Observed close-following gaps have a
  median of 0.86 s (default min gap: 0.7 s).

**First backtest** (2023 at C4; Laplace pace fallback, since no `lap_v2` posteriors are stored yet;
default knobs; 20k simulated races):

| | Winner log-loss | Winner hit | Podium Brier | Spearman | Order log-lik |
|---|---|---|---|---|---|
| world_v1 | **0.667** | 0.818 | **0.065** | **0.809** | **−1.491** |
| race_sim_v2 | 1.319 | **0.864** | 0.079 | 0.790 | −1.787 |
| race_sim_v1 | 1.458 | 0.864 | 0.088 | 0.735 | −1.813 |

v2 beats v1 on every metric but is still overconfident: it hits the winner more often than
`world_v1` but with extreme probabilities. The two planned remedies:

- `lap_v2` posterior pace, which is wider and honest, in place of the Laplace sd;
- §B.4 stage 2, proper-score tuning of the pace scale on *tune* (step 7).

The simulator's real target is the stack gate (§B.6.3), not standalone parity.

### Step 7 — tuning machinery (done; campaigns are M6 nights)

`f1p.tuning` is a new layer between `cli` and `models`.

- **Search spaces** (`spaces.py`):

  | Space | Parameters | Sampler |
  |---|---|---|
  | combiner | L2, training seasons | multivariate TPE |
  | filter | all 11 `RatingParams` | CMA-ES |
  | race_sim | pace scale, overtaking shift, start noise | GP when PyTorch is present, else TPE |
  | quali | its combiner | multivariate TPE |

  Each space defines a complexity measure for the one-SE rule: standardised distance from the
  hand-set configuration.
- **Engine** (`search.py`):
  - Optuna journal storage (resumable, multi-process safe), workers spawned single-threaded;
  - trials scored season by season over *tune*, with Hyperband using seasons as the budget;
  - the hand-set configuration is trial 0 and is never pruned;
  - the prepared-signal cache makes combiner trials cheap.
- **Selection:** the top trials are rescored on *select*, and the simplest within one SE (paired over
  events) of the best wins.
- **Search optimism, a deviation:** the plan's "shuffled-label null run" is implemented as a
  sign-flip null on every trial's per-event differences to the baseline. Under the null those
  differences are exchangeable, and it costs no re-runs.
- **Commands:** `f1p tune run <space> --trials N` and `f1p tune select <space>`.

**Smoke run** (12 combiner trials, real lake):

- *Tune:* the best trial (3 training seasons) beat the hand-set configuration, 1.5332 vs 1.5368.
- *Select:* it was worse, 1.5754 vs 1.5708.
- The one-SE rule kept the hand-set configuration.
- The null puts the tune "win" at p = 0.043: 0.0036 against 0.0011 expected by chance from the best
  of 11 trials.

The combiner's L2 optimum is flat: 0.6–1.4 give the same loss to 5 decimals.

### Step 8 — tabular foundation model (done)

**Licence decision (checked first, §D.1).** TabICL's code and weights are BSD-3-Clause, so it is the
member. TabPFN-2.5 and later weights are licensed non-commercially (gated, no production use), so
they are not used. Mitra and LimiX were not needed.

- `models/tfm.py`: `TFMRanker` runs TabICL on the challengers' causal feature table (the same
  pipeline as the GBMs: a context per season from seasons S-8..S-1, and the out-of-sample
  score → Plackett-Luce mapping), in both framings:
  - `tabicl_pointwise`: regression of each car's normalised finishing score;
  - `tabicl_pairwise`: classification of "A beats B" on feature differences, turned into
    strengths per race by a Bradley-Terry fit (Newton steps; the most recent 40 races of
    pairs as context).
- Optional extra `learned` (the plan's name): `tabicl`, `torch`. On Linux, torch comes from the
  PyTorch CPU index (PyPI's Linux build pulls ~3 GB of CUDA); macOS takes PyPI's build, with
  the MPS back end used when available.

**2023, C4 (22 races, CPU):**

| Model | winner log-loss | winner hit | podium Brier | points Brier | Spearman | finisher PL loglik |
|---|---|---|---|---|---|---|
| world_v1 | 0.667 | 0.818 | 0.065 | 0.152 | 0.809 | -1.491 |
| tabicl_pointwise | 0.895 | 0.818 | 0.064 | 0.151 | 0.789 | -1.491 |
| lgbm_rank | 1.046 | 0.682 | 0.079 | 0.161 | 0.793 | -1.554 |
| grid | 1.938 | 0.636 | 0.087 | 0.181 | 0.708 | -1.758 |

Zero-shot, TabICL beats the tuned GBM on every metric and matches `world_v1` on the full-order
likelihood and the Briers. It is weaker on the winner, where `world_v1`'s stage-wise
temperatures help. It is a different model on the same information, which is what a stack
member should be. Step 10 decides its weight.

The pairwise framing did not finish on this CPU within the run's limit: 40 races of pairs make a
~16k-row context, and ranking every pair of a race is 462 predictions. It is an M6 (MPS) run.
Fine-tuning (§D.1) waits on the stack weight, as planned.

### Step 9 — in-race checkpoint C5 and the lap-sequence model `lapseq` (done; full size is M6)

**C5, the in-race target (§D.2 gate 3).**

- `eval/in_race.py`: C5 is the race after lap `k`, one third of the distance (most cars have not
  stopped yet). The tables are cut at the start, as at C4. The race's own laps `1..k` come
  through `race_so_far`, the one place where the in-race cut is made.
- C5 contexts exist only for races with lap timing, and only when asked for
  (`--checkpoints C5`).
- Two approximations, both small at one third of the distance:
  - a lapped car's own lap `k` ends up to a lap after the leader's;
  - "still running" comes from the classification's laps completed, a fact that is public by
    lap `k`.
- A test scrambles every lap after `k` of a race; none of its in-race signals change.
- **Baselines**, each a Plackett-Luce with β and P(classified | running) fitted on earlier races:
  - `running_order`;
  - `pit_cycle`: gap to the leader plus the pit loss times the expected stops still owed (from
    the circuit's stop distribution), over √(laps left);
  - `grid`, kept at C5 as a reference.
- The work exposed a harness bug, now fixed: a model with none of the requested checkpoints was
  silently given the defaults.

**Simulator traces.** `sim_v2.simulate(trace=True)` records every lap: the lap as run (traffic
and holding included), gap, position, tyres, stop, neutralisation and still running. It draws no
extra random numbers, so a traced race is the untraced race.

**`lapseq`** (`lapseq_data.py`, `lapseq.py`, `lapseq_torch.py`, `lapseq_model.py`):

- **Input.** Real laps and simulated traces share one scale-free form: lap time relative to the
  field's lap, log gaps, tyres, stops, flags, progress, grid and a pre-race pace covariate.
  Token 0 is the grid; token `t` is the end of lap `t+1` (live timing has no lap-1 times).
  A test checks that the first `k` tokens of a race equal the tokens of its first `k` laps.
- **Network.** A factorised spatio-temporal transformer: causal attention over a car's laps,
  then attention across the cars of a lap, then a feed-forward layer, pre-norm.
  - Heads: next-lap Student-t (learned ν), P(pit), P(retire), P(neutralisation starts),
    P(classified | running), and a Plackett-Luce strength scored against the classification
    at every token.
  - Tests cover causality (changing laps after `t` leaves outputs up to `t` unchanged), padding
    invariance, and the losses against scipy and the harness's PL likelihood.
  - Sizes: `small` 1.1M parameters; `full` 6.3M (the plan's 5–30M).
- **Training.** Pretraining on `race_sim_v2` simulations of the training races (each simulated
  with the sub-models and pace known before it), then fine-tuning on real races with early
  stopping on the most recent 15%.
  - One network per forecast season, cached under `artifacts/lapseq/<signature>/<season>/` and
    shared by worker processes through a file lock.
  - A forecast's input never reads the classification.
- **Back ends** (a deviation, forced by the hardware): MLX's Linux CPU build has no optimised
  matrix kernels (2.9 GFLOP/s against 53 for NumPy on this container), so the network also
  exists in PyTorch.
  - Same layers, same parameter names, one training driver, one learning-rate schedule.
  - Weights are stored back-end neutral, so a network trained on one back end runs on the other.
  - Parity tests: MLX and PyTorch agree on the CPU, and MLX on Metal agrees with MLX on the CPU.
  - PyTorch on MPS did **not** agree with PyTorch on the CPU on the macOS runner (heads up to
    1.0 apart), so the PyTorch twin runs on the CPU and the Apple GPU path is MLX
    (`backend="auto"`).
  - The attention is written out explicitly in both back ends rather than using the fused
    kernels.
- **Commands:** `f1p lapseq train <season> [--size small|full]`, `f1p lapseq nextlap <season>`,
  and `just lapseq <season>`.

**Results, select split (2022–2023, 44 races).** The `small` network on CPU (PyTorch): 32
simulated races per real race, 3 pretraining epochs, fine-tuning stopped early after epoch 34.
Each season's network trained in about 25 min.

- Pretraining alone took the real validation loss from 1.09 to 0.98, before any real race.
  Fine-tuning reached 0.49.

| C5 (after lap ⌈L/3⌉) | winner log-loss | winner hit | podium Brier | points Brier | Spearman | order / car |
|---|---|---|---|---|---|---|
| **lapseq** | **1.218** | **0.795** | **0.060** | **0.116** | **0.853** | **−1.532** |
| pit_cycle | 1.615 | 0.659 | 0.071 | 0.145 | 0.792 | −1.629 |
| running_order | 1.700 | 0.727 | 0.071 | 0.132 | 0.795 | −1.657 |
| grid | 2.030 | 0.545 | 0.087 | 0.183 | 0.710 | −1.755 |

Δ lapseq − pit_cycle, paired bootstrap 95% CI:

- winner log-loss −0.397 (−0.510…−0.285);
- order/car +0.097 (+0.037…+0.155);
- podium Brier −0.011 (−0.016…−0.006);
- points Brier −0.029 (−0.036…−0.021).

Against `running_order`, every interval also excludes 0. **Gate 3 passes:** the in-race
forecast is a new capability, and it beats both its baselines significantly.

**Gate 1 (next-lap log score), 2023, 21,221 laps:**

- lapseq 1.381 nats per lap against 1.073 for the in-race running mean of each car's clean laps
  with a Student-t fitted on the training races: Δ +0.308 ± 0.013.
- The event heads are only a little better than their base rates (log-loss): pit 0.156 vs
  0.159, retirement 0.0145 vs 0.0147, neutralisation 0.090 vs 0.100.
- The plan's reference is `lap_v2`'s posterior predictive, which needs the stored posteriors
  (`f1p lap validate --store default` on the M6). The running-mean baseline stands in for it
  here.

**At C4**, lapseq sees only the grid and a pre-race pace covariate, and is in between: winner
log-loss 1.850, between `grid` (2.005) and `world_v1` (1.004). Gate 2 (a positive stack weight
at C1/C4) is not expected from this input. lapseq has no C1 forecast at all, so its stack value
is in the race.

**M6 follow-ups:**

- the `full` network with `sims_per_race=256` (the plan's 10⁷ simulated races over the
  training window);
- the Mamba/state-space ablation;
- the `lap_v2` predictive as the gate-1 reference;
- C5 at other fractions of the distance.

### Step 10 — stack v2 (done; the holdout scoring is the M6's)

**What changed** (`models/stacking.py`, `f1p eval stack`):

- The members of a checkpoint are the listed models that forecast it (`lapseq` has no C1, the
  in-race models only C5), so one stack spans every checkpoint.
- An optional isotonic recalibration of the pooled P(win) (`--recalibrate`), reported as
  `stack_iso`. The winner ECE (`ece_top1`) is in every stack summary.
- `--split select|holdout`: on the holdout, a stack is scored once through the holdout ledger. A
  stack is identified by its members' signatures, the checkpoints, the training seasons and
  recalibration.
- The default members are stack v2's: `world_v1, race_sim_v2, tabicl_pointwise, lapseq, lgbm_rank`.

**Select split, C4 (2022–2023, 44 races).** Members `world_v1`, `tabicl_pointwise` and
`lgbm_rank`, with weights fitted on the two seasons before each. On this CPU, lapseq's C4 member
(grid and pace only) and race_sim_v2 were left out: each extra season of lapseq costs about
25 min.

| Model | winner log-loss | podium Brier | points Brier | order / car | ECE top-1 |
|---|---|---|---|---|---|
| world_v1 | 1.004 | 0.068 | 0.160 | −1.516 | 0.019 |
| **stack** | 1.016 (+0.012, n.s.) | 0.066 | 0.157 | **−1.489** (+0.027, CI +0.007…+0.045) | 0.021 |
| stack + isotonic P(win) | 1.138 (+0.134, CI +0.028…+0.243) | 0.066 | 0.157 | −1.489 | 0.017 |
| tabicl_pointwise | 1.159 (+0.155, sig. worse) | 0.067 | 0.159 | −1.497 (+0.019, CI −0.001…+0.038) | 0.023 |
| lgbm_rank | 1.342 | 0.078 | 0.165 | −1.574 | 0.006 |

Pool weights at C4:

| Season | world_v1 | tabicl_pointwise | lgbm_rank |
|---|---|---|---|
| 2022 | 0.40 | 0.69 | 0.00 |
| 2023 | 0.18 | 1.19 | 0.00 |

Readings:

- **First significant stack gain.** The pool beats `world_v1` on the full-order likelihood with
  an interval excluding 0. On the winner it is level (n.s.). The M5 stack (world_v1 + simulator
  + GBMs) was not significant.
- **TabICL carries the pool.** It gets most of the weight, zero-shot, and on its own it nearly
  matches `world_v1` on the full order. `lgbm_rank` gets none. By the plan's §D.1 rule,
  TabICL's positive weight is the condition for trying fine-tuning (an M6 run).
- **Isotonic recalibration is off by default** (a deviation from §E's "final isotonic
  recalibration"). It made winner log-loss significantly worse for a marginal ECE gain. Two
  seasons give ~44 winners, and an isotonic step function on them overfits: it is
  over-confident at its plateaus. The raw pool's winner ECE (0.021) is already at
  `world_v1`'s level. A smooth one-parameter recalibration (a temperature on the pool) is the
  candidate if one is wanted.

**The holdout is not scored here.** It is scored once, and the full members need the M6:
`lap_v2` posteriors for `world_v1_lapv2`, the `full` lapseq, pairwise TabICL and the calibrated
simulator. The run is:

```bash
just stack --split holdout --checkpoints C1,C4
```

It records itself in `artifacts/ledger/holdout.jsonl`, and a second scoring needs `--rescore`.
