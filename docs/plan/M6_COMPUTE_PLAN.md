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
