# f1predict — Ground-Up Rebuild Plan (v2)

> **Status:** planning only. No code is written until this plan is approved.
> **Date:** 15 October 2026. The 2026 season is in progress; the repo holds data only up to 2026 Round 5 (Canada, 24 May).
> **Scope:** clear the repository and rebuild f1predict from scratch, with three goals:
> 1. the most **accurate, honestly calibrated** F1 forecasting model we can build, covering as many race factors as possible;
> 2. **efficiency** at every layer;
> 3. a **striking, fast, interactive web experience**.
>
> **v2 changelog:** the modelling system (§7) and the technology stack (§3, §5, §11, §12.2, §13a) are **redesigned around 2025–2026 frontier methods**. The new elements are:
> - a single generative "world model" with three inference tiers: MCMC (NUTS and the newer MAMS sampler), fast variational inference, and **amortized neural inference** (a network that answers in milliseconds);
> - a **differentiable** race simulator calibrated by gradient descent;
> - **tabular foundation models** (TabPFN-2.5, TabICLv2) and a **purpose-built F1 network pretrained on simulated races** as challenger models;
> - **hierarchical Bayesian stacking** to combine models;
> - a **Claude-powered document and analyst layer**;
> - DuckLake time-travel storage, Polars 2, JAX, Rust/WASM, the three.js WebGPU renderer, and DuckDB in the browser.
>
> Frontier methods are **challengers, not assumptions**: each one ships only if it beats the current champion model on the rolling-origin evaluation (§10).

---

## Table of contents

0. [Executive summary](#0-executive-summary)
1. [Audit of the current repo](#1-audit-of-the-current-repo)
2. [Product vision and measurable success criteria](#2-product-vision--measurable-success-criteria)
3. [Decisions (defaults — override any)](#3-decisions-defaults--override-any)
4. [Repository reset procedure](#4-repository-reset-procedure)
5. [Target architecture](#5-target-architecture)
6. [Data platform — beating the small-data problem](#6-data-platform--beating-the-small-data-problem)
7. [Modelling system — frontier, layered, probabilistic](#7-modelling-system--frontier-layered-probabilistic)
8. [The factor catalogue — every aspect we model](#8-the-factor-catalogue)
9. [Small-data strategy — the 18 techniques](#9-small-data-strategy--the-18-techniques)
10. [Evaluation, honesty and MLOps](#10-evaluation-honesty--mlops)
11. [Performance engineering](#11-performance-engineering)
12. [The web experience (UI/UX)](#12-the-web-experience-uiux)
13. [Command line and terminal UI](#13-cli--tui)
13a. [Technology bill of materials](#13a-technology-bill-of-materials)
14. [API and serving](#14-api--serving)
15. [Phased roadmap with acceptance gates](#15-phased-roadmap-with-acceptance-gates)
16. [Risks and mitigations](#16-risks--mitigations)
17. [Legal, licensing and ethics](#17-legal-licensing--ethics)
18. [Appendices](#18-appendices)
19. [Sources consulted](#19-sources-consulted)

---

## 0. Executive summary

The current f1predict learns from **5,215 race-result rows** using tree-ensemble regression on finishing position. Its probabilities come from noise added using the spread of its own training residuals, and one data bug labels 6.5% of classified finishers as retirements. It has a hard ceiling.

The rebuild replaces it with a **generative world model of a race weekend**. This is a single probabilistic program describing how cars, drivers, tyres, fuel, weather, safety cars and failures produce every lap of every session. The program is fitted to **≈1M laps plus 1950-onward history**, and then *used three ways* from the same code:

| Use | How | Speed |
|---|---|---|
| **Learn** (fit the model to the data) | MCMC (MAMS/NUTS) as the weekly gold standard; variational inference (SVI) with normalizing flows after each session | Minutes |
| **Infer instantly** | **Amortized neural posterior estimation**: a network trained on millions of runs of the world model maps session data to posteriors in milliseconds (BayesFlow 2) | Milliseconds |
| **Forecast and explain** | Lap-by-lap simulation from the posterior; a **differentiable** twin of the simulator calibrates its own parameters by gradient descent | 100k races in seconds |

On top of the world model, **challengers** compete to correct what it misses:
- tabular foundation models (**TabPFN-2.5**, **TabICLv2**: state of the art on tables of fewer than about 50k rows, which is exactly our size);
- gradient-boosted trees with a **Plackett–Luce ranking objective**;
- **TabM**;
- **F1-PFN**, a permutation-equivariant transformer pretrained on tens of millions of *simulated* races, which learns to forecast in-context.

The winners are combined with **hierarchical Bayesian stacking**: the blend weights depend on context (checkpoint, wet or dry, regulation era). The result is then temperature-calibrated and given **conformal** prediction ranges.

| | Today | Rebuild v2 |
|---|---|---|
| Unit of learning | ≈5k result rows | ≈1M laps + stints + sessions + 1950+ results |
| Core model | Random Forest / LightGBM regression | **Generative world model** (JAX/NumPyro) + challengers + hierarchical stacking |
| Inference | — | **3 tiers**: MCMC (BlackJAX MAMS/NUTS), SVI with flow guides, **amortized neural** |
| Simulator | Rank shuffle | Lap-by-lap, **differentiable twin** for calibration, **Rust/WASM** for production |
| Small data | Not addressed | 18 techniques (§9), including **simulation-pretrained transformers** and **tabular foundation models** |
| Probabilities | In-sample noise spread | Simulation distributions + stacking + temperature + conformal; ECE gate ≤ 0.02 |
| Storage | One Parquet file in git | **DuckLake** (snapshots + time travel) over Parquet, **Polars 2** streaming, DuckDB |
| UI | Streamlit | Next.js 16 + React Compiler + **three.js WebGPU** + **WASM simulator** + **DuckDB-WASM** |
| Live | — | OpenF1 MQTT → amortized posterior updates **every lap in milliseconds** |
| Documents | — | **Claude** turns FIA/Pirelli PDFs into structured, timestamped features |

**Timing:** Phases 0–7 aim to be ready for the closing 2026 races (US GP 25 October → Abu Dhabi) as a **live forward test**. The full product ships before the 2027 opener; 2027 continues the 2026 rules.

---

## 1. Audit of the current repo

### 1.1 Ideas worth keeping (to be reimplemented, not copied)

| Idea | Old location | Fate |
|---|---|---|
| Forgiving race shorthand (`suzuka26`, `vegas26`, `spain26 → Madrid`) | `data_pipeline/racref.py` | **Keep**: data-driven from the circuits table; reused in the web command palette (⌘K) |
| Interactive shell (history, completion, warm caches) | `cli/shell.py` | **Keep**: rebuilt |
| Separate predictions for race, qualifying and sprint | `features/factory.py` | **Keep and extend**: every session, every checkpoint |
| Regulation-era awareness | `features/regulations.py` | **Keep the idea**: reset dynamics are **learned**, not hand-set |
| Monte Carlo, what-if, championship swing | `simulation/`, `whatif/` | **Keep the idea**: new lap-by-lap engine |
| Track climate and physics table | `features/tracks.py` | **Demote to priors**: measured from telemetry instead |
| Model presets | `models/train.py` | **Replace** with fit profiles (`quick`/`standard`/`full`) |
| Model cards, data dictionary, glossary | `reporting/docs.py` | **Keep**: auto-generated, shown in the Model Lab |
| Property-based tests | `tests/` | **Keep the approach**, expand it a lot |

### 1.2 Defects found (each one must not recur)

| # | Defect | Evidence | Rebuild safeguard |
|---|---|---|---|
| D1 | Status **"Lapped" treated as a DNF**: 338 rows (6.5%) | `jolpica.py:62` | Status taxonomy (Appendix B) + data test |
| D2 | **Probabilities use the in-sample residual spread**; one σ for everyone | `train.py::_fit_predictor` | Simulation + out-of-sample calibration only |
| D3 | **Wrong objective** (position error; DNFs set to `n+1`) | `factory.py::_target_position` | Plackett–Luce likelihood; DNFs as a censored hazard |
| D4 | **Future information leaks** (`f_season_progress` uses the season's last round; medians over all data) | `factory.py` | `as_of(t)` access layer + future-perturbation tests |
| D5 | **Crude expected points** | `predictor.py::_expected_points` | Exact points from the full distribution (Appendix D) |
| D6 | **Thin evaluation** (29 races, no proper scores, no baselines) | `train.py::_evaluate` | §10 |
| D7 | **Stale data** (nothing after 2026 Round 5) | `data/master.parquet` | Automated session-level refresh |
| D8 | **Simulation shuffles ranks, not pace** | `simulation/engine.py` | Lap-by-lap world-model simulator |
| D9 | **O(n²) Python ELO loop, duplicated** | `features/elo.py` | JAX / Rust, one source of truth |
| D10 | **Repo clutter** (`__pycache__`, `.DS_Store`, `egg-info`, models, data in git) | git tree | `.gitignore` + release assets |
| D11 | Packaging hack + `sys.path` edits | `pyproject.toml`, `dashboard/app.py` | Standard `src/` layout with uv |
| D12 | Hand-set track constants | `features/tracks.py` | Measured; hand-set values only as priors |

### 1.3 Baseline numbers to beat (legacy, `experiments/experiments.jsonl`)

| Model | Races in test | MAE | Spearman | Winner correct | Podium correct |
|---|---|---|---|---|---|
| Race, `deep` preset | 29 | 3.70 | 0.659 | 55.2% | 66.7% |
| Qualifying, `deep` | 29 | 3.37 | 0.700 | 27.6% | 63.2% |
| Sprint, `deep` | 9 | 3.96 | 0.599 | 55.6% | 70.4% |

These were measured under D1–D4, so they aren't directly comparable. Phase 2 re-scores the frozen legacy model in the **new** evaluation harness.

---

## 2. Product vision and measurable success criteria

**Vision:** the most rigorous publicly available F1 forecasting system. It shows what will happen, how sure it is and why, updates live through the weekend, and lets anyone explore counterfactuals instantly.

### 2.1 Accuracy gates (rolling-origin 2022–2026 + live forward test; 95% paired-bootstrap confidence intervals)

| Gate | Target |
|---|---|
| G1 | Race-winner log-loss at the post-qualifying checkpoint beats the **qualifying-order Plackett–Luce baseline** by **≥ 10%** |
| G2 | Beats the **legacy model** on every headline metric; confidence intervals exclude 0 |
| G3 | **ECE ≤ 0.02** for P(podium) and P(points); reliability-diagram slope within [0.9, 1.1] |
| G4 | Pre-weekend (C1) predictions beat the "last race's order" baseline on Ranked Probability Score (RPS) |
| G5 | Qualifying: pole log-loss and Q3-cut accuracy beat the "previous qualifying" and "championship order" baselines |
| G6 | Pace model: out-of-sample clean-lap RMSE ≤ **0.35%** of lap time (provisional) |
| G7 | Simulator posterior-predictive checks pass for ≥ 90% of circuits (DNF count, SC count, positions gained, winning margin) |
| G8 | **Amortized inference:** simulation-based calibration (SBC) uniformity passes; 90% posterior intervals have 88–92% coverage; KL to the MCMC posterior below a threshold on held-out real weekends |
| G9 | Each challenger joins the stack only if it improves rolling-origin log score with a CI excluding 0 |
| G10 | *(Optional)* gap to the betting market's log-loss reported honestly |

### 2.2 Performance targets (MacBook Air M5, 24 GB, CPU unless stated)

| Area | Target |
|---|---|
| Incremental refresh after a session | < 2 min end to end |
| Full historical rebuild | < 45 min (limited by download speed; cached afterwards) |
| World-model SVI refresh (warm start) | < 5 min |
| Weekly MCMC (MAMS, 4–8 chains in parallel via `vmap`) | < 45 min |
| **Amortized posterior** (one weekend) | **< 50 ms** CPU; < 10 ms on GPU |
| Simulator (JAX reference) | 100k races × 22 cars × ~60 laps in < 10 s |
| **Simulator (Rust/WASM in the browser)** | 10k races < 1 s (SIMD + threads) |
| Peak memory | < 6 GB per stage |
| Web | LCP < 1.5 s, CLS < 0.05, INP < 200 ms, 60 fps 3D (WebGPU), graceful 2D fallback |
| API (precomputed) | p95 < 100 ms |

---

## 3. Decisions (defaults — override any)

| # | Decision | Default | Why | Rejected alternatives |
|---|---|---|---|---|
| A1 | Numerical core | **JAX** (jit/vmap/scan) on CPU locally | Composable transforms (vmap over chains, simulations and checkpoints), compiled XLA, one framework for Bayesian inference, simulation and neural nets | PyTorch (weaker functional transforms for MCMC); **JAX on Apple Metal**: the official `jax-metal` plugin appears unmaintained (no release since 0.1.0; maintainers reportedly closed Metal issues in Dec 2025), and community MLX/MPS backends are beta. So: **CPU locally, CUDA in the cloud** |
| A2 | Probabilistic programming | **NumPyro** (model language) + **BlackJAX** (samplers: NUTS, **MAMS**) + NumPyro SVI (flow guides) | MAMS (NeurIPS 2025) reports up to ~7× faster than NUTS; BlackJAX gives sampler choice over the same log-density | Stan (slower iteration, no JAX), PyMC (fine, less sampler choice) |
| A3 | Amortized inference | **BayesFlow 2** (Keras 3, **JAX backend**) | Multi-backend, modern generative networks (flow matching, coupling flows), summary networks, misspecification diagnostics; the authors recommend the JAX backend as fastest | `sbi` (PyTorch; kept as a cross-check) |
| A4 | Neural nets (F1-PFN, TabM) | **Flax NNX** or **Equinox** on JAX; trained on **rented CUDA GPUs** (serverless, per-job) | Same stack; training is offline, roughly weekly or monthly | Local MPS/MLX (possible fallback via Keras's torch backend; not the main path) |
| A5 | Tabular foundation models | **TabICLv2** (open) + **TabPFN-2.5** (licence-gated) | State of the art on small tables (TabArena); our race-level table is ≈10–30k rows | AutoGluon full stack (heavy) |
| A6 | Boosted trees | **LightGBM** (custom Plackett–Luce objective) + **CatBoost** | Strong, fast, reliable | XGBoost (no edge here) |
| A7 | Python | **3.13** (pinned) | BayesFlow 2 supports 3.11–3.13; JAX supports 3.14 but the whole stack must agree. Move to 3.14 (and evaluate free-threading) once every dependency ships wheels | 3.14 at kickoff |
| A8 | Tooling | **uv** (env, lock, Python), **ruff** (lint and format), **basedpyright** in CI, **ty** as the fast local checker (beta; promote at 1.0) | Fastest toolchain; ty's stable release is targeted for 2026 | mypy (slow) |
| A9 | Dataframes | **Polars 2** (streaming engine by default — verify at kickoff) | Streaming and out-of-core, multithreaded, Arrow-native | pandas |
| A10 | Storage | **DuckLake 1.0** catalogue over Parquet (zstd), queried by **DuckDB** | **Snapshots and time travel** (`AT (TIMESTAMP => …)`) make every backtest reproducible against the exact data version; zero servers | Iceberg/Delta (heavier), bare Parquet (no versioning) |
| A11 | Simulator in production | **Rust** crate `f1sim` → **PyO3** (Python) + **wasm-bindgen** (browser, SIMD + threads) | One fast engine everywhere; enables in-browser what-ifs | JAX-only (what-ifs on the server) |
| A12 | LLM layer | **Claude API**, default model **Claude Opus 5.5** (`claude-opus-5-5`), with PDF input, structured outputs, Message Batches, prompt caching and strict tool use (§7.11) | Best quality for document understanding and grounded writing; batch + caching keep it cheap | Rule-based parsers only (brittle) |
| A13 | Web | **Next.js 16.x** (App Router, Turbopack), **React 19 + React Compiler**, TypeScript strict, Tailwind v4 | Best ceiling for interactive UI, static export and ISR | Python-only UIs |
| A14 | 3D | **three.js WebGPURenderer** (WebGL2 fallback) via react-three-fiber; shaders in **TSL** | WebGPU in all major browsers incl. Safari 26; TSL compiles to WGSL and GLSL | Plain WebGL |
| A15 | In-browser analytics | **DuckDB-WASM** over Parquet/Arrow marts | The Model Lab can slice backtests and laps client-side, with no API | Precomputed charts only |
| A16 | In-browser ML | **ONNX Runtime Web** (WebGPU execution provider) for the amortized posterior and F1-PFN (if the export path is clean); otherwise server inference | Live what-if posteriors without a round trip | — |
| A17 | Orchestration | In-house content-hashed **asset DAG** + `just` + GitHub Actions cron; GPU jobs dispatched to serverless GPU | One-person operations | Dagster/Prefect (overkill at this scale) |
| A18 | Experiment tracking | **MLflow** (local SQLite) + model cards in git + DuckLake snapshot ids | Reproducible end to end | Weights & Biases (SaaS) |
| A19 | Notebooks | **marimo** (reactive, plain `.py`) | Diff-friendly | Jupyter |
| A20 | Hosting | **Vercel** (web), **Cloudflare R2** (marts, snapshots, ONNX models), **Fly.io** (small API), serverless GPU for training | ≈ €0–30/month + GPU minutes | Single VPS |
| A21 | Live data | **OpenF1 Sponsor** (€9.90/month), MQTT/WebSocket | Live feed ~3 s after events | Skip live |
| A22 | Brand | Name **f1predict**; no official F1, FIA or team marks or fonts | Trademark safety (§17) | — |

---

## 4. Repository reset procedure

1. **Tag** the current `main` as `v0-legacy` and push the tag; write `docs/legacy/LEGACY_SNAPSHOT.md` (metrics and dependencies).
2. **Archive** `data/master.parquet`, `models_store/*.joblib` and `experiments.jsonl` as assets on a `v0-legacy` GitHub Release.
3. **Clean-slate commit**: remove everything except `LICENSE` (MIT, to be added) and this plan (moved to `docs/plan/REBUILD_PLAN.md`).
4. **Scaffold** (Phase 0): the tree in §5.2; `.gitignore` (data, artefacts, caches, `.DS_Store`, `__pycache__`, `*.egg-info`, `.venv`, `node_modules`, `.next`, `target/`); `.editorconfig`; pre-commit; CI.
5. **No history rewrite** (`git filter-repo` only if the clone size ever becomes a problem).
6. **Branch protection**: CI green to merge; squash merges.

---

## 5. Target architecture

### 5.1 System overview

```
 SOURCES ── F1DB · Jolpica · FastF1/live-timing archive · OpenF1 (+live MQTT) · TracingInsights ·
            Open-Meteo (ERA5 hourly, ensemble forecasts) · FIA PDFs · Pirelli · [opt] Betfair
   │  async, rate-limited connectors → immutable raw/
   ▼
 DATA PLATFORM ── Polars 2 (streaming) + DuckDB + DuckLake catalogue (snapshots, time travel)
   │  raw → staged → curated (entity model) · pandera quality gates · as_of(t) (knowability)
   │  Claude document layer: PDFs → structured, timestamped rows (batch + caching)
   ▼
 SIGNALS ── clean laps · stint/programme classification · telemetry geometry · weather windows
   ▼
 WORLD MODEL (one NumPyro program: latent car/driver/tyre/track states → laps, quali, race, hazards)
   ├─ Tier 1  MCMC: BlackJAX MAMS / NUTS (weekly gold standard, vmapped chains)
   ├─ Tier 2  SVI: flow / low-rank guides, warm-started after every session
   ├─ Tier 3  Amortized NPE: BayesFlow 2 (trained on ~10⁷ simulated weekends) → ms posteriors
   └─ Differentiable simulator twin (soft ranking) → gradient calibration of race dynamics
   ▼
 FORECAST ENGINE ── lap-by-lap simulator (JAX reference → Rust f1sim; PyO3 + WASM)
   ▼
 CHALLENGERS ── LightGBM-PL · CatBoost · TabICLv2 · TabPFN-2.5 · TabM · F1-PFN (sim-pretrained set transformer)
   ▼
 COMBINER ── hierarchical Bayesian stacking (context-dependent weights) → temperature → conformal
   ▼
 EVALUATION (rolling origin × 6 checkpoints, SBC, PPC, ledger)   PUBLISHING (marts on R2/CDN · FastAPI · Next.js + WASM + DuckDB-WASM + ONNX Runtime Web · CLI/TUI)
```

### 5.2 Repository layout

```
f1predict/
├── pyproject.toml · uv.lock · justfile · rust-toolchain.toml
├── src/f1p/
│   ├── core/          # config (pydantic-settings), ids, clock & as_of, points, regulations, logging, otel
│   ├── sources/       # f1db, jolpica, fastf1, openf1 (REST + MQTT), openmeteo, fia_docs, pirelli, betfair
│   ├── lake/          # DuckLake catalogue, raw→staged→curated transforms, entity resolution, quality gates
│   ├── docs_ai/       # Claude document extraction: schemas, prompts, batch runner, verification sampler
│   ├── signals/       # clean laps, stints, corrections, track geometry, weather alignment
│   ├── world/         # THE world model: priors, latent dynamics, observation models (NumPyro)
│   │   ├── infer/     # mcmc (BlackJAX MAMS/NUTS), svi (flow guides), amortized (BayesFlow), diagnostics
│   │   └── sim/       # JAX reference simulator + soft (differentiable) twin + Rust bindings
│   ├── challengers/   # lgbm_pl, catboost, tabicl, tabpfn, tabm, f1pfn (pretraining + fine-tuning)
│   ├── combine/       # hierarchical stacking, calibration, conformal
│   ├── season/        # championship simulator
│   ├── live/          # OpenF1 stream → amortized posterior updates → live simulations
│   ├── eval/          # harness, metrics, baselines, bootstrap, SBC, PPC, ledger
│   ├── pipeline/      # asset DAG, content hashing, schedules, GPU job dispatch
│   ├── publish/       # marts (JSON/Arrow/Parquet), JSON Schemas, ONNX exports
│   ├── api/           # FastAPI
│   └── cli/           # Typer + Rich + prompt_toolkit; Textual TUI
├── crates/f1sim/ · crates/f1sim-py/ · crates/f1sim-wasm/
├── web/               # Next.js app (pnpm): app/, components/, lib/, workers/ (WASM sim, DuckDB-WASM, ORT-Web), content/ (MDX)
├── research/          # marimo notebooks
├── docs/              # ADRs, plan, methodology, generated data dictionary
├── tests/             # mirrors src/f1p
├── data/  (gitignored)  # DuckLake catalogue + Parquet
├── artifacts/ (gitignored)  # posteriors, networks, ONNX, reports, ledger
└── .github/workflows/ # ci, weekend, release, web, gpu-train
```

Boundaries are enforced with **import-linter**: `core ← sources ← lake ← signals ← world ← challengers/combine ← eval/publish ← api/cli`.

---

## 6. Data platform — beating the small-data problem

### 6.1 Source inventory

| Source | Coverage | Granularity | Access and limits | Licence and terms | Role |
|---|---|---|---|---|---|
| **F1DB** | 1950 → present, released after each race | Results, qualifying, **free practice and warm-up**, sprint, starting grid, fastest laps, pit stops, driver of the day, standings; drivers, constructors, chassis, **engines**, tyre makers, **circuits with layouts and SVGs** | One SQLite/CSV download per release | **CC BY 4.0** | Historical backbone |
| **Jolpica-F1** | 1950+; **laps 1996+**; **pit stops 2011+** | Results, laps, pit stops, standings | ≈4 req/s, ≈500 req/hour (verify); 429 + Retry-After | Open | Lap and pit history 1996–2017 |
| **FastF1 3.8.x** | **2018+** incl. testing | Laps (sectors, compound, tyre life, stint, pit, track status, deleted), telemetry ~4 Hz, **x/y/z position**, **per-minute weather**, race control, circuit info, schedule | Local cache | Unofficial timing data; non-commercial | Core lap source |
| **OpenF1** | 2023+; **live** | 18 endpoints incl. intervals, stints, pit, race control, weather, team radio, starting grid | Free: 3 req/s, 30 req/min (history); Sponsor €9.90/month: live MQTT/WebSocket | OpenF1 terms | Live mode, cross-checks |
| **TracingInsights** | 1950+ CSV archive; 2018+ per-session JSON | Laps, telemetry | GitHub bulk download | CC0 vs Apache-2.0 per repo (**verify**) | Mirror / fallback |
| **Open-Meteo** | ERA5 **hourly 1940+** (0.25°), ERA5-Land 0.1°, forecast + **ensemble** | Temperature, humidity, precipitation, wind, cloud | Free for non-commercial use | Non-commercial | Session-window weather; probabilistic forecasts |
| **FIA documents** | ≈2019+ | **Car Presentation Submissions** (upgrades), **PU element usage**, stewards' decisions, entry lists | PDF; polite | Public | Upgrade, PU and penalty features (via Claude, §7.11) |
| **Pirelli** | Per event | C-number nomination (2026: **C1–C5**, C6 dropped), pressures, previews | Press releases | Public | Compound mapping, degradation priors |
| **Betfair** *(optional)* | Varies | Exchange prices | BASIC free / PRO paid | Betfair terms | Benchmark / optional prior |
| **F2/F3** *(later)* | — | Results | Curation | Check terms | Rookie priors |

**Volume:** ≈27k race results (1950+), ≈650k race laps (1996+), ≈350k practice, qualifying and sprint laps (2018+), 10⁸–10⁹ telemetry samples (streamed, aggregated), ≈15–20k pit stops. That is **5k rows → ≈1M laps**.

### 6.2 Storage layers (on DuckLake)

| Layer | Content | Rules |
|---|---|---|
| `raw` | Exact source payloads + fetch metadata (URL, time, ETag, hash) | Immutable, append-only |
| `staged` | Typed, unit-normalised (ms, kg, °C, UTC), source IDs kept | Idempotent per source |
| `curated` | Canonical entity model (Appendix A) with `observed_at` on every row | Pandera schemas enforced |
| `signals` | Clean laps, stints, corrected pace, geometry, weather windows, document features | Content-hashed |
| `marts` | What the web and API read: JSON (brotli) + Arrow IPC + Parquet (for DuckDB-WASM) | JSON Schema-versioned; generated TS types |

**Two kinds of time, both enforced:**
- **Data version time (DuckLake snapshot):** "which data had we ingested at time T?" Every backtest run records its snapshot id, so results are reproducible bit for bit.
- **Knowability time (`observed_at`):** "when did this fact become public?" `as_of(t)` filters on it. Post-race penalties, for example, are knowable only from the stewards' decision time.

### 6.3 Entity resolution

- **Drivers:** F1DB ↔ Jolpica ↔ FastF1 (abbreviation and number per season) ↔ OpenF1 numbers; reserve drivers in FP1; mid-season swaps.
- **Constructor lineage** (Appendix C), e.g. Sauber → … → **Audi (2026)**; **Cadillac (2026)**.
- **PU supplier per team and season** (2026: Audi, Red Bull Ford Powertrains, Honda with Aston Martin — verify all).
- **Circuit layout versions** with date ranges (e.g. Barcelona chicane removed 2023; **Madrid new for 2026**).
- **Tyre compounds:** the label + Pirelli nomination → C-number.

### 6.4 Data quality gates (fail, never warn silently)

1. Classification integrity + status taxonomy (fixes D1).
2. Points reconciliation to the official standings (incl. 2014 Abu Dhabi double points, 2021 Spa half points, fastest-lap point 2019–2024).
3. Lap consistency (lap totals ≈ race time; sectors ≈ lap).
4. Cross-source agreement with a stated source priority.
5. Freshness alerts after each session.
6. Leakage contract (`observed_at` + `as_of`).

### 6.5 Clean-lap pipeline

1. Exclude in-laps, out-laps, lap 1, SC/VSC/yellow/red laps, deleted laps, pit laps, invalid timing.
2. Classify practice programmes (qualifying sim / long run / installation) with a small **hidden-Markov model over stints** (fuel and tyre regime as latent states). This replaces fixed rules and gives *probabilistic* labels that the world model uses directly.
3. Traffic: within 1.5 s of the car ahead → `dirty_air` (modelled, not dropped).
4. Wet regime from compounds and session rainfall.
5. Normalise to % of the session reference lap.
6. Carry: C-number, tyre age (incl. laps from earlier sessions), stint, fuel index, session time, track and air temperature, wind along the main straight, speed trap, gap ahead.

### 6.6 Track geometry from telemetry

Per `(circuit, layout_version)`: length; corners; **speed histogram**; % full throttle; braking zones; longest straight; **elevation** (z); main-straight bearing; 2026 straight-mode zones; measured overtaking difficulty; pit-lane loss; SC/VSC/red-flag and lap-1 hazards; evolution rate; degradation scale. These are the **priors for the circuit loadings** in the world model.

Additionally, a **circuit embedding** is learned from the resampled speed trace: a 1D-CNN autoencoder over the distance-normalised speed and throttle profile, giving an 8–16 dimensional vector. It lets the model place new or changed layouts (e.g. Madrid 2026) next to similar circuits immediately.

### 6.7 Weather alignment

- **Past races:** ERA5 hourly over the session window, combined with FastF1 per-minute session weather (preferred where available).
- **Upcoming races:** the Open-Meteo **ensemble** gives *members*, each a full rain/temperature timeline, which the simulator samples directly. Beyond the forecast horizon, use ERA5 climatology for the calendar window.
- **Derived:** per-lap-window rain probability, crossover risk, heat index, wind along the main straight.

---

## 7. Modelling system — frontier, layered, probabilistic

### 7.0 Principles

1. **One world model, many uses.** The same probabilistic program is the likelihood (learning), the simulator (forecasting), the what-if engine and the live engine. There is no drift between them.
2. **Probabilistic end to end.** Every output is a distribution.
3. **Structure first, then learned corrections, then stacking.** The world model carries the causal structure; challengers correct systematic misses; stacking decides by out-of-sample evidence.
4. **Frontier methods are challengers.** A method is promoted only via the gates in §10.
5. **Temporal safety by construction:** `as_of(t)` + DuckLake snapshot pinning.
6. **Six weekend checkpoints:**

| Checkpoint | When | New information |
|---|---|---|
| C0 | Pre-season / long range | Ratings, testing, climatology |
| C1 | Thursday | Upgrades, PU usage, forecasts, compounds |
| C2 | After FP1 (or after SQ on sprint weekends) | First weekend pace |
| C3 | After final practice (or after the sprint) | Long runs, qualifying sims |
| C4 | After qualifying | Grid, confirmed penalties |
| C5 | Live (every lap) | Race state |

### 7.1 The world model (generative core)

A single **NumPyro** program with these blocks:

**(a) Long-horizon latent states**, learned from 1950+ results and the 1996+ / 2018+ lap data jointly:
- **Driver skill** `κ_d(t)`: a slowly varying **Gaussian-process / random-walk** trajectory, with an **age and experience curve** as the mean function and the mean as a learned spline. Sub-skills, partially pooled: one-lap, race pace, tyre management `ω_d`, wet skill, starts, overtaking, defending, consistency `σ_d`, incident propensity.
- **Car performance** `θ_k(t) ∈ ℝ^F` (F ≈ 4–6 latent factors): a random walk across events with innovation variance `σ²_base + σ²_upgrade·u_{k,e} + σ²_reset·1[reset]`. Here `u_{k,e}` is the **upgrade score from FIA documents**, and the **reset variance and the carry-over correlation ρ_reset are learned from every reset since 1950** (1961, 1966, 1983, 1989, 1995, 1998, 2009, 2014, 2017, 2022, 2026). Team lineage links the seasons; the PU supplier adds a shared component (critical in 2026).
- **Circuit loadings** `λ_e`: priors from telemetry geometry + the learned circuit embedding (§6.6). Car pace at event e = `λ_eᵀ θ_k(e)`, a low-rank **car × circuit factorisation**.

**(b) Lap observation model** (2018+; 1996+ with reduced covariates):

```
y_{dl} = μ_{e,s} + λ_eᵀθ_k(e) + κ_d(e) + κ^{1lap}_d·[quali run] + κ^{wet}_d·wet
       + β^{fuel}_e·m_{dl} + g_{c,e}(age; T_track)·ω_k·ω_d + τ_{e,s}(t) + δ^{air}·dirty_air + ε,
ε ~ StudentT(ν, 0, σ_{e,s,d})
```

- **Fuel:** in races, from lap index × burn rate; in practice, a **latent starting fuel per stint**, informed by the HMM programme labels.
- **Tyres:** `g_c(a) = r_c·a + q_c·softplus(a − a*_c)`, linear wear plus a learned cliff; `r_c` depends on the circuit's degradation level, track temperature and the C-number.
- **Track evolution:** `τ_{e,s}` as a Gaussian random walk over session time.

**(c) Results-era observation model** (1950–2017, pre-telemetry): **dynamic rank-ordered logit (Plackett–Luce)** on classified finishers, **censoring retirements**. This anchors long-history priors and reset dynamics (extends van Kesteren & Bergkamp 2023, which attributed ~88% of 2014–2021 result variance to the constructor).

**(d) Hazard models** (discrete-time survival per lap): mechanical (era × PU supplier × team × PU element age × heat), incident (driver × circuit × wet × field density), SC/VSC/red (circuit × lap-1 × rain × incident triggers), deleted laps and penalties.

**(e) Session generators:** the qualifying knockout (exact format per season; 2026: 22 cars, expected Q1 22→16, Q2 16→10 — verify), the sprint and the race (§7.4). These *are* part of the program, so the model can be fitted to qualifying and race outcomes, not only to laps.

### 7.2 Inference — three tiers on one model

| Tier | Engine | When | Notes |
|---|---|---|---|
| **1 — Gold standard** | **BlackJAX MAMS** (Metropolis-adjusted microcanonical HMC; NeurIPS 2025, up to ~7× NUTS) and NUTS as a cross-check; chains `vmap`ped across CPU cores; Pathfinder initialisation | Weekly, and for the evaluation reference | R-hat, ESS/sec, divergences, energy diagnostics (ArviZ) |
| **2 — Fast** | **NumPyro SVI** with **normalizing-flow guides** (block neural autoregressive / IAF) or structured low-rank multivariate normal; minibatch over laps; warm start from the last posterior | After every session (C1–C4) | Checked against Tier 1 (KL, posterior mean/SD agreement); falls back to Tier 1 if the agreement gate fails |
| **3 — Instant** | **Amortized neural posterior estimation** with **BayesFlow 2** (JAX backend): a **summary network** (set transformer over laps and stints, per driver and per team) + an **inference network** (flow matching / coupling flow), trained on ~10⁷ weekends simulated from the world model's prior predictive, with the long-horizon states fixed at their current posterior ("local amortization") | Live (C5), the instant what-if lab, rapid checkpoint previews | Validated by **SBC**, coverage, and agreement with Tier 2 on real weekends (gate G8). **Misspecification detection** (distance in summary space, MMD) flags weekends unlike anything simulated → automatic fallback to Tier 2 |

Why all three: Tier 1 is the most accurate but slowest; Tier 2 is accurate enough to refresh after each session; Tier 3 makes **live every-lap updates and interactive what-ifs** possible. Because all three target the same model, their results can be checked against each other.

### 7.3 Differentiable simulator twin (gradient-calibrated race dynamics)

Some race dynamics are hard to learn as a likelihood: overtaking thresholds, start-phase position changes, SC bunching, strategy reactions. We build a **soft twin** of the simulator in JAX:
- Discrete events (a pass or not, a pit stop or not, a ranking) are relaxed with **Gumbel-softmax** and **differentiable soft sorting / ranking** (OTT-JAX soft-sort; SoftJAX operators).
- The parameters are fitted by **gradient descent on the simulated-vs-observed discrepancy** of summary statistics: position-trace distributions, positions gained by lap 1, overtakes per circuit, finishing-order likelihood under a PL surrogate. Uses Optax, with gradients that are `vmap`ped across circuits.
- **Annealing:** relaxation temperature → 0; a test asserts that the soft twin converges to the exact simulator.
- This replaces dozens of hand-set constants with **learned, per-circuit, per-era** values (e.g. 2026 active-aero / Overtake-Mode passing). The exact simulator (§7.4) uses the fitted values.

### 7.4 Forecast engine — lap-by-lap race simulator

State per car: position, cumulative time, gap, tyre (C-number, age), fuel, stops, damage/DNF, penalty seconds. Global: track status, weather member, lap.

1. **Start and lap 1:** grid slot, grid side, driver start skill, launch, run to Turn 1, lap-1 incident hazard (all learned).
2. **Lap time:** joint posterior draw per simulation (parameter uncertainty) + lap noise + degradation + fuel + dirty air + wet regime.
3. **Strategy:** candidate compound sequences (two-compound rule when dry), expected-time optimisation, team aggressiveness, reactive rules (cheap SC stops, undercut cover, crossover to inters). An optional **learned strategy policy** comes later: a small network imitating real team pit decisions (2018+ stints), evaluated against the rule-based policy.
4. **Overtaking:** calibrated by the soft twin (§7.3), with 2026-specific parameters.
5. **Pit stops:** measured lane loss + per-team stationary-time distribution (heavy tail), unsafe-release risk.
6. **SC/VSC/red flags:** circuit hazards, incident triggers, bunching, restarts, free tyre changes under red flags.
7. **Reliability and incidents:** per-lap hazards.
8. **Weather:** one ensemble member per simulation.
9. **Penalties:** time-penalty hazards.
10. **Scoring:** exact per-season rules (Appendix D).

**Implementations:** the JAX reference (`lax.scan` over laps, `vmap` over simulations) is used for fitting and backtests. The **Rust `f1sim`** (rayon, SIMD) handles production and the API. **WASM** (SIMD + threads; requires COOP/COEP cross-origin isolation) runs the browser what-if lab. Statistical-equivalence tests tie them together. **WebGPU compute** is evaluated later as a fourth backend.

### 7.5 Qualifying and grid

- **Knockout simulation:** one-lap pace draws + session evolution + execution noise + deleted laps + red-flag truncation + rain arrival (ensemble member).
- **Penalties:** confirmed (stewards' documents, with `observed_at`) + **projected** from PU element usage vs the allowance (2026: 4 ICE/TC/EX, 3 MGU-K/ES/CE; 10 places for the first excess, 5 for each further one, >15 → back of the grid — verify).

### 7.6 Challengers (learned corrections)

All challengers are trained only on **rolling-origin out-of-fold data** and predict either a **per-driver strength offset** (fed back into the simulator: "simulator-first") or **pairwise / listwise probabilities** for stacking.

| Challenger | What it is | Why it might win | Key settings |
|---|---|---|---|
| **LightGBM-PL** | Gradient-boosted trees with a **custom Plackett–Luce objective** (analytic gradients and Hessians) + LambdaRank variant | Strong default on tables | Monotone constraints, recency weights, strong regularisation |
| **CatBoost** | Ordered boosting, YetiRank | Robust on small, categorical-heavy data | — |
| **TabICLv2** | **Open tabular foundation model** (Feb 2026), in-context learning, reported to beat RealTabPFN-2.5 on several benchmarks | No training; excellent on small tables | Race-level table (≈10–30k rows): features + world-model summaries → residual / pairwise target |
| **TabPFN-2.5** | Tabular foundation model (Nov 2025), ≤50k rows / 2k features; distillation to an MLP or trees for serving | Top TabArena results on small data | **Licence gate**: research/internal weights; no distillation for commercial use. Benchmark-only unless the licence fits (§17) |
| **TabM** | Parameter-efficient ensembled MLP (ICLR 2025); on par with GBDTs in TabArena when tuned and ensembled | Diversity for the stack | Optuna-tuned, ensembled |
| **F1-PFN** *(our flagship challenger)* | A **permutation-equivariant set transformer** over the field (22 drivers × features, plus context tokens for circuit and conditions) with a **Plackett–Luce output head**. **Pretrained on ~10⁷–10⁸ races simulated from the world model's prior predictive** (the prior-fitted-network idea: learn to forecast in context across all plausible worlds), then **fine-tuned on real 2018–2026 races** with recency weighting | Turns the simulator into effectively unlimited training data, then lets real data correct the simulator's blind spots; captures field-level interactions (teammates, traffic) that per-row models cannot | Flax NNX/Equinox; Muon or AdamW optimiser; cloud GPU pretraining; ONNX export for the browser |

**Leakage note:** foundation models and F1-PFN see only `as_of` features. Pretraining data is synthetic (it carries no future real information); fine-tuning respects rolling origins.

### 7.7 Combination — hierarchical Bayesian stacking

- Each candidate (world-model simulator, each challenger-adjusted simulator, each direct challenger) produces a **predictive distribution over finishing orders** (as PL parameters or simulation samples).
- **Bayesian stacking** (Yao, Vehtari, Simpson & Gelman) chooses weights that maximise the rolling-origin **log predictive density**. **Hierarchical stacking** (Yao et al.) lets the weights vary with context: checkpoint, wet vs dry, era (reset year or not), street vs permanent circuit. Each model is trusted where it has earned it.
- Mixing happens at the **distribution level** (mixture of simulators / PL mixtures), so outputs stay **coherent** (P(win) ≤ P(podium) ≤ P(points)) and usable by the what-if, live and championship modules.

### 7.8 Calibration and uncertainty

- **Temperature scaling** of order distributions per (session type × checkpoint), fitted on rolling log-loss.
- **Conformal prediction:** split-conformal ranges for finishing position + **conformal risk control** for top-k sets ("podium set covers the true podium with ≥ 90% probability").
- Reliability diagrams per driver tier and per checkpoint, published in the Model Lab.

### 7.9 Season and championship simulator

Simulate all remaining events with car-state random walks propagated forward (uncertainty grows with distance), sprint weekends, projected PU penalties and reliability → title odds, points fan charts, **magic numbers and clinch scenarios**, team-orders sensitivity.

### 7.10 Live in-race engine (C5)

OpenF1 MQTT/WebSocket (intervals, positions, stints, pit, race control, weather) → **sequential Monte Carlo (particle filter)** over the race state + **amortized posterior refresh** of pace parameters every lap (Tier 3, milliseconds) → re-simulate 10k continuations in Rust → live P(win/podium), pit windows, undercut threat, SC-in-next-N-laps probability. Fallback: particle filter only, if the misspecification detector fires.

### 7.11 Claude document and analyst layer

| Job | Model and features | Output | Safeguards |
|---|---|---|---|
| **FIA and Pirelli PDF → tables** (upgrades, PU usage, stewards' penalties, compound nominations) | **Claude Opus 5.5** (`claude-opus-5-5`), **PDF document input**, **structured outputs** (`output_config.format`, schema-validated parsing), **Message Batches API** (asynchronous, ~50% cost), **prompt caching** of the fixed taxonomy/system prompt, `effort: low` for routine pages | Rows in `curated.upgrades`, `pu_usage`, `penalties`, `compound_nominations` with `observed_at` = document publication time | ≥10% human-verified sample per season; accuracy gate ≥ 95%; deterministic re-runs cached by document hash |
| **Upgrade semantics** | Same; categorisation into floor, front wing, rear wing, sidepod/cooling, suspension, beam wing, other + "major package" flag | `u_{k,e}` upgrade score for the world model | Ablation must show benefit (§10) |
| **News → timestamped events** *(gated experiment)* | Same, with structured outputs over curated, licensed news feeds | Events (illness, confirmed PU change, team-orders statement) with `observed_at` | Strict knowability timestamps; off by default until it proves itself |
| **Grounded race narratives** | Opus 5.5 with the prediction bundle passed as a document with **citations** enabled (every claim cites model output) | Storylines in the Hub, report text | No numbers that aren't in the bundle; automated check that every number matches the source |
| **"Ask f1predict" analyst** | Opus 5.5 + **strict tool use** (Tool Runner) over our API: `get_predictions`, `simulate(scenario)`, `explain(driver)`, `compare`, `backtest_stats` | A chat panel answering "what does Norris need at Vegas?" by actually running simulations | Tools are read-only/sandboxed; rate-limited; refusal handling with server-side fallback (`fallbacks: "default"`) |

**Cost control:** batch + caching first; the model choice per job (e.g. a smaller model for bulk extraction) is a cost decision for the owner, measured against the extraction accuracy gate before any switch.

---

## 8. The factor catalogue

✱ = new compared with the legacy repo. Layer key: **W** = world model (state/observation/hazard), **S** = simulator, **Q** = qualifying/grid, **C** = challengers, **D** = Claude document layer.

### 8.1 Driver

| Factor | Source | Layer | Phase |
|---|---|---|---|
| Race pace (time-varying, GP/random walk) | Laps + results | W | 3–4 |
| One-lap pace ✱ | Qualifying and qualifying-sim laps | W, Q | 4–5 |
| Wet skill ✱ | Wet laps, wet races since 1950 | W, S | 4 |
| Tyre management ✱ | `ω_d` | W, S | 4 |
| Consistency ✱ | `σ_d` | W, S | 4 |
| Start / launch ✱ | Lap-1 position changes | S (soft twin) | 6 |
| Overtaking / defending ✱ | Lap-level passes | S (soft twin) | 6 |
| Incident / mistake propensity ✱ | Incidents, red flags | W hazards | 3, 6 |
| Penalty propensity ✱ | Stewards' decisions | D, S | 8 |
| Circuit and circuit-type affinity | Laps + embedding | W | 4 |
| Age and experience curve ✱ | DOB, starts | W | 3 |
| Rookie prior (later: feeder series) ✱ | F1DB (+F2/F3) | W | 3, 13 |
| Championship situation ✱ | Standings | C | 7 |
| Home race ✱ | Nationality | C | 7 |
| Teammate dynamics / team orders ✱ | Gaps, patterns | S, season | 6–7 |
| Reserve / substitute ✱ | Entry lists | W | 4 |

### 8.2 Car and team

| Factor | Source | Layer | Phase |
|---|---|---|---|
| Car pace (dynamic, F-factor) ✱ | Laps | W | 4 |
| Circuit-type profile ✱ | Laps × geometry/embedding | W | 4 |
| Straight-line speed ✱ | Speed traps | W prior | 4 |
| Upgrade trajectory ✱ | FIA submissions | D → W | 8 |
| Development-rate prior (aero-testing allowance) ✱ | Regulations | W | 8 |
| Tyre-thermal behaviour ✱ | Laps | W | 4 |
| Mechanical reliability (fixed taxonomy) | DNFs | W hazards | 3 |
| PU supplier pace and reliability ✱ | Entity model | W | 3–4 |
| PU element age and penalty risk ✱ | FIA PU documents | D → Q, W | 8 |
| Pit-crew speed, slow stops ✱ | Pit stops | S | 6 |
| Strategy style ✱ | Stints | S | 6 |
| Lineage continuity ✱ | Lineage table | W | 3 |
| Learned reset sensitivity ✱ | 1950+ resets | W | 3 |

### 8.3 Circuit

| Factor | Source | Layer | Phase |
|---|---|---|---|
| Geometry and **learned embedding** ✱ | Telemetry | W loadings | 4 |
| Overtaking difficulty (measured, gradient-calibrated) ✱ | Laps | S | 6 |
| Pit-lane loss ✱ | Laps | S | 6 |
| SC/VSC/red hazard ✱ | Track status | W, S | 6 |
| Lap-1 hazard ✱ | Lap 1 | S | 6 |
| Degradation / abrasion | Laps + Pirelli | W | 4, 8 |
| C-number nomination ✱ | Pirelli | W, S | 8 |
| Track evolution ✱ | Session laps | W | 4 |
| Altitude | Elevation | W | 4 |
| Street circuit / walls | Entities | W hazards | 6 |
| Grid-to-Turn-1 distance, grid side ✱ | Geometry + starts | S | 6 |
| New layout / resurfacing ✱ | Layout versions | W (wider variance) | 4 |

### 8.4 Conditions

| Factor | Source | Layer | Phase |
|---|---|---|---|
| Air and track temperature per session ✱ | FastF1, ERA5 hourly | W, S | 4 |
| Humidity | Same | W | 4 |
| Wind relative to the main straight ✱ | ERA5 + geometry | W | 4 |
| Rain timeline (ensemble members) ✱ | Open-Meteo ensemble | Q, S | 6 |
| Forecast uncertainty ✱ | Ensemble spread | Q, S | 6 |
| Night / twilight ✱ | Session times | W | 4 |

### 8.5 Weekend, regulations, race dynamics

| Factor | Source | Layer | Phase |
|---|---|---|---|
| Sprint format (less practice) | Schedule | W, S | 4 |
| Practice running lost ✱ | Laps | W (variance) | 4 |
| Long-run / qualifying-sim pace (HMM-labelled) ✱ | FP laps | W | 4 |
| Pre-season testing ✱ | FastF1 | W (C0 prior) | 4 |
| Confirmed / projected grid penalties ✱ | D | Q | 8 |
| Points systems per season | Rules table | S, season | 1 |
| 2026 active aero / Overtake Mode / energy limits ✱ | 2026 data | W, S | 4, 6 |
| Undercut / overcut, SC timing ✱ | Simulator | S | 6 |
| Market prior (optional) ✱ | Betfair | C | 13 |

---

## 9. Small-data strategy — the 18 techniques

1. **Learn from laps, not results** (≈200× more signal).
2. **Fuse every session** into one latent pace.
3. **Use 1950+ history for hyper-parameters**: reset dynamics, drift, rookie priors, DNF bases.
4. **Hierarchical partial pooling** throughout.
5. **Teammate and transfer identification** of driver vs car.
6. **Listwise (Plackett–Luce) likelihoods** with censored retirements.
7. **Low-rank car × circuit factorisation**.
8. **Physics-informed priors** (fuel, degradation shape, pit loss, wet offsets).
9. **Telemetry geometry + a learned circuit embedding** for new and changed layouts.
10. **Amortized inference trained on simulations**: the simulator becomes a training set for inference itself.
11. **Simulation-pretrained foundation model (F1-PFN)**: tens of millions of synthetic races teach in-context forecasting; real data fine-tunes.
12. **Tabular foundation models** (TabICLv2 / TabPFN-2.5): priors learned from millions of synthetic tables, the strongest known learners for tables of fewer than 50k rows.
13. **Differentiable-simulator calibration**: learn race-dynamics parameters from aggregate statistics, where per-event labels are too few.
14. **Documents → structured signals** (upgrades, PU usage, penalties, compounds) via Claude.
15. **Online Bayesian updating** through the weekend (C0 → C5).
16. **Hierarchical stacking**: use each model only where it has proven itself.
17. **Honest uncertainty**: wide posteriors when data is thin, calibrated to reality.
18. **Optional crowd information** (market prices), opt-in and benchmarked.

---

## 10. Evaluation, honesty and MLOps

### 10.1 Rolling-origin protocol

For each event (2019–2026 for lap-era components; 1996+ for the results-era layer), at each checkpoint C0–C4: pin the **DuckLake snapshot**, apply `as_of(checkpoint_time)`, update the posterior, predict, score. **Locked holdout** (the last N events before model freeze) → **forward test** on the live 2026 races.

### 10.2 Metrics

| Target | Metrics |
|---|---|
| Winner | Log-loss, Brier, top-1 |
| Podium / points / DNF | Brier, ECE, reliability diagrams, AUC |
| Full order | RPS, PL log-likelihood, Spearman, Kendall, NDCG@10, exact podium, teammate head-to-head |
| Position distribution | CRPS (via `scoringrules`), conformal coverage and width |
| Qualifying | Pole log-loss, Q3-cut F1, RPS |
| Pace | Clean-lap RMSE/MAE (% of lap), interval coverage |
| Simulator | Posterior-predictive-check distances (energy distance / KS) |
| Inference quality | **SBC rank histograms**, coverage, Tier 3 vs Tier 2 vs Tier 1 KL and moment agreement, misspecification-alarm rate |
| Championship | Log-loss of title odds per round |

### 10.3 Baselines

Grid order · qualifying order · previous race · championship order · uniform · results-era ratings only · **legacy v0** · *(optional)* market.

### 10.4 Statistical rigour

Paired bootstrap (10k) over events; mean difference + 95% CI; Diebold–Mariano as a secondary test; multiplicity correction for challenger selection. **Champion/challenger automation:** the CI job evaluates every challenger and proposes promotion only when gate G9 passes.

### 10.5 Prediction ledger

Before each session, prediction bundles are hashed (SHA-256), committed with a timestamp and shown in the UI. Scores are appended after the event. `f1p ledger verify` re-checks the hashes.

### 10.6 Testing strategy

| Kind | Examples |
|---|---|
| Unit | Points tables, taxonomy, race refs, units |
| Property-based (Hypothesis) | Simulator invariants (valid permutations, coherent probabilities, Σ P(win) = 1) |
| Data contracts | Pandera + reconciliation |
| **Leakage** | Future-perturbation invariance (randomise data after t → identical outputs at t) |
| **Parameter recovery** | Simulate from known parameters → fit with Tiers 1/2/3 → recover within intervals |
| **SBC** | Simulation-based calibration for every inference tier |
| **Soft-twin convergence** | As temperature → 0, the soft simulator's statistics → the exact simulator's |
| Cross-implementation equivalence | JAX ↔ Rust ↔ WASM distributions match (KS/energy) for fixed seeds |
| Golden / regression | Frozen snapshot → frozen predictions |
| Performance | pytest-benchmark + Criterion (Rust) budgets in CI |
| Web | Vitest, Playwright end-to-end, visual regression, Lighthouse CI, axe |

### 10.7 MLOps

- **Registry:** each release bundle = code commit + DuckLake snapshot id + configuration + posterior artefacts + network weights (+ ONNX) + diagnostics + backtest scores with CIs + calibration plots + model card.
- **Observability:** OpenTelemetry traces for pipeline stages and the API; MLflow runs per fit.
- **Workflows:** `ci.yml` (lint, types, tests, Rust, web, budgets); `weekend.yml` (cron after each session: fetch → gates → SVI refresh → simulate → stack → ledger → publish → redeploy); `gpu-train.yml` (dispatches F1-PFN pretraining, amortized-network training and TabM sweeps to serverless GPU; artefacts back to R2); `release.yml` (data and model snapshots); a Monday reconciliation run.

---

## 11. Performance engineering

- **Arrow end to end:** Polars 2 (streaming / out-of-core) ↔ DuckDB ↔ JAX via zero-copy where possible; DuckLake partition pruning; zstd Parquet.
- **JAX discipline:** `jit` all hot paths; `vmap` over chains, simulations, circuits and checkpoints; `lax.scan` over laps; float32 by default (float64 only where diagnostics demand it); static shapes via padding (22-car fields, max-lap padding) to avoid recompilation; persistent compilation cache.
- **MCMC efficiency:** MAMS + Pathfinder initialisation + chains `vmap`ped across cores; reparameterisation (non-centred) of hierarchical blocks.
- **Amortization economics:** train once (GPU hours), then infer in milliseconds forever; retrain only when the world model changes or the misspecification-alarm rate rises.
- **Rust simulator:** structure-of-arrays state, SIMD lap updates, rayon across simulations, deterministic counter-based RNG (Philox) so JAX and Rust streams can be compared.
- **Content-addressed assets:** key = hash(inputs + code + configuration). Unchanged assets are skipped, which keeps weekend refreshes under 2 minutes.
- **Streaming telemetry:** one session at a time; aggregates only; peak < 6 GB.
- **Polite async fetching:** `httpx` + `aiolimiter`, backoff honouring `Retry-After`, ETags; bulk F1DB downloads.
- **Serving:** static marts on a CDN; simulation in the browser (WASM); amortized posteriors and F1-PFN in the browser via ONNX Runtime Web (WebGPU), with the server as fallback.
- **Web:** React Server Components; React Compiler (automatic memoisation); route-level code splitting; 3D and DuckDB-WASM lazy-loaded; Web Workers for all heavy compute; Service Worker caching of event bundles.

---

## 12. The web experience (UI/UX)

### 12.1 Design direction — "Pit wall at night"

Carbon-black base, smoked-glass panels, timing-tower typography, team colours as precise accents, and motion that **expresses probability** (particles, fans, ribbons). Open-licence fonts (e.g. Titillium Web / Inter Tight + JetBrains Mono, tabular figures). OKLCH design tokens for light and dark; a colour-blind-safe mode; WCAG 2.2 AA; `prefers-reduced-motion` respected.

### 12.2 Technology

| Concern | Choice |
|---|---|
| Framework | Next.js 16.x (App Router, Turbopack, RSC, ISR/static export), React 19 + **React Compiler** |
| Language / styling | TypeScript strict; Tailwind CSS v4; shadcn/ui on Radix (restyled) |
| Motion | Motion (Framer Motion successor) + the **View Transitions API** for route changes |
| 3D | **three.js WebGPURenderer** (WebGL2 fallback) via react-three-fiber/drei; **TSL** shaders; instanced cars; GPU particles for probability fields |
| Charts | visx + d3 (custom), canvas/WebGL for dense views (spaghetti of 1,000 traces) |
| Compute in browser | **Rust/WASM simulator** (SIMD + threads; COOP/COEP headers) in Web Workers; **ONNX Runtime Web** (WebGPU) for the amortized posterior and F1-PFN; **DuckDB-WASM** for Model Lab analytics over Parquet |
| Data | TanStack Query; Arrow IPC/Parquet for big payloads; JSON for small ones; types generated from JSON Schema / OpenAPI; zod at the edges |
| State | Zustand (scenarios), **nuqs** (URL state → shareable what-ifs) |
| Live | Server → browser via WebSocket/SSE (bridged from OpenF1 MQTT) |
| Quality | Storybook (design system), Vitest, Playwright, visual regression, Lighthouse CI, axe; Biome lint/format |

### 12.3 Information architecture and pages

| Route | Page | Highlights |
|---|---|---|
| `/` | **Race Weekend Hub** | 3D circuit with cars running a sampled simulation; countdown; checkpoint timeline (C0–C5); weather strip with rain-probability band drawn from the ensemble members; headline odds; "what changed" diff; Claude-written storylines with citations |
| `/e/[event]/[session]` | **Predictions** | Timing-tower rows with **position heat-strips**; P(win/podium/points/DNF); expected points; conformal range; animated reorder across checkpoints; post-event score card |
| `/e/[event]/sim` | **Simulation Theatre** | Lap-by-lap replay (2D/3D); gaps, tyre bars, SC banners; "roll again"; spaghetti of position traces; strategy explorer |
| `/live` | **Live mode** | Live tower + live P(win) chart (every-lap amortized updates), pit windows, undercut threats, SC hazard, event log |
| `/championship` | **Championship Centre** | Title-odds river chart, points fan charts, magic numbers and clinch scenarios, path-to-title calculator, constructors |
| `/lab` | **What-if Lab** | Drag the grid; paint rain on a lap timeline; car pace ± tenths; **driver swap** (X in car Y); reliability toggles; SC on lap N; force strategies; **in-browser** simulation + posterior; share link |
| `/ask` | **Ask f1predict** | Claude analyst that runs real simulations through tools and shows its working |
| `/drivers/[id]` | **Driver profile** | Skill trajectories with bands; sub-skill radar with uncertainty; teammate head-to-head; circuit affinity; career |
| `/teams/[id]` | **Team profile** | Car-factor trajectory with **upgrade markers**; circuit fingerprint; reliability; pit stops; PU usage gauges; lineage |
| `/circuits/[id]` | **Circuit profile** | Speed-coloured racing line (3D); embedding neighbours ("most similar circuits"); overtaking hotspots; SC history; degradation; climatology; layout versions |
| `/model` | **Model Lab** | Leaderboard vs baselines with CIs; **stacking weights by context**; reliability diagrams; SBC plots; ledger; model cards; data freshness; DuckDB-WASM explorer |
| `/model/how-it-works` | **Methodology** | Scroll-driven interactive explainers for each layer |
| `/compare` | **Compare** | Driver, team and season comparisons |
| global | **⌘K palette** | Race shorthand, search, actions |

**"Why?" drawer** on every prediction: a waterfall of expected pace (car factors × circuit loadings, driver skill, grid, weather, reliability, challenger corrections), in seconds per lap *and* in win-probability terms, plus each model's stacking weight for this context.

### 12.4 Quality bar

Mobile-first (≥ 360 px); keyboard-first; screen-reader summaries for every chart; Lighthouse budgets; per-prediction Open Graph cards; PWA with an offline cached last event.

---

## 13. CLI & TUI

Typer + Rich + prompt_toolkit, on the same core.

| Command | Purpose |
|---|---|
| `f1p` | Interactive shell (history, completion, warm caches) |
| `f1p update [--event] [--full]` | Incremental / full refresh |
| `f1p predict monaco26 [--session] [--checkpoint C3]` | Predictions with distribution bars |
| `f1p sim vegas26 [--n 50000] [--rain-from 30] [--sc-lap 12]` | Simulate a scenario (Rust engine) |
| `f1p whatif …` | Same scenario language as the web lab |
| `f1p title [--round]` | Championship odds and clinch scenarios |
| `f1p fit [--tier mcmc\|svi\|amortized] [--profile quick\|standard\|full]` | Fit / refresh the world model |
| `f1p train-gpu [f1pfn\|npe\|tabm]` | Dispatch GPU training jobs |
| `f1p challengers [--evaluate]` | Run the champion/challenger evaluation |
| `f1p backtest [--from 2022] [--checkpoint all]` | Evaluation report |
| `f1p explain <driver> <event>` | Waterfall decomposition |
| `f1p ask "…"` | Claude analyst in the terminal |
| `f1p ledger verify` | Verify prediction hashes |
| `f1p doctor` | Data freshness, gates, environment |
| `f1p live` | **Textual TUI** live timing + live probabilities |
| `f1p serve` / `f1p web` | Run the API / web locally |

All commands support `--json` output.

---

## 13a. Technology bill of materials

Versions are "latest stable at kickoff, then pinned". Items marked † have known caveats to re-check at kickoff.

| Layer | Component | Notes |
|---|---|---|
| Python runtime | CPython **3.13** via uv | 3.14 once BayesFlow and every other dependency supports it |
| Env / lint / types | uv · ruff · basedpyright (CI) · **ty**† (beta; promote at 1.0) | — |
| Data | **Polars 2**† (streaming default — verify) · DuckDB · **DuckLake 1.0** · PyArrow · pandera | — |
| Ingestion | httpx · aiolimiter · tenacity · FastF1 3.8.x · paho-mqtt (OpenF1 live) · pdfplumber (fallback parsing) | — |
| Numerics | **JAX** (CPU locally, CUDA in the cloud) · Optax · OTT-JAX (soft sort) · SoftJAX† | JAX on Apple Metal not relied upon† |
| Bayesian | **NumPyro** · **BlackJAX** (MAMS†, NUTS, Pathfinder) · ArviZ | Confirm MAMS is available in BlackJAX at kickoff; otherwise use MCLMC/NUTS |
| Amortized | **BayesFlow 2** (Keras 3, JAX backend) · `sbi` (cross-check) | Saving/loading API stabilising toward 2.1† |
| Neural | Flax NNX or Equinox · Muon/AdamW | — |
| Tabular | LightGBM · CatBoost · **TabICLv2** · **TabPFN-2.5**† (licence) · TabM | — |
| Tuning | Optuna (multi-objective: log score vs runtime) | — |
| Scoring | `scoringrules` (CRPS, energy score) · custom PL/RPS | — |
| LLM | Anthropic Python SDK · **Claude Opus 5.5** (`claude-opus-5-5`) · PDF input · structured outputs · Batches · prompt caching · strict tool use · citations | — |
| Rust | `f1sim` · rayon · `rand` + Philox · PyO3/maturin · wasm-bindgen + wasm-pack (SIMD, threads) · Criterion | — |
| API | FastAPI · Pydantic v2 · orjson · uvicorn · OpenTelemetry | — |
| CLI | Typer · Rich · prompt_toolkit · Textual | — |
| Web | Next.js 16.x · React 19 + React Compiler · TypeScript · Tailwind v4 · shadcn/ui · Motion · three.js WebGPU† + r3f/drei · visx/d3 · TanStack Query · Zustand · nuqs · zod · DuckDB-WASM · ONNX Runtime Web† · Biome · Vitest · Playwright · Storybook | — |
| Infra | GitHub Actions · Vercel · Cloudflare R2 · Fly.io · serverless GPU provider (per-job) · MLflow | — |

**Considered and rejected (for now):**
- Time-series foundation models for lap pace: our structure is relational (car × driver × circuit), not univariate series.
- JAX-Metal: maintenance risk.
- Dagster: operational weight.
- TabPFN for published production use: licence.

---

## 14. API & serving

- FastAPI + Pydantic v2 + orjson; OpenAPI → generated TS client.
- Endpoints: `GET /events`, `/events/{id}`, `/events/{id}/predictions?session=&checkpoint=`, `POST /simulate` (Rust engine), `POST /posterior` (amortized network, scenario inputs), `GET /drivers/{id}`, `/teams/{id}`, `/circuits/{id}`, `/championship?round=`, `/model/backtest`, `/model/ledger`, `/model/stacking`, `POST /ask` (Claude analyst, streaming), `GET /live/stream` (WebSocket/SSE), `GET /health`.
- Precomputed data is served as static marts through the CDN. The API only handles scenario simulation, posterior queries, the analyst and live streaming. Rate limits on POST routes; ETags everywhere.

---

## 15. Phased roadmap with acceptance gates

Sizes: S ≈ 1–2 focused sessions, M ≈ 3–5, L ≈ 6–10, XL ≈ 10+.

| Phase | Name | Main deliverables | Acceptance gate | Size | Depends on |
|---|---|---|---|---|---|
| **0** | Reset and foundation | Legacy tag/release; clean slate; uv/ruff/types/pre-commit/CI; core (config, ids, as_of, points, regulations) | CI green; points property tests incl. the special cases | M | — |
| **1** | Data platform | F1DB, Jolpica, FastF1, Open-Meteo connectors; DuckLake layers; entity resolution; taxonomy; gates; `as_of`; 2026 up to date | §6.4 gates pass 1950 → now; refresh < 2 min | XL | 0 |
| **2** | Evaluation harness | Rolling origin × checkpoints; metrics (incl. `scoringrules`); bootstrap; baselines incl. legacy re-score; ledger; SBC/PPC tooling | Baseline table published; leakage tests pass | L | 1 |
| **3** | World model v1 — results era + hazards | Dynamic PL (1950+) with learned resets; hazards; Tier 1/2 inference | Beats order baselines at C0/C1; parameter recovery passes | L | 2 |
| **4** | World model v2 — lap level | Clean laps, HMM programmes, geometry + circuit embedding, weather; lap observation model with car × circuit factors | G6; Tier 2 vs Tier 1 agreement | XL | 1, 3 |
| **5** | Qualifying and grid | Knockout generator; penalty model | G5 | M | 4 |
| **6** | Simulator + differentiable twin | JAX reference; soft twin; gradient calibration of start, overtaking and SC dynamics | G7; soft-to-hard convergence test | XL | 4, 5 |
| **7** | Combiner + forward test | LightGBM-PL + CatBoost + **TabICLv2** challengers; hierarchical stacking; calibration; conformal; season simulator; **model freeze → live 2026 forward test** | G1–G4, G9; ledger live | L | 6 |
| **8** | Document intelligence | Claude extraction (upgrades, PU usage, penalties, compounds) with verification; integrated into W/Q | Extraction ≥ 95%; ablation non-negative with CI | L | 4, 7 |
| **9** | Amortized inference (Tier 3) | BayesFlow summary + inference networks; GPU training pipeline; SBC; misspecification fallback | G8 | L | 4, 6 |
| **10** | F1-PFN + more challengers | Simulation pretraining at scale; real fine-tuning; TabM; TabPFN-2.5 (benchmark) | G9 (or documented as not promoted) | XL | 6, 7 |
| **11** | Publish + API | Marts (JSON/Arrow/Parquet) with schemas; FastAPI; ONNX exports | Contract tests; p95 budget | M | 7 |
| **12** | Web app | Design system → Hub → Predictions → Championship → Profiles → Model Lab (DuckDB-WASM) → Methodology → Theatre → 3D (WebGPU) → What-if (server simulation first) → Ask | Lighthouse and accessibility budgets; Playwright; visual review | XL | 11 (design system from Phase 2 with mock marts) |
| **13** | Rust core + in-browser compute | `f1sim` + PyO3 + WASM; ORT-Web posterior; the What-if Lab goes fully client-side | Equivalence tests; 10k sims < 1 s in the browser | L | 6, 12 |
| **14** | Live mode | OpenF1 MQTT; particle filter + amortized updates; live pages; Textual TUI | Replay test on recorded 2026 sessions | L | 9, 12, 13 |
| **15** | Polish and extras | Market benchmark; feeder-series priors; learned strategy policy; WebGPU-compute simulator experiment; OG cards; PWA; launch | — | M | all |

**Critical path:** 0 → 1 → 2 → 4 → 6 → 7 → 11 → 12. Phases 3, 5, 8, 9 and 10 run alongside it. **Calendar intent:** 0–2 immediately; 3–7 for the closing 2026 rounds where possible; 8–13 over the winter; launch before the 2027 opener; live mode in early 2027.

---

## 16. Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Live-timing / FastF1 access changes | Medium | High | Immutable raw; TracingInsights mirror; OpenF1 |
| Rate limits | High | Medium | F1DB bulk; caching; limiters |
| 2026 data quirks (energy deployment unobserved) | High | Medium | 2026-specific parameters; wide early-era uncertainty |
| Practice fuel unknown | High | High | HMM programme labels + latent fuel; validate vs race pace |
| **Amortized posterior wrong on unusual weekends** | Medium | High | SBC + misspecification detector → automatic Tier 2 fallback |
| **Sim-to-real gap for F1-PFN** | Medium | Medium | Fine-tuning with recency weights; stacking only trusts it where it wins; never in evaluation as synthetic data |
| **Differentiable twin diverges from the exact simulator** | Low–Medium | Medium | Annealing + convergence tests; the exact simulator is authoritative |
| MCMC too slow on the full model | Medium | Medium | MAMS + Pathfinder; subset fits; Tier 2 as the operational default |
| Frontier-library churn (BayesFlow API, ty, Polars 2, three.js WebGPU) | Medium | Low–Medium | Pin versions; adapter layers; cross-checks (`sbi`, basedpyright, WebGL2 fallback) |
| Cloud GPU cost creep | Low | Low | Per-job serverless, budgets, retrain only on triggers |
| LLM extraction errors | Medium | Low–Medium | Schemas, verification sample, ablation gating |
| Overfitting the backtest | Medium | High | Locked holdout; forward ledger; CI-based gates; multiplicity control |
| Scope creep | High | High | Phase gates; challengers must earn promotion |
| Licensing (TabPFN-2.5, Open-Meteo, data) | Medium | High | §17; TabICLv2 as the open default |
| Single maintainer | High | Medium | Automation, ADRs, `f1p doctor` |

---

## 17. Legal, licensing and ethics

- **Trademarks:** no official F1, FIA or team logos, fonts or imagery; a clear unofficial / non-affiliated statement.
- **Data:** F1DB CC BY 4.0 (attribution); Open-Meteo non-commercial terms; OpenF1 terms; TracingInsights licences per repo; FIA/Pirelli documents cited.
- **Models:** **TabPFN-2.5 weights** are under a Prior Labs licence aimed at research and internal evaluation, with restrictions on distilling its outputs for commercial use. It is a benchmark-only challenger unless the project's use fits, or a commercial licence or API is obtained. TabICLv2 is the open default.
- **LLM use:** Claude outputs are grounded in our data with citations; extraction is verified; no personal data sent.
- **Gambling:** no betting advice or affiliate links; market data only as a benchmark or opt-in feature, within the provider's terms.
- **Privacy:** no accounts needed; privacy-friendly analytics or none.
- **Code licence:** MIT.

---

## 18. Appendices

### Appendix A — curated entity model (core tables)

| Table | Key columns |
|---|---|
| `seasons` | season, regulation_era, is_reset, points_system_id, sprint_format_id, quali_format_id |
| `circuits` | circuit_id, name, country, lat, lon, tz |
| `circuit_layouts` | layout_id, circuit_id, valid_from, valid_to, length_m, corners, geometry_ref (x/y/z), derived features, embedding |
| `events` | event_id, season, round, layout_id, name, format, date_local |
| `sessions` | session_id, event_id, type (TEST, FP1–3, SQ, S, Q, R), start_utc, end_utc, scheduled_laps |
| `drivers` | driver_id, name, abbreviation, dob, nationality, debut, external IDs |
| `teams` | team_id, lineage_id, name, pu_supplier_id, chassis, aero_testing_allowance |
| `entries` | session_id, driver_id, team_id, car_number, is_reserve, observed_at |
| `results` | session_id, driver_id, position, classified, status_class, status_raw, laps, time_ms, points, grid, observed_at |
| `laps` | session_id, driver_id, lap, lap_ms, s1–s3, compound_c, compound_label, tyre_age, stint, pit_in, pit_out, track_status, deleted, position, gap_ahead_ms, speed_trap_kmh, is_clean, regime, programme_probs |
| `stints` | session_id, driver_id, stint, compound_c, start_lap, end_lap, tyre_age_start, new_tyre |
| `pit_stops` | session_id, driver_id, lap, lane_ms, stationary_ms |
| `track_status` | session_id, start_ts, end_ts, status |
| `race_control` | session_id, ts, category, message, driver_id |
| `weather_session` | session_id, ts, air_c, track_c, humidity, rainfall, wind_ms, wind_dir |
| `weather_hourly` | layout_id, ts, member, temperature, humidity, precip_mm, wind_ms, wind_dir, cloud, source |
| `upgrades` | event_id, team_id, component, category, reason, description, major_flag, extractor_version, verified, observed_at |
| `pu_usage` | event_id, driver_id, element, count, allowance, observed_at |
| `penalties` | event_id, session_id, driver_id, kind, value, reason, observed_at |
| `compound_nominations` | event_id, hard_c, medium_c, soft_c |
| `points_systems` | points_system_id, session, position→points, fastest_lap_rule, multipliers |

### Appendix B — status taxonomy (fixes D1)

| Class | Classified? | Examples | Used for |
|---|---|---|---|
| `FINISHED` | ✓ | Finished | Ranking |
| `LAPPED` | ✓ | +1 Lap, +2 Laps, …, **Lapped** | Ranking |
| `DNF_MECHANICAL` | ✗ (✓ if ≥ 90% distance) | Engine, Gearbox, Hydraulics, Power Unit, Brakes, Suspension, Electrical, … | Mechanical hazard; censored |
| `DNF_INCIDENT` | ✗ (✓ if ≥ 90%) | Accident, Collision, Collision damage, Spun off, Damage, Puncture | Incident hazard; censored |
| `DNF_OTHER` | ✗ | Retired (unspecified), Withdrew, Illness | Weak hazard signal |
| `DSQ` | ✗ | Disqualified, Excluded | Penalty propensity |
| `DNS` / `DNQ` / `DNPQ` | — | — | Excluded |

### Appendix C — constructor lineage (seed; verify in Phase 1)

| Lineage | Entrant names |
|---|---|
| Red Bull junior | Minardi → Toro Rosso → AlphaTauri → RB → Racing Bulls |
| Silverstone | Jordan → Midland → Spyker → Force India → Racing Point → Aston Martin |
| Hinwil | Sauber → BMW Sauber → Sauber → Alfa Romeo → Kick Sauber → Audi (2026) |
| Enstone | Toleman → Benetton → Renault → Lotus → Renault → Alpine |
| Brackley | Tyrrell → BAR → Honda → Brawn → Mercedes |
| Milton Keynes | Stewart → Jaguar → Red Bull |
| New | Haas (2016), Cadillac (2026) |

### Appendix D — points systems (verify every special case in Phase 0)

| Seasons | Race (P1→P10) | Extras |
|---|---|---|
| 2010–2018 | 25-18-15-12-10-8-6-4-2-1 | **2014 Abu Dhabi double points** |
| 2019–2024 | Same | **+1 fastest lap** if in the top 10 |
| 2021 Belgian GP | **Half points** | — |
| 2022+ | Shortened-race scales per the regulations | — |
| 2025+ | Same | **Fastest-lap point removed** |
| Sprint 2021 | 3-2-1 | — |
| Sprint 2022+ | 8-7-6-5-4-3-2-1 | — |
| Pre-2010 | Historic tables from F1DB | Ordering / reconciliation only |

### Appendix E — 2026 regulation notes (verify against FIA text)

- MGU-H removed; MGU-K ≈ 350 kW; nominal ≈ 50/50 combustion/electric; energy-limited deployment ("clipping").
- **Active aero** (straight mode for all on designated straights); **Overtake Mode** (extra energy within ~1 s).
- Smaller, lighter cars; narrower tyres (front −25 mm, rear −30 mm); Pirelli **C1–C5**.
- 11 teams / 22 cars; PU allowances and penalties as in §7.5.
- 2026 calendar: Bahrain and Saudi Arabia **cancelled** (22 rounds); Madrid added.

### Appendix F — glossary

**PL** Plackett–Luce · **SVI** stochastic variational inference · **NUTS** No-U-Turn Sampler · **MAMS** Metropolis-Adjusted Microcanonical Sampler · **NPE** neural posterior estimation · **SBC** simulation-based calibration · **PPC** posterior-predictive check · **PFN** prior-fitted network · **RPS / CRPS** ranked / continuous ranked probability score · **ECE** expected calibration error · **C0–C5** checkpoints · **as_of(t)** knowability-time filter · **DuckLake snapshot** data-version time.

---

## 19. Sources consulted

**Data and domain**
- OpenF1 — <https://openf1.org/>, <https://openf1.org/auth.html>
- FastF1 (PyPI, v3.8.3) — <https://pypi.org/project/fastf1/>; Jolpica docs — <https://docs.fastf1.dev/api_reference/jolpica.html>
- F1DB — <https://github.com/f1db/f1db>
- TracingInsights — <https://github.com/TracingInsights/2026>
- Open-Meteo Historical API — <https://open-meteo.com/en/docs/historical-weather-api>
- FIA Car Presentation Submissions — <https://www.fia.com/system/files/decision-document/2025_british_grand_prix_-_car_presentation_submissions.pdf>
- 2026 PU usage and allocations — <https://www.formula1.com/en/latest/article/how-many-power-unit-components-has-each-driver-used-in-2026-so-far.6DLZYWn0uEpsB0409mFRTF>, <https://racingnews365.com/fia-confirm-future-f1-engine-penalty-rules-after-2026-change>
- 2026 rules — <https://www.formula1.com/en/latest/article/everything-you-need-to-know-about-the-new-f1-rules-for-2026.48bv0VTxhIlhrQXmxercXk>
- Pirelli 2026 — <https://press.pirelli.com/the-range-of-compounds-for-the-2026-season-has-been-set/>
- 2026 calendar — <https://www.skysports.com/f1/news/13519453/f1-confirms-cancellation-of-bahrain-and-saudi-arabian-grands-prix-due-to-war-in-middle-east-as-2026-calendar-reduced-to-22-races>

**Modelling and inference**
- van Kesteren & Bergkamp (2023) — <https://arxiv.org/abs/2203.08489>
- MAMS (NeurIPS 2025) — <https://arxiv.org/abs/2503.01707>; MCLMC — <https://arxiv.org/pdf/2303.18221>
- BayesFlow 2.0 — <https://arxiv.org/abs/2602.07098>, <https://github.com/bayesflow-org/bayesflow>
- TabPFN-2.5 — <https://arxiv.org/pdf/2511.08667>, <https://huggingface.co/Prior-Labs/tabpfn_2_5>
- TabICLv2 — <https://arxiv.org/html/2602.11139>
- TabArena — <https://arxiv.org/abs/2506.16791>
- OTT-JAX soft sort — <https://ott-jax.readthedocs.io/en/latest/tutorials/soft_sort.html>; SoftJAX — <https://arxiv.org/pdf/2603.08824v1>

**Stack**
- JAX on Apple silicon — <https://developer.apple.com/metal/jax/>, <https://developer.apple.com/forums/thread/815139>
- DuckLake — <https://duckdb.org/docs/current/core_extensions/ducklake.html>, <https://ducklake.select/docs/stable/duckdb/usage/time_travel>
- Polars 2.0 (secondary report; verify) — <https://braindetox.kr/en/posts/polars_2_0_streaming_engine_default_2026.html>
- ty — <https://pydevtools.com/blog/ty-beta/>
- three.js WebGPU — <https://www.utsubo.com/blog/webgpu-threejs-migration-guide>
- Next.js — <https://nextjs.org/blog>

Third-party figures marked "verify" or † are checked against primary sources during the relevant phase.
