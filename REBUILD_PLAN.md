# f1predict — Ground-Up Rebuild Plan

> **Status:** planning only. No code is written until this plan is approved.
> **Date:** 15 October 2026. The 2026 season is in progress. The current repo holds data only up to 2026 Round 5 (Canada, 24 May).
> **Scope:** clear the repository and rebuild f1predict from scratch, with three goals:
> 1. the most **accurate, honestly calibrated** F1 prediction model we can build, covering as many race factors as possible;
> 2. a **direct solution to the shortage of training data**;
> 3. a **striking, fast, interactive web experience**, keeping the command line people liked.

---

## Table of contents

0. [Executive summary](#0-executive-summary)
1. [Audit of the current repo — what we keep and what we discard](#1-audit-of-the-current-repo)
2. [Product vision and measurable success criteria](#2-product-vision--measurable-success-criteria)
3. [Decisions (defaults — override any)](#3-decisions-defaults--override-any)
4. [Repository reset procedure](#4-repository-reset-procedure)
5. [Target architecture](#5-target-architecture)
6. [Data platform — beating the small-data problem](#6-data-platform--beating-the-small-data-problem)
7. [Modelling system — layered, probabilistic, simulation-first](#7-modelling-system)
8. [The factor catalogue — every aspect we model](#8-the-factor-catalogue)
9. [Small-data strategy — the 15 techniques, consolidated](#9-small-data-strategy--the-15-techniques)
10. [Evaluation, honesty and MLOps](#10-evaluation-honesty--mlops)
11. [Performance engineering](#11-performance-engineering)
12. [The web experience (UI/UX)](#12-the-web-experience-uiux)
13. [Command line and terminal UI](#13-cli--tui)
14. [API and serving](#14-api--serving)
15. [Phased roadmap with acceptance gates](#15-phased-roadmap-with-acceptance-gates)
16. [Risks and mitigations](#16-risks--mitigations)
17. [Legal, licensing and ethics](#17-legal-licensing--ethics)
18. [Appendices — schemas, taxonomies, reference tables](#18-appendices)
19. [Sources consulted](#19-sources-consulted)

---

## 0. Executive summary

The current f1predict learns from **5,215 race-result rows**, one per driver per race from 2014 to 2026. It predicts finishing position by regression with tree ensembles, then turns that into probabilities by adding noise measured on its own training data. That design has a hard ceiling. The data is thin, the training objective is wrong, the probabilities are miscalibrated, and one data bug labels 6.5% of classified finishers as retirements.

The rebuild changes **what the model learns from** and **how it reasons**:

| | Today | Rebuild |
|---|---|---|
| Unit of learning | 1 row per driver per race (≈5k rows) | Every timed lap, stint, pit stop, sector and session (**≈1M laps**), plus 1950-onward results used to learn priors |
| Sessions used | Race only (plus qualifying and sprint position) | Pre-season testing, FP1–3, Sprint Qualifying, Sprint, Qualifying, Race — all feed the **same** underlying pace estimate |
| Core model | Random Forest / LightGBM regression on finishing position | A **hierarchical Bayesian pace model** feeding a **lap-by-lap race simulator**, corrected by a **learned ranking model** and **calibrated** |
| Probabilities | Gaussian noise using the training-set spread | Full simulation-based distributions, temperature-calibrated with rolling-origin validation |
| Regulation resets | A hand-set 0.35 pull of team ratings toward the mean | **Reset dynamics learned** from every reset since 1950 |
| Upgrades, engine penalties, tyre compounds | Not modelled | Parsed from FIA and Pirelli documents into features |
| Weather | Daily averages | **Hourly** ERA5 reanalysis for past races, ensemble forecasts for upcoming ones, per-minute session weather |
| Evaluation | 29 held-out races, error in positions | Rolling-origin backtest at **6 points in each weekend**, proper scoring rules, baselines, bootstrap confidence intervals, and a **timestamped public prediction ledger** |
| UI | Streamlit | Next.js with a 3D track, simulation replay, what-if lab running in the browser via WebAssembly, live mode, championship centre, model lab |

**Timing:** the 2026 season has about 6 races left, starting with the US GP on 25 October. Phases 0–7 build the model, and the closing 2026 races serve as a **live forward test**. The full UI ships **before the 2027 season opener**. 2027 continues the 2026 rules, which is ideal: the model will have one full season of data under those rules.

---

## 1. Audit of the current repo

### 1.1 Ideas worth keeping (to be reimplemented, not copied)

| Idea | Old location | Fate |
|---|---|---|
| Forgiving race shorthand (`suzuka26`, `vegas26`, `spain26 → Madrid`) | `data_pipeline/racref.py` | **Keep**: rewritten, data-driven from the circuits table, and reused for the web command palette (⌘K) |
| Interactive shell with history, tab completion and warm caches | `cli/shell.py` | **Keep**: rebuilt on the new command line |
| Separate predictions for race, qualifying and sprint | `features/factory.py` (session targets) | **Keep and extend**: all sessions, all weekend checkpoints |
| Regulation-era awareness | `features/regulations.py` | **Keep the idea, replace the mechanism**: learned reset dynamics instead of hand-set constants |
| Monte Carlo, what-if and championship swing | `simulation/`, `whatif/` | **Keep the idea, replace the engine**: lap-by-lap simulator |
| Track climate and physics table | `features/tracks.py` (`_TRACK_ENV`) | **Replace** with telemetry-derived track features plus hourly weather; keep the hand-set values only as *priors* where data is missing |
| Model presets (`test` … `max`) | `models/train.py` | **Replace** with fit profiles (`quick`/`standard`/`full`) for the Bayesian fits and backtests |
| Model cards, data dictionary, feature glossary | `reporting/docs.py` | **Keep**: auto-generated, published in the Model Lab |
| Property-based tests | `tests/` | **Keep the approach**; expand it a lot |

### 1.2 Defects found (each one must not recur)

| # | Defect | Evidence | Rebuild safeguard |
|---|---|---|---|
| D1 | Status **"Lapped" treated as a retirement**: 338 rows (6.5%) of classified finishers counted as DNFs | `jolpica.py:62` only treats `Finished` / `+N Laps` as finished | Explicit status taxonomy (Appendix B), plus a data test that every classified finisher has `classified=True` |
| D2 | **Probabilities use the spread of in-sample residuals.** Tree ensembles memorise their training data, so the spread is wrong; one σ is used for every driver | `train.py::_fit_predictor` | Spreads come only from simulation and out-of-sample calibration |
| D3 | **Wrong objective**: mean absolute error on position, with DNFs set to `n+1` | `factory.py::_target_position` | Plackett–Luce / ranking likelihood; DNFs modelled as a separate hazard and treated as censored in the ranking |
| D4 | **Future information leaks into features**: `f_season_progress` divides by the season's last round; missing values are filled with medians over the whole dataset | `factory.py` | A single `as_of(t)` data access layer, plus "future-perturbation invariance" tests |
| D5 | **Crude expected points**: every podium scored at P3 points, every points finish at P8 points | `predictor.py::_expected_points` | Exact points from the full position distribution, using per-season points tables (Appendix D) |
| D6 | **Thin evaluation**: 29 races, no proper scoring rules, no baselines in the main metric | `train.py::_evaluate` | Rolling-origin backtest at every weekend checkpoint (§10) |
| D7 | **Stale data**: nothing after 2026 Round 5 | `data/master.parquet` | Incremental session-level refresh, automated on race weekends |
| D8 | **Simulation shuffles ranks, not pace**; safety cars modelled as squashing the noise | `simulation/engine.py` | Lap-by-lap state simulator (§7.5) |
| D9 | **O(n²) Python ELO loop, duplicated** (`compute_elo` vs `latest_ratings`); slow per-group loops in feature building | `features/elo.py`, `factory.py` | Vectorised / JAX code, one source of truth for every computation |
| D10 | **Repo clutter**: `__pycache__`, `.DS_Store`, `egg-info`, saved models, Parquet files committed | git tree | Strict `.gitignore`; data and models published as release assets with manifests |
| D11 | Package layout hack (`package-dir = {"f1predict" = "src"}`) plus `sys.path` edits in the dashboard | `pyproject.toml`, `dashboard/app.py` | Standard `src/` layout with uv |
| D12 | Hand-set track constants (degradation, abrasion, overtaking difficulty) | `features/tracks.py` | Values *measured* from lap and telemetry data, with hand-set values kept only as priors |

### 1.3 Baseline numbers to beat (legacy, from `experiments/experiments.jsonl`)

| Model | Races in test | MAE (positions) | Spearman | Winner correct | Podium correct |
|---|---|---|---|---|---|
| Race, `deep` preset | 29 | 3.70 | 0.659 | 55.2% | 66.7% |
| Qualifying, `deep` | 29 | 3.37 | 0.700 | 27.6% | 63.2% |
| Sprint, `deep` | 9 | 3.96 | 0.599 | 55.6% | 70.4% |

These were measured on a small holdout under conditions affected by D1–D4, so they are not directly comparable. Phase 2 re-runs the frozen legacy model inside the **new** evaluation harness so we get a fair comparison.

---

## 2. Product vision and measurable success criteria

**Vision:** the most rigorous publicly available F1 forecasting system. It shows what will happen, how sure it is and why, it updates live through the weekend, and it lets anyone explore counterfactuals instantly.

### 2.1 Accuracy targets (rolling-origin backtest, 2022–2026, plus the live forward test)

These are relative gates, because absolute F1 predictability varies by season. Every gate is reported with a 95% paired-bootstrap confidence interval over races.

| Gate | Target |
|---|---|
| G1 | Race-winner log-loss at the post-qualifying checkpoint beats the **qualifying-order baseline** (a Plackett–Luce model on grid order with fitted temperature) by **≥ 10%** |
| G2 | Beats the **legacy model** on every headline metric at the same checkpoint, with confidence intervals excluding 0 |
| G3 | Calibration: expected calibration error (ECE) **≤ 0.02** for P(podium) and P(points); reliability-diagram slope within [0.9, 1.1] |
| G4 | Pre-weekend (Thursday) predictions beat the "last race's order" baseline on Ranked Probability Score (RPS) |
| G5 | Qualifying: pole-position log-loss and Q3-cut accuracy beat the "previous qualifying order" and "championship order" baselines |
| G6 | Pace model: out-of-sample race-pace RMSE on clean laps ≤ **0.35%** of lap time (≈0.3 s on a 90 s lap). The target is provisional and will be tuned in Phase 4 |
| G7 | Simulator posterior-predictive checks: the simulated distributions of DNF count, safety-car count, positions gained versus grid, and winning margin are not rejected (KS test/energy distance at α=0.01) for ≥ 90% of circuits |
| G8 | *(Optional, if market data is obtained)*: report the gap to the betting market's implied log-loss honestly, without making beating the market a launch requirement |

### 2.2 Performance targets

| Area | Target (MacBook Air M5, 24 GB) |
|---|---|
| Incremental refresh after a session | < 2 min, end to end (fetch → features → refit → publish) |
| Full historical rebuild | < 45 min, limited by download speed and cached afterwards |
| Pace model refresh (warm-started SVI) | < 5 min; full NUTS fit, run weekly: < 60 min |
| Simulator (reference JAX version) | 100k races × 22 cars × ~60 laps in < 10 s |
| Simulator (Rust/WASM in browser) | 10k races in < 1 s on a mid-range laptop |
| Peak memory | < 6 GB for every pipeline stage |
| Web | LCP < 1.5 s, CLS < 0.05, INP < 200 ms, 60 fps animations, a 3D scene that degrades gracefully to 2D |
| API (precomputed endpoints) | p95 < 100 ms |

---

## 3. Decisions (defaults — override any)

I made these calls so planning didn't stall. Each one has a rationale and can be changed.

| # | Decision | Default | Why | Alternative |
|---|---|---|---|---|
| A1 | Web stack | **Next.js (App Router, React 19, TypeScript, Tailwind v4)** | Best ceiling for a striking interactive UI; static export and ISR; mature 3D (react-three-fiber) and chart ecosystems | Python-only (Reflex, or restyled Streamlit): simpler, far less impressive |
| A2 | Python tooling | **uv, Python 3.13, ruff, basedpyright, pytest** | Fast and reproducible; 3.13 has full wheel coverage for JAX, LightGBM and Polars | 3.14 once all wheels are confirmed |
| A3 | Dataframes and storage | **Polars + DuckDB + Parquet (zstd)** | Columnar and multi-threaded; DuckDB gives SQL over Parquet with no server | pandas (slower, more memory) |
| A4 | Probabilistic modelling | **JAX + NumPyro** (SVI for speed, NUTS weekly) | Vectorised, compiled, and runs fine on Apple-silicon CPU | Stan/cmdstanpy (excellent, slower to iterate), PyMC |
| A5 | Ranking model | **LightGBM** (lambdarank plus a custom Plackett–Luce objective) | Strong on small tabular data, fast | CatBoost YetiRank as an extra ensemble member |
| A6 | Simulator implementation | **Reference version in JAX first, then a Rust port** (`crates/f1sim`) exposed via PyO3 *and* WebAssembly | One fast engine for backtests, the API, and instant in-browser what-ifs | JAX only, with what-ifs run on the server |
| A7 | Compute | **Local-first (M5 Air)**; cloud optional for the weekly full NUTS fit or large backtests | No ongoing cost; the plan fits within 24 GB | GPU box for neural research tracks |
| A8 | Hosting | **Vercel** (web, static + ISR), **Cloudflare R2 or GitHub Releases** (data and model snapshots), **Fly.io** (small API, only if needed) | ≈ €0–15/month | Single VPS |
| A9 | Live data | **OpenF1 Sponsor tier (€9.90/month)**, only from Phase 12 | Live REST/MQTT/WebSocket; the free tier has history only | Skip live mode |
| A10 | Orchestration | **In-house content-hashed "asset" DAG** (plain Python) + `just` + GitHub Actions cron | Dagster or Prefect would be overkill for one-person ops | Dagster if the team grows |
| A11 | Experiment tracking | **MLflow, local SQLite backend** (optional UI), plus model cards in git | Standard and inspectable | JSONL only |
| A12 | Notebooks | **marimo** (reactive, stored as plain `.py`, diff-friendly) in `research/` | No JSON-diff noise | Jupyter |
| A13 | Python packaging | **One package `f1p` with subpackages** | Avoids multi-package overhead; boundaries enforced by import-linter | uv workspace with 6 packages |
| A14 | Brand | Product name **f1predict**; **no official F1, FIA or team logos or fonts** | Trademark safety (§17) | Get licences (unrealistic) |

---

## 4. Repository reset procedure

The rebuild clears the working tree but **never destroys history**.

1. **Freeze the legacy code.** Tag the current `main` as `v0-legacy` and push the tag. Record the legacy model's metrics and the exact dependency set in `docs/legacy/LEGACY_SNAPSHOT.md`.
2. **Archive the legacy artefacts.** Upload `data/master.parquet`, `models_store/*.joblib` and `experiments.jsonl` as assets on a `v0-legacy` GitHub Release, so they no longer live in the git tree but stay reproducible.
3. **Clean-slate commit** on the working branch: delete everything except `LICENSE` (to be added: MIT) and this plan (moved to `docs/plan/REBUILD_PLAN.md`).
4. **Scaffold** (Phase 0): the tree in §5.2, `.gitignore` (data, artefacts, caches, `.DS_Store`, `__pycache__`, `*.egg-info`, `.venv`, `node_modules`, `.next`), `.editorconfig`, pre-commit, CI.
5. **Rewrite history? No.** Big binary files already in history stay there; the repo is small enough. `git filter-repo` is an option only if the clone size becomes a problem.
6. **Branch protection:** CI must be green to merge, plus squash merges.

---

## 5. Target architecture

### 5.1 System overview

```
                         ┌──────────────────────────── SOURCES ───────────────────────────────┐
                         │ F1DB (1950+) · Jolpica (1950+, laps 1996+, pit stops 2011+) ·      │
                         │ FastF1 / F1 live-timing archive (2018+) · OpenF1 (2023+, live) ·   │
                         │ TracingInsights mirrors · Open-Meteo (ERA5 hourly + forecasts) ·   │
                         │ FIA documents (upgrades, power-unit usage, penalties, entry lists)·│
                         │ Pirelli compound nominations · [optional] Betfair odds             │
                         └───────────────┬────────────────────────────────────────────────────┘
                                         │  async connectors (rate-limited, cached, immutable raw)
                                         ▼
 ┌──────────────────────────── DATA PLATFORM (Polars + DuckDB + Parquet) ────────────────────────┐
 │ raw/ (as fetched)  →  staged/ (typed, IDs resolved)  →  curated/ (entity model, Appendix A)   │
 │ quality gates (pandera + reconciliation)  ·  as_of(t) access layer (time-travel, leak-proof)  │
 └───────────────┬───────────────────────────────────────────────────────────────────────────────┘
                 ▼
 ┌──────────────────────────── FEATURE & SIGNAL LAYER ──────────────────────────────────────────┐
 │ clean-lap pipeline · stint segmentation · fuel/tyre/evolution corrections · track geometry   │
 │ from telemetry · weather alignment · document-derived features (upgrades, power units)       │
 └───────────────┬───────────────────────────────────────────────────────────────────────────────┘
                 ▼
 ┌──────────────────────────── MODELLING (JAX / NumPyro / LightGBM) ────────────────────────────┐
 │ A Ratings (1950+, dynamic Plackett–Luce)  →  priors                                          │
 │ B Pace model (lap level, hierarchical, dynamic, car×track factors)                           │
 │ C Qualifying / knockout simulator   D Grid & penalty model                                   │
 │ E Lap-by-lap race simulator (start, tyres, pits, safety cars, DNFs, weather, overtaking)     │
 │ F Ranking corrector (LightGBM-PL)   G Calibration   H Season / championship simulator        │
 │ I Live state filter (in-race)                                                                │
 └───────────────┬───────────────────────────────────────────────────────────────────────────────┘
                 ▼
 ┌──────── EVALUATION HARNESS ────────┐   ┌────────── PUBLISHING ──────────────────────────────┐
 │ rolling origin × 6 checkpoints     │   │ marts (JSON/Arrow per event) → R2/CDN              │
 │ proper scores · baselines · CIs    │   │ FastAPI (scenario simulation, search)              │
 │ prediction ledger (hash-stamped)   │   │ Next.js web + WASM simulator · CLI/TUI             │
 └────────────────────────────────────┘   └────────────────────────────────────────────────────┘
```

### 5.2 Repository layout

```
f1predict/
├── pyproject.toml                # uv project, single package "f1p"
├── uv.lock
├── justfile                      # just setup | update | fit | backtest | web | release …
├── src/f1p/
│   ├── core/                     # config (pydantic-settings), ids, clock/as_of, points, regulations, logging
│   ├── sources/                  # one connector per source: f1db, jolpica, fastf1, openf1, openmeteo, fia_docs, pirelli, betfair
│   ├── warehouse/                # raw→staged→curated transforms, DuckDB views, entity resolution, quality gates
│   ├── signals/                  # clean laps, stints, corrections, track geometry, weather alignment, doc features
│   ├── models/
│   │   ├── ratings/              # Layer A
│   │   ├── pace/                 # Layer B
│   │   ├── quali/                # Layer C
│   │   ├── grid/                 # Layer D
│   │   ├── sim/                  # Layer E reference (JAX) + bindings to the Rust core
│   │   ├── rank/                 # Layer F
│   │   ├── calibrate/            # Layer G
│   │   ├── season/               # Layer H
│   │   └── live/                 # Layer I
│   ├── eval/                     # harness, metrics, baselines, bootstrap, ledger, reports
│   ├── pipeline/                 # asset DAG, content hashing, weekend schedules
│   ├── publish/                  # mart exporters (JSON/Arrow), JSON Schemas for the web
│   ├── api/                      # FastAPI app
│   └── cli/                      # Typer + Rich + prompt_toolkit shell; Textual TUI
├── crates/
│   ├── f1sim/                    # Rust simulator core (no_std-friendly maths, rayon)
│   ├── f1sim-py/                 # PyO3 bindings (maturin)
│   └── f1sim-wasm/               # wasm-bindgen bindings for the browser
├── web/                          # Next.js app (pnpm)
│   ├── app/                      # routes (§12.3)
│   ├── components/               # design system + feature components
│   ├── lib/                      # typed API client (generated), state, formatters
│   ├── workers/                  # Web Worker hosting the WASM simulator
│   ├── content/                  # MDX methodology articles
│   └── tests/                    # Vitest + Playwright
├── research/                     # marimo notebooks (not shipped)
├── docs/                         # ADRs, plan, methodology, data dictionary (generated)
├── tests/                        # Python tests mirror src/f1p
├── data/          (gitignored)   # raw/ staged/ curated/ warehouse.duckdb marts/
├── artifacts/     (gitignored)   # models/ reports/ ledger/
└── .github/workflows/            # ci.yml, weekend.yml, release.yml, web.yml
```

Module boundaries are enforced with **import-linter**: `core ← sources ← warehouse ← signals ← models ← eval/publish ← api/cli`. Nothing imports upward.

---

## 6. Data platform — beating the small-data problem

### 6.1 Source inventory

| Source | Coverage | Granularity | Access and limits | Licence and terms | Role |
|---|---|---|---|---|---|
| **F1DB** | 1950 → present, released after every race | Results, qualifying, pre-qualifying, **free practice and warm-up results**, sprint, starting grid, fastest laps, pit stops, driver of the day, standings; entities: drivers, constructors, chassis, **engine manufacturers**, tyre manufacturers, **circuits with layouts and SVGs** | One SQLite/CSV download per release (no rate-limit problems) | **CC BY 4.0** (attribution) | **Historical backbone**; entity reference; layout SVGs |
| **Jolpica-F1** (Ergast successor) | 1950 → present; **laps 1996+**; **pit stops 2011+** | Results, qualifying, sprint, laps (position and time), pit stops, standings | ≈4 req/s burst, ≈500 req/hour (third-party reports; verify); paginated up to 1000 rows; HTTP 429 with Retry-After | Open | Lap and pit history 1996–2017; cross-check against F1DB |
| **FastF1 3.8.x** (F1 live-timing archive) | **2018 → present** incl. **pre-season testing** | Laps (sectors, compound, tyre life, stint, pit in/out, track status, deleted laps, position), car telemetry (~4 Hz: speed, throttle, brake, gear, RPM, DRS), **position x/y/z**, **per-minute session weather**, **race-control messages**, circuit info (corners, marshal sectors, rotation), event schedule | Local file cache; be polite | Unofficial use of F1 timing; personal and non-commercial | **Core lap-level source** |
| **OpenF1** | 2023 → present; **live** | 18 endpoints: car data, location, laps (incl. mini-sectors, speed traps), intervals, position, pit, stints, team radio, weather, race control, session results, starting grid, championship standings | Free: 3 req/s, 30 req/min, history only. Sponsor €9.90/month: live REST/MQTT/WebSocket, 6 req/s | OpenF1 terms | Live mode; cross-validation; intervals |
| **TracingInsights** | Archive 1950+ (CSV); per-session JSON 2018+ | Laps, telemetry, positions | Bulk download from GitHub | Licence differs per repo (CC0 vs Apache-2.0): **verify LICENSE files** | Bulk mirror to avoid hitting live-timing servers; fallback |
| **Open-Meteo** | ERA5 **hourly 1940+** (0.25°); ERA5-Land 0.1°; forecast and **ensemble** APIs | Temperature, humidity, precipitation, wind speed and direction, cloud, radiation | Free for non-commercial use | Non-commercial terms | Session-window weather for past races; probabilistic forecasts for upcoming ones |
| **FIA documents** | ≈2019+ (varies by document type) | **Car Presentation Submissions** (each team's new parts per event: component, reason, geometric change, description), **power-unit element usage per driver**, stewards' decisions (penalties), entry lists, event notes | PDF scraping; polite cadence | Public documents | Upgrade, power-unit and penalty features |
| **Pirelli** | Per event | Dry compound nomination (C1–C5 for 2026; **the C6 was dropped for 2026**), minimum pressures and camber limits, tyre preview | Press releases | Public | Map SOFT/MEDIUM/HARD labels to C-numbers; degradation priors |
| **Betfair historic data** *(optional)* | Varies | Exchange prices | BASIC tier free; PRO paid | Betfair terms | Benchmark only by default; optional "market prior" feature behind a flag |
| **Feeder series (F2/F3)** *(later)* | — | Results | No clean API found; manual curation or terms-checked scraping | Check terms | Rookie priors |

**Volume estimate (orders of magnitude):**

| Data | Rows | Note |
|---|---|---|
| Race results, 1950 → 2026 | ≈ 27k | Used for priors and hyper-parameters, not point estimates |
| Race laps, 1996 → 2026 | ≈ 650k | Times and positions; compound info only from 2018 |
| Practice, qualifying and sprint laps, 2018 → 2026 | ≈ 350k | Includes pre-season testing |
| Telemetry samples, 2018 → 2026 | ≈ 10⁸–10⁹ | Never loaded whole: aggregated to per-lap, per-corner and per-circuit features in a streaming pass |
| Pit stops, 2011 → 2026 | ≈ 15–20k | Lane loss plus stationary time where available |

That takes us from **5k rows to about 1M timed laps**: roughly 200× more signal, plus the structure (teammates, stints, sessions) that lets us separate car from driver.

### 6.2 Storage layers ("medallion")

| Layer | Content | Format | Rules |
|---|---|---|---|
| `raw/` | Exactly what each source returned, plus fetch metadata (URL, time, ETag, hash) | Source format (JSON, SQLite, PDF) | **Immutable**, append-only; never edited |
| `staged/` | Typed, renamed, unit-normalised (ms, kg, °C, UTC), source IDs kept | Parquet, partitioned `season=/event=/session=` | One staging function per source, idempotent |
| `curated/` | Canonical entity model (Appendix A), cross-source IDs resolved, conflicts reconciled | Parquet + DuckDB views | Pandera schemas enforced; every row carries `source` and `observed_at` |
| `signals/` | Clean laps, stints, corrected pace, track geometry, weather windows, document features | Parquet | Content-hashed; recomputed only when inputs change |
| `marts/` | Exactly what the web and API read: per-event prediction bundles, driver and team profiles, backtest results | JSON (gzip/brotli) and Arrow IPC | Versioned with JSON Schema; generated TypeScript types |

### 6.3 Entity resolution (a hidden accuracy killer)

- **Drivers:** map F1DB id ↔ Jolpica `driverId` ↔ FastF1 abbreviation and number per season ↔ OpenF1 `driver_number` per session. Watch for number reuse, reserve drivers in FP1, and mid-season swaps.
- **Constructor lineage** (Appendix C): e.g. Toro Rosso → AlphaTauri → RB → Racing Bulls; Force India → Racing Point → Aston Martin; Sauber → Alfa Romeo → Kick Sauber → **Audi (2026)**; Renault → Alpine; **Cadillac (new in 2026)**. Lineage drives car-continuity priors; the entrant name alone does not.
- **Power-unit supplier per team and season:** e.g. 2026 adds **Audi** and **Red Bull Ford Powertrains**, and Honda moves to Aston Martin (verify every pairing at kickoff). The supplier is a key factor for 2026 pace and reliability.
- **Circuit layout versions:** each `(circuit, layout_version)` is its own entity with an effective date range (e.g. Barcelona's final-sector chicane removed for 2023, Albert Park and Yas Marina reprofiled, **Madrid street circuit new in 2026**). Layout geometry comes from FastF1 position data.
- **Tyre compounds:** the FastF1 label (SOFT/MEDIUM/HARD) plus the Pirelli event nomination give the C-number. Without this, "SOFT" means different rubber at every track.

### 6.4 Data quality gates (fail the pipeline, never warn silently)

1. **Classification integrity:** classified positions form a contiguous 1..k permutation; exactly one winner; finishing statuses map into the taxonomy in Appendix B (**fixes D1**).
2. **Points reconciliation:** recomputed championship points match the official standings after every round, including special cases (2014 Abu Dhabi double points, 2021 Belgian GP half points, fastest-lap point 2019–2024; Appendix D).
3. **Lap consistency:** lap count ≤ race laps; Σ lap times ≈ race time for the winner (±0.5 s); sector sums ≈ lap time.
4. **Cross-source agreement:** F1DB vs Jolpica vs FastF1 results agree; disagreements are logged and resolved by a stated source priority.
5. **Freshness:** after a session ends, the expected data appears within N hours, otherwise alert.
6. **Leakage contract:** every curated row has `observed_at` (when the fact became knowable). `as_of(t)` returns only rows with `observed_at ≤ t`. Post-race penalties have their own `observed_at` (the stewards' decision time), not the race start.

### 6.5 Clean-lap pipeline (turns raw laps into usable pace samples)

For each lap:

1. **Flag and exclude:** in-laps and out-laps, lap 1, laps under SC/VSC/yellow/red (from track status and race-control messages), deleted laps (track limits), laps with pit activity, invalid timing.
2. **Classify the running programme** (practice): short stint on softs with improving sectors → *qualifying simulation*; ≥ 5 consecutive laps on the same tyre → *long run*; installation and aero-rake laps → exclude.
3. **Traffic flag:** within 1.5 s of the car ahead (from intervals or position data) → `dirty_air=1`. Kept, but modelled with a penalty term.
4. **Wet flag:** intermediate or wet tyres, or rain in the session weather → wet regime.
5. **Normalise** lap time to a percentage of a reference lap for the session (or seconds per km), so pace pools across circuits.
6. **Features carried per lap:** compound C-number, tyre age (laps, including laps used in earlier sessions), stint index, fuel index (laps since start), session elapsed time (for track evolution), track and air temperature, wind relative to the main straight (direction from geometry), speed-trap reading, gap to car ahead.

### 6.6 Track geometry from telemetry (replaces hand-set constants)

For each `(circuit, layout_version)`, computed from fastest clean laps across seasons:

- Length; number of corners; **speed histogram** (% of lap under 120, 120–200, 200–260 and over 260 km/h); % full throttle; number and severity of braking zones; longest straight; **elevation change** (from z); main-straight bearing (for wind).
- **Straight-mode / overtake zones** (2026 rules) from race-control data and track documentation.
- **Measured overtaking difficulty:** rate of on-track passes per lap, net of pit cycles and lapped cars, against pace delta (§7.5.4).
- **Pit-lane time loss** measured from in-lap + out-lap versus clean laps.
- **Safety-car, VSC and red-flag hazard** per lap, and lap-1 incident rate (from 2018+ track status; results-era proxies before that).
- **Track evolution rate** (lap-time improvement per hour of running) and **degradation scale** (see Layer B).
- These numbers feed the **track factor loadings** in the pace model (§7.3).

### 6.7 Weather alignment

- **Past races:** hourly ERA5 for each session window (start − 1 h → end + 1 h): temperature, humidity, precipitation, wind speed and direction, cloud cover. Combined with **FastF1 per-minute session weather** (air and track temperature, rainfall flag, wind) where available (2018+), which is better than reanalysis.
- **Upcoming races:** Open-Meteo **ensemble** forecasts give a *distribution* of rain timelines, which the simulator samples directly (§7.5.8). Further out than the forecast horizon, fall back to **circuit climatology** computed from 20+ years of ERA5 for the event's calendar window.
- **Derived:** rain probability per lap window, "crossover risk" (light rain near the slick/inter boundary), heat index, wind component along the main straight.

### 6.8 Features from documents (the unstructured-data advantage)

| Document | Extraction | Feature |
|---|---|---|
| FIA Car Presentation Submissions | `pdfplumber` tables → rows (team, component, reason, description) → **LLM-assisted categorisation** (floor, front wing, rear wing, sidepod/cooling, suspension, beam wing, other) using the Claude API with a fixed taxonomy and JSON schema; **human-verified sample** of ≥ 10%; results cached | Upgrade count per category per team per event; "major package" flag → increases the car-pace jump variance in Layer B |
| FIA power-unit element usage | Table parsing | Elements used vs allowance (2026: 4 ICE/TC/EX, 3 MGU-K/ES/CE per driver per the published rules; verify at kickoff) → **penalty probability** for upcoming events (Layer D) and PU age → reliability hazard |
| Stewards' decisions | Regex/LLM over the decision PDFs | Grid penalties (with time of issue), time penalties, penalty-propensity priors per driver |
| Pirelli previews | Text parsing | C-number nomination, softness rank, prescribed pressures (degradation prior) |
| FIA Aerodynamic Testing Restrictions | Rule table (constructors' position → wind-tunnel and CFD allowance) | **Development-rate prior**: teams lower in the standings get more aero testing |

---

## 7. Modelling system

### 7.0 Principles

1. **Probabilistic end to end.** Every output is a distribution. Point predictions are summaries of it.
2. **Structure first, flexibility second.** A causal structural model (pace → qualifying → race simulation) does most of the work; a discriminative model only corrects what the structure misses.
3. **One latent pace, many observations.** Every session in a weekend informs the same latent car and driver quantities.
4. **Temporal safety by construction.** Every model reads data only through `as_of(t)`.
5. **Every layer must earn its place.** A component joins the production ensemble only if it improves the rolling-origin scores (§10) with confidence intervals that exclude zero.
6. **Six weekend checkpoints** (C0–C5). Each layer produces predictions at each one:

| Checkpoint | When | New information |
|---|---|---|
| C0 | Pre-season / pre-weekend, long range | Ratings, testing, climatology |
| C1 | Thursday (event week) | Upgrades, power-unit usage, forecasts, compound nominations |
| C2 | After FP1 (or after Sprint Qualifying on sprint weekends) | First pace for the weekend |
| C3 | After final practice (or after the sprint) | Long runs, qualifying simulations |
| C4 | After qualifying (pre-race) | Grid and confirmed penalties |
| C5 | In race (live) | Live state |

### 7.1 Layer A — long-history dynamic ratings (1950 → present)

**Purpose:** learn priors and hyper-parameters from the full history, especially **how much regulation resets scramble the order**, how fast car performance drifts within a season, rookie distributions, and baseline DNF rates. These are things 5k modern rows cannot pin down.

**Model:** dynamic Bayesian **rank-ordered logit (Plackett–Luce)**, extending van Kesteren & Bergkamp (2023, *JQAS*), whose multilevel model found the constructor explains about 88% of result variance in 2014–2021.

- Strength `s_{d,r} = κ_d(t_r) + φ_{team(d),season}(r) + ψ_{d,circuit-type}`.
- `κ_d`: driver skill as a slow random walk over races, plus an age curve as the mean function (a learned spline over age and experience).
- `φ`: car strength as a within-season random walk; at season boundaries `φ_{s+1} = ρ·φ_s + η`, where **ρ and the variance of η are learned separately for normal seasons and reset seasons** (1961, 1966, 1983, 1989, 1995, 1998, 2009, 2014, 2017, 2022, 2026). Team lineage links the seasons.
- **Censoring:** retirements are excluded from the ranking at the point they retire (censored), not ranked last. They are modelled separately by a hazard model (mechanical vs incident).
- Era scaling: the noise temperature varies by era, since older eras were less predictable.
- Fit: NumPyro SVI (low-rank multivariate normal guide), checked against NUTS on subsets; ArviZ diagnostics (R-hat, ESS).

**Outputs used downstream:** prior means and variances for driver and car pace in Layer B; the learned reset parameters ρ_reset and σ_reset (the replacement for the hand-set 0.35); DNF base hazards per era; rookie prior.

### 7.2 Layer A′ — reliability and incident hazards

- **Mechanical DNF hazard per lap:** `h_mech = base_era × PU_supplier_effect(season, seasons_since_reset) × team_effect × PU-element-age effect × heat factor`.
- **Incident hazard per lap:** `h_inc = driver_aggression × circuit (walls, lap-1 hotspots) × wet multiplier × field density (lap 1, restarts)`.
- Hierarchical Poisson / discrete-time survival model fitted on 1996+ laps (lap of retirement known), with priors from 1950+ results.
- **Classification rule:** a driver is classified if they complete ≥ 90% of the winner's distance.

### 7.3 Layer B — hierarchical dynamic pace model (lap level, 2018+) — *the core*

For a clean lap `l` by driver `d` (team `k`) in session `s` of event `e`, normalised lap time `y`:

```
y_{dl} = μ_{e,s}                                   # session baseline
       + Σ_f λ_{e,f} · θ_{k,f}(e)                  # car pace = team factor vector × circuit loadings
       + κ_d(e) + κ^{1lap}_d·[quali run] + κ^{wet}_d·wet_s      # driver skill (+ one-lap, wet)
       + β^{fuel}_e · m_{dl}                       # fuel effect (latent fuel load in practice)
       + g_{c,e}(a_{dl}; T_track) · ω_k · ω_d      # tyre degradation: compound c, tyre age a, temperature;
                                                   #   team and driver tyre-management multipliers
       + τ_{e,s}(t_{dl})                           # track evolution within the session (random walk)
       + δ^{air} · dirty_air_{dl}                  # traffic penalty
       + ε_{dl},   ε ~ StudentT(ν, 0, σ_{e,s,d})   # robust noise; σ varies by driver (consistency)
```

**Key design elements**

- **Car-by-circuit factorisation.** Each team has a small vector `θ_k ∈ ℝ^F` (F ≈ 4–5 latent factors, e.g. low-speed, high-speed, straight-line/power, traction, tyre-thermal). Each circuit has loadings `λ_e` whose **priors come from the telemetry geometry** (§6.6). This captures "car X is strong at high-speed circuits" with very few parameters. It is the main tool for generalising to circuits where a car has not yet raced this season.
- **Dynamics and upgrades.** `θ_k(e) = θ_k(e−1) + η_{k,e}` with innovation variance `σ²_base + σ²_upgrade · upgrade_score_{k,e}` (from FIA submissions) + `σ²_reset · 1[reset season start]`. Development-rate priors use Aerodynamic Testing Restriction allowances.
- **One-lap vs race pace.** Qualifying runs get their own driver and car offsets, correlated with race pace (2026 energy management, "clipping", widens this gap).
- **Fuel.** Race fuel follows from lap index with a burn rate `b_e ~ prior(track length)`. Practice stints get a latent starting fuel `m0_stint ~ prior(programme type)`. This is the classic difficulty of reading practice; it is handled explicitly, not ignored.
- **Tyres.** `g_c(a) = r_c·a + q_c·softplus(a − a*_c)` (linear wear plus a learned "cliff"); `r_c` is scaled by the circuit's degradation level, track temperature, and the team and driver multipliers `ω_k`, `ω_d`.
- **Teammate identification.** Two drivers in the same car at the same event separate driver from car directly; drivers moving between teams link the whole graph.
- **Fitting:** NumPyro SVI with minibatches (≈1M laps) and warm starts after each session; weekly full NUTS on the last 3 seasons to validate the SVI approximation. Parameter-recovery tests on synthetic data (§10.6).

**Outputs:** the posterior of race pace, one-lap pace, degradation and consistency for every car-and-driver pairing at the target event and checkpoint. These are sampled jointly, never as independent marginals.

### 7.4 Layer C — qualifying model and knockout simulator

- One-lap pace draws come from Layer B (one-lap component), plus the qualifying-specific execution noise σ_q(driver) and track-evolution gains through Q1→Q3.
- **Exact knockout format per season** (2026 has 22 cars; verify the elimination counts at kickoff, expected Q1 22→16, Q2 16→10, Q3 10), plus the Sprint Qualifying (SQ1–SQ3) format.
- Hazards: deleted lap (track-limits rate by circuit and driver), mistake or crash (red flag, which ends a segment early and favours whoever already set a lap), mechanical issue, rain arriving mid-session (from the forecast ensemble).
- Output: the full distribution of qualifying order → P(pole), P(Q3), P(front row), expected position.

### 7.5 Layer D — grid and penalty model

- Grid = qualifying order + **penalties**: confirmed (from stewards' documents, with `observed_at`) and **projected** (P(new power-unit element this weekend) from the element usage curve against the allowance and the event's position in the season; 2026 rules: 10 places for the first excess element, 5 for each further one, more than 15 → back of the grid).
- Pit-lane starts (parc fermé changes), DNS risk.

### 7.6 Layer E — lap-by-lap race simulator (reference version in JAX → Rust core)

State per simulated race, per car: position, cumulative time, gap ahead, tyre (compound, age), fuel, pit-stop count, damage or DNF flag, penalty seconds; global: track status (green/SC/VSC/red), weather state, lap.

**7.6.1 Start and lap 1.** Positions gained or lost as a function of grid slot, grid side (clean vs dirty), the driver's start skill, the car's launch, and the run to Turn 1, all learned from 2018+ position data. Lap-1 incident hazard per circuit.

**7.6.2 Lap time.** A joint draw from the Layer B posterior per car per race (parameter uncertainty), plus lap-to-lap noise, degradation, fuel burn, dirty-air penalty, and wet-pace effects.

**7.6.3 Strategy.** For each car, choose among candidate compound sequences (1-stop, 2-stop, 3-stop, respecting the mandatory two-compound rule in dry races) by minimising expected race time under degradation; apply team-specific aggressiveness. **Reactive rules:** pit under SC/VSC when it is cheap, cover undercuts, switch to inters at the crossover.

**7.6.4 Overtaking.** A pass succeeds with probability `σ(α_e · (Δpace − threshold_e) + overtake-aid term)`. The circuit threshold and slope are learned from 2018+ lap-level position changes (net of pit cycles and lapped cars). **Era-specific in 2026**, since active aero and Overtake Mode change passing. Defending skill per driver.

**7.6.5 Pit stops.** Lane loss per circuit (measured) plus a stationary-time distribution per team (with a heavy tail for slow stops), plus unsafe-release and penalty risk.

**7.6.6 Safety car, VSC and red flags.** Lap-level hazards per circuit, raised on lap 1, in rain, and after incidents (incident DNFs trigger SC/VSC with learned probabilities). Effects: bunching (gaps reset), cheap pit stops, restart chaos; red flag: free tyre change.

**7.6.7 Reliability and incidents.** Hazards from Layer A′, applied per lap.

**7.6.8 Weather.** Each simulated race samples a rain timeline from the forecast ensemble; wet laps switch the pace regime (wet-skill terms, higher noise, higher incident risk).

**7.6.9 Penalties.** Time-penalty hazards (track limits, unsafe release, collisions) per driver and circuit; applied at the flag.

**7.6.10 Scoring.** Exact per-season points rules (Appendix D), sprint scoring, classification rules.

**Validation:** posterior-predictive checks per circuit (gate G7), plus a "simulator vs history" dashboard in the Model Lab.

### 7.7 Layer F — ranking corrector (discriminative)

- **LightGBM with a custom Plackett–Luce objective** (gradient and Hessian of the PL log-likelihood, races as groups); a LambdaRank variant as a check.
- **Inputs:** simulator summaries (P(win), expected position and its spread, P(DNF)), Layer B posterior means and SDs, grid, qualifying gap %, upgrade and power-unit features, weather, circuit factors, rating-layer output, checkpoint indicator.
- **Output:** a **strength offset** per driver, fed back into the simulator as a pace adjustment (architecture "simulator-first"). This keeps every downstream use coherent: what-if, championship and live mode all read the same simulator.
- Alternative architecture (log-linear pooling of the simulator-implied PL strengths and the GBM-PL strengths) is evaluated in Phase 7; the winner is chosen by the backtest.
- **Small-data measures:** recency sample weights, heavy regularisation, monotone constraints (e.g. better grid never worsens expected finish), out-of-fold training only on rolling origins, optional pretraining on simulator-generated seasons (§9).

### 7.8 Layer G — calibration

- **Temperature scaling** of the simulator noise per (session type × checkpoint), fitted to minimise rolling-origin log-loss.
- Check reliability diagrams; apply isotonic recalibration of reported marginals only if needed, then **re-derive coherent orders** (resample from a PL model consistent with the calibrated marginals) so the reported numbers stay mutually consistent (e.g. P(win) ≤ P(podium)).
- **Conformal intervals** for finishing position (split-conformal on rolling residuals) → a guaranteed-coverage "likely range" for every driver.

### 7.9 Layer H — season and championship simulator

- Simulate all remaining events: car pace keeps evolving (random walk; uncertainty grows with distance into the future), sprint weekends, projected power-unit penalties, reliability.
- Outputs: title odds (drivers and constructors), points fan charts, **magic numbers and clinch scenarios** ("X clinches at Las Vegas if…"), probability of each final classification position, team-orders sensitivity (e.g. a teammate's support role).

### 7.10 Layer I — live in-race model (Phase 12)

- A **particle filter** over the latent race state using OpenF1 live intervals, positions, stints, pit stops and race-control messages; it updates the pace posteriors with live laps.
- Every lap: re-simulate 10k continuations from the current state → live P(win) and P(podium), pit-window projections, undercut threat, SC probability for the next N laps.

---

## 8. The factor catalogue

Every factor below names its source, where it enters the model, and the phase that delivers it. ✱ = new compared with the legacy repo.

### 8.1 Driver

| Factor | Source | Layer | Phase |
|---|---|---|---|
| Underlying race pace (time-varying) | Laps + results | A, B | 3–4 |
| One-lap (qualifying) pace ✱ | Qualifying and qualifying-sim laps | B, C | 4–5 |
| Wet-weather skill ✱ | Wet laps, wet races since 1950 | A, B, E | 4 |
| Tyre management ✱ | Degradation multiplier ω_d | B, E | 4 |
| Consistency (lap-time variance) ✱ | Clean-lap residuals | B, E | 4 |
| Start / launch skill ✱ | Lap-1 position changes | E | 6 |
| Overtaking and defending skill ✱ | Lap-level passes | E | 6 |
| Incident / mistake propensity ✱ | Incident DNFs, red flags, spins | A′, C, E | 3, 6 |
| Penalty propensity (track limits, collisions) ✱ | Stewards' decisions | E | 8 |
| Track- and circuit-type affinity | Results + laps | A, B | 3–4 |
| Experience and age curve ✱ | Date of birth, starts | A | 3 |
| Rookie prior (optionally feeder-series record) ✱ | F1DB (+ F2/F3 later) | A | 3, 13 |
| Championship pressure and situation ✱ | Standings at the time | F | 7 |
| Home race ✱ | Nationality vs country | F | 7 |
| Teammate dynamics / team orders ✱ | Teammate gaps, team-order patterns | E, H | 6–7 |
| Reserve or substitute driver ✱ | Entry lists, FP1 rookie running | B | 4 |

### 8.2 Car and team

| Factor | Source | Layer | Phase |
|---|---|---|---|
| Car pace (dynamic) | Laps | B | 4 |
| Circuit-type profile (low-speed, high-speed, power, traction, tyre-thermal) ✱ | Laps × geometry | B | 4 |
| Straight-line speed ✱ | Speed traps | B (loading prior) | 4 |
| Upgrade trajectory ✱ | FIA submissions | B | 8 |
| Development-rate prior (aero-testing allowance) ✱ | Regulations | B | 8 |
| Tyre-thermal behaviour (team degradation multiplier) ✱ | Laps | B | 4 |
| Mechanical reliability | DNFs (with fixed taxonomy) | A′ | 3 |
| Power-unit supplier pace and reliability ✱ | Entity model | A, A′, B | 3–4 |
| Power-unit element age and penalty risk ✱ | FIA PU documents | A′, D | 8 |
| Pit-crew speed and slow-stop risk ✱ | Pit stops | E | 6 |
| Strategy aggressiveness and quality ✱ | Stint history | E | 6 |
| Team lineage continuity ✱ | Lineage table | A | 3 |
| Regulation-reset sensitivity (learned) ✱ | 1950+ resets | A | 3 |

### 8.3 Circuit

| Factor | Source | Layer | Phase |
|---|---|---|---|
| Geometry: speed profile, % full throttle, braking zones, straights, elevation ✱ | Telemetry, position data | B loadings | 4 |
| Overtaking difficulty (measured) | Laps | E | 6 |
| Pit-lane time loss ✱ | Laps | E | 6 |
| SC/VSC/red-flag hazard ✱ | Track status | E | 6 |
| Lap-1 incident hazard ✱ | Lap 1 data | E | 6 |
| Degradation and abrasion (measured, prior from Pirelli) | Laps + Pirelli | B | 4, 8 |
| Compound nomination (C-numbers) ✱ | Pirelli | B, E | 8 |
| Track evolution rate ✱ | Session laps | B | 4 |
| Altitude (power-unit and cooling effects) | Elevation | B | 4 |
| Street circuit / walls | Entity data | A′, E | 6 |
| Distance from grid to Turn 1, grid-side advantage ✱ | Geometry + starts | E | 6 |
| New layout / resurfacing flag ✱ | Layout versions | B (widens variance) | 4 |

### 8.4 Conditions

| Factor | Source | Layer | Phase |
|---|---|---|---|
| Air and track temperature per session ✱ | FastF1 per-minute, ERA5 hourly | B, E | 4 |
| Humidity | Same | B | 4 |
| Wind speed and direction relative to the main straight ✱ | ERA5 + geometry | B | 4 |
| Rain timeline (distribution, not a single number) ✱ | Ensemble forecast | C, E | 6 |
| Forecast uncertainty ✱ | Ensemble spread | C, E | 6 |
| Night or twilight races (temperature drop) ✱ | Session times | B | 4 |

### 8.5 Weekend, regulations and race dynamics

| Factor | Source | Layer | Phase |
|---|---|---|---|
| Weekend format (sprint = less practice → more variance) | Schedule | B, E | 4 |
| Practice running lost (laps completed, missed sessions) ✱ | Laps | B (variance) | 4 |
| Long-run and qualifying-sim pace ✱ | FP laps | B | 4 |
| Pre-season testing ✱ | FastF1 testing sessions | B (C0 prior) | 4 |
| Confirmed and projected grid penalties ✱ | Stewards + PU documents | D | 8 |
| Era and points system per season (incl. 2014 double points, fastest-lap point 2019–24) | Rules table | E, H | 1 |
| 2026 active aero / Overtake Mode / energy-limited deployment ✱ | 2026 data (era-specific parameters) | B, E | 4, 6 |
| Undercut / overcut, SC timing luck ✱ | Simulator | E | 6 |
| Optional market prior ✱ | Betfair | F | 13 |

---

## 9. Small-data strategy — the 15 techniques

1. **Learn from laps, not results.** About 1M timed laps instead of 5k rows (≈200×).
2. **Fuse every session.** Testing, FP, SQ, sprint, qualifying and race all inform one latent pace (Layer B).
3. **Use 1950+ history for hyper-parameters, not point estimates.** Reset dynamics, drift rates, rookie priors and DNF bases are learned where the data is plentiful and applied where it is thin (Layer A).
4. **Hierarchical partial pooling.** Drivers ← population; teams ← lineage and power-unit supplier; circuits ← circuit type; 2026 cars ← learned reset prior. Thin cells borrow strength.
5. **Teammate and transfer identification.** Same-car comparisons separate driver from car; moves between teams connect the whole graph.
6. **Listwise likelihoods.** A Plackett–Luce race of n drivers contributes n−1 conditional choices, not n independent regressions.
7. **Low-rank car × circuit factorisation.** About 5 numbers per team generalise to circuits not yet visited this season.
8. **Physics-informed priors.** Fuel effect, degradation shape (linear + cliff), pit loss, wet-pace offsets come from engineering knowledge and the Pirelli data.
9. **Telemetry-derived circuit features.** Measured, not guessed, so new and changed layouts are placed sensibly at once.
10. **Simulation as a data generator.** The fitted simulator generates realistic synthetic seasons for (a) stress tests, (b) parameter-recovery tests, (c) *optional* pretraining of the ranking corrector, fine-tuned on real data with recency weights. Synthetic data **never** enters evaluation.
11. **Unstructured documents → structured signals.** Upgrades, power-unit usage, penalties, compound nominations: information most models ignore.
12. **Online Bayesian updating through the weekend.** Every session updates the posterior (checkpoints C0–C5), so the model learns *within* the weekend.
13. **Recency weighting and era embeddings** in the discriminative layer; monotone constraints and strong regularisation.
14. **Honest uncertainty instead of false precision.** When data is thin (early 2026, new circuits, rookies), the posterior is wide and the calibrated probabilities say so.
15. **Optional crowd information.** Market prices capture information we can't observe (illness, setup trouble); opt-in, benchmarked separately.

---

## 10. Evaluation, honesty and MLOps

### 10.1 Rolling-origin protocol

- For each event in the evaluation window (**2019–2026** for the lap-era models; **1996+** for the rating layer), at each checkpoint C0–C4: refit (or warm-update) using `as_of(checkpoint_time)`, predict, score.
- **Locked holdout:** the last N events before model freeze are excluded from all tuning; looked at once. Then **forward testing** on live 2026 races.

### 10.2 Metrics

| Target | Metrics |
|---|---|
| Winner | Multiclass log-loss, Brier, top-1 hit rate |
| Podium / points / DNF | Brier, ECE, reliability diagram, AUC |
| Full order | RPS, Plackett–Luce log-likelihood, Spearman ρ, Kendall τ, NDCG@10, exact-podium rate, teammate head-to-head accuracy |
| Position distribution | CRPS, conformal coverage and width |
| Qualifying | Pole log-loss, Q3-cut F1, RPS |
| Pace model | Clean-lap RMSE/MAE (% of lap), interval coverage |
| Simulator | Posterior-predictive check distances (gate G7) |
| Championship | Log-loss of title odds at each round (retrospective) |

### 10.3 Baselines (all re-implemented inside the harness)

Grid order · qualifying order · previous race's order · current championship order · uniform · Layer A ratings only · **legacy v0 model** · *(optional)* market.

### 10.4 Statistical rigour

Paired bootstrap over events (10k resamples) for every comparison; report the mean difference and 95% CI; Diebold–Mariano test as a secondary check; a correction for the number of comparisons when selecting among candidates.

### 10.5 Prediction ledger (proof against hindsight)

Before each session, the published prediction bundle is hashed (SHA-256); the hash and timestamp are committed to `artifacts/ledger/` and shown in the UI. After the event, the scores are appended. Anyone can verify predictions were made in advance.

### 10.6 Testing strategy

| Kind | Examples |
|---|---|
| Unit | Points tables, status taxonomy, race-reference parser, unit conversions |
| Property-based (Hypothesis) | Simulator invariants (one winner per sim, valid permutations, P(win) ≤ P(podium) ≤ P(points), Σ P(win) = 1); points sum identities |
| Data contracts | Pandera schemas; reconciliation (§6.4) |
| **Leakage** | *Future-perturbation invariance*: randomise all data after `t`; features and predictions as of `t` must be bit-identical |
| Parameter recovery | Simulate data from known parameters → fit → recover within credible intervals (Layers A, B, E) |
| Statistical equivalence | Rust simulator vs JAX reference: same distributions (KS/energy tests) for fixed seeds |
| Golden / regression | Frozen data snapshot → frozen predictions; diffs must be explained |
| Performance | pytest-benchmark budgets (§2.2) enforced in CI |
| Web | Vitest (units), Playwright (end-to-end), visual regression screenshots, Lighthouse CI budgets, axe accessibility checks |

### 10.7 Model registry and cards

Each release bundle contains: code commit, data snapshot hash, configuration, fit diagnostics (R-hat, ESS, ELBO curve), backtest scores with CIs, calibration plots, known limitations. It is published to the Model Lab.

### 10.8 Automation

- `ci.yml`: lint (ruff, biome), types (basedpyright, tsc), tests, Rust build and tests, web build and end-to-end tests, budgets.
- `weekend.yml`: cron during event windows (Thursday → Monday, hourly after each scheduled session end): incremental fetch → quality gates → refit → predict → ledger → publish marts → redeploy the web app.
- `release.yml`: data and model snapshot release (R2 / GitHub Releases) with manifest.
- Monday reconciliation run: post-race penalties, final classifications, and new F1DB and Jolpica releases.

---

## 11. Performance engineering

- **Columnar end to end:** Arrow between Polars, DuckDB and JAX; lazy Polars queries; partition pruning; zstd Parquet.
- **No Python loops on hot paths:** JAX `jit`/`vmap`/`scan` for the pace model and simulator; Rust (rayon) for the production simulator; precomputed index arrays.
- **Content-addressed assets:** each pipeline asset's key = hash(input hashes + code version + configuration). Unchanged assets are skipped, which is what keeps weekend refreshes under 2 minutes.
- **Streaming telemetry:** process one session at a time; keep only aggregates; peak memory < 6 GB.
- **Async, polite fetching:** `httpx` + `aiolimiter` per-source limits (Jolpica ≈ 4/s and 500/h; OpenF1 3/s and 30/min free), retries with exponential backoff honouring `Retry-After`, ETag/If-Modified-Since, bulk F1DB downloads instead of thousands of API calls.
- **Warm starts:** SVI starts from the previous posterior; LightGBM refits incrementally.
- **Serving:** everything predictable is precomputed into static marts on a CDN; only scenario simulation needs compute, and that runs **in the browser** (WASM in a Web Worker) or on a tiny API.
- **Web performance:** React Server Components for static content; 3D scenes and heavy charts lazy-loaded; per-route bundle budgets; image and font optimisation; prefetching; a Service Worker to cache event bundles.

---

## 12. The web experience (UI/UX)

### 12.1 Design direction — "Pit wall at night"

- **Mood:** carbon-black base, smoked-glass panels, crisp timing-tower typography, team colours as precise accents, and motion that *expresses probability* (particles, fans, ribbons) rather than decoration.
- **Typography:** an open-licence technical sans (e.g. *Titillium Web* or *Inter Tight*) + tabular-figure monospace (e.g. *JetBrains Mono*) for timing data. No proprietary F1 fonts.
- **Colour system:** design tokens (OKLCH) in light and dark themes; team colours sourced from data with automatic contrast adjustment; a **colour-blind-safe mode** (team colours collide, e.g. similar blues); WCAG 2.2 AA throughout.
- **Motion:** Motion (Framer Motion successor) for layout and shared-element transitions; spring physics; respects `prefers-reduced-motion`.
- **3D:** react-three-fiber + drei: circuits extruded from FastF1 x/y/z (real elevation), cars as instanced meshes, restrained bloom; automatic fallback to 2D canvas/SVG on low-end devices and mobile.

### 12.2 Technology

Next.js (App Router, latest stable at kickoff: 16.x), React 19, TypeScript (strict), Tailwind CSS v4, shadcn/ui on Radix primitives (restyled), Motion, react-three-fiber/drei/postprocessing, visx + d3 for custom charts, TanStack Query, Zustand (scenario state), nuqs (URL state → shareable scenarios), zod (runtime validation), types generated from the publish layer's JSON Schemas / OpenAPI, MDX for methodology, Storybook for the design system, Biome, Vitest, Playwright.

### 12.3 Information architecture and pages

| Route | Page | Highlights |
|---|---|---|
| `/` | **Race Weekend Hub** | Next-event hero: 3D circuit with cars running a sampled simulation; countdown; session timeline (FP1 → race) showing which checkpoint the predictions come from; hourly weather strip with a rain-probability band; headline win and podium odds; "what changed since the last update" diff; auto-written storylines grounded in model outputs |
| `/e/[event]/[session]` | **Predictions** | Timing-tower layout: each driver's row has a **position heat-strip** (full finishing distribution), P(win/podium/points/DNF), expected points, conformal range; checkpoint switcher (C0–C5) with animated reordering; after the event, actual vs predicted with a score card |
| `/e/[event]/sim` | **Simulation Theatre** | Replay one sampled race lap by lap on a 2D/3D map; gaps chart, tyre-strategy bars, SC/VSC banners, retirements; "roll again" button; **spaghetti view** of 100 position traces; strategy explorer (which strategies win, by how much) |
| `/live` | **Live mode** | Live timing tower + live P(win) chart, pit-window and undercut threats, SC-in-next-5-laps probability, event log (Phase 12) |
| `/championship` | **Championship Centre** | Title-odds **river chart** over rounds, points fan chart, magic numbers and clinch scenarios table, "path to the title" calculator, constructors' view |
| `/lab` | **What-if Lab** | Drag-and-drop grid; paint rain onto a lap timeline; car pace slider (± tenths); **driver swap** (put driver X in car Y, possible because driver and car are separated in the model); reliability toggles; inject a SC on lap N; force strategies; results in under 1 s (WASM); side-by-side delta vs baseline; share link |
| `/drivers/[id]` | **Driver profile** | Skill posterior over time with uncertainty bands; skill radar (one-lap, race pace, tyre management, wet, starts, consistency, overtaking) with uncertainty; teammate head-to-head; circuit affinity grid; career timeline |
| `/teams/[id]` | **Team profile** | Car pace trajectory with **upgrade markers** (from FIA documents); circuit-profile fingerprint; reliability; pit-stop distribution; PU element usage vs allowance gauges; lineage history |
| `/circuits/[id]` | **Circuit profile** | 3D/2D layout with a speed-coloured racing line; corners; overtaking hotspots; SC history; degradation profile; climatology; pole-to-win rate; layout versions |
| `/model` | **Model Lab** | Backtest leaderboard vs baselines (with CIs); reliability diagrams; metrics over time; per-event audit; **prediction ledger** with hashes; model cards; data freshness and quality status |
| `/model/how-it-works` | **Methodology** | MDX scroll-driven explainers with interactive diagrams for each layer |
| `/compare` | **Compare** | Driver vs driver, team vs team, season vs season |
| global | **⌘K command palette** | Race shorthand (`monaco26`, `vegas26`), driver and team search, quick actions |

**Explainability everywhere:** each prediction has a "Why?" drawer: a waterfall decomposition of the driver's expected pace (car, driver skill, circuit fit, grid, weather, reliability, ranking correction), in seconds per lap *and* in win-probability terms, plus SHAP values for the ranking corrector.

### 12.4 Quality bar

- Mobile-first responsive design; every page usable at 360 px.
- Accessibility: keyboard-first, focus rings, screen-reader text summaries for every chart, AA contrast, reduced-motion versions.
- Performance budgets as in §2.2, enforced by Lighthouse CI.
- Open Graph images generated per prediction (shareable cards); PWA install; offline cached last event.
- A design-system pass first (tokens → primitives → patterns in Storybook), then pages. Figma optional for the visual exploration.

---

## 13. CLI & TUI

Typer + Rich + prompt_toolkit, the same core as the web app.

| Command | Purpose |
|---|---|
| `f1p` | Interactive shell (history, context-aware completion, warm caches) — kept from v0 |
| `f1p update [--event] [--full]` | Incremental or full data refresh |
| `f1p predict monaco26 [--session race\|quali\|sprint] [--checkpoint C3]` | Predictions table with distribution bars |
| `f1p sim vegas26 [--n 50000] [--rain-from 30] [--sc-lap 12]` | Simulate with scenario flags |
| `f1p whatif …` | Same scenario language as the web lab |
| `f1p title [--round]` | Championship odds and clinch scenarios |
| `f1p backtest [--from 2022] [--checkpoint all]` | Evaluation report |
| `f1p fit [--profile quick\|standard\|full]` | Fit the model layers |
| `f1p explain <driver> <event>` | Waterfall decomposition |
| `f1p ledger verify` | Verify prediction hashes |
| `f1p doctor` | Data freshness, quality gates, environment check |
| `f1p live` | **Textual TUI**: live timing + live probabilities (Phase 12) |
| `f1p serve` / `f1p web` | Run the API / web app locally |

All commands support `--json` output.

---

## 14. API & serving

- **FastAPI + Pydantic v2 + orjson**; OpenAPI → generated TypeScript client.
- Endpoints: `GET /events`, `GET /events/{id}`, `GET /events/{id}/predictions?session=&checkpoint=`, `POST /simulate` (scenario → distribution), `GET /drivers/{id}`, `GET /teams/{id}`, `GET /circuits/{id}`, `GET /championship?round=`, `GET /model/backtest`, `GET /model/ledger`, `GET /health`.
- Precomputed endpoints are served as static mart files through the CDN (the API only for search and scenario simulation). ETag caching; rate limiting on `/simulate`.
- Deployment: Vercel (web), R2 (marts and snapshots), Fly.io (API, only if WASM-in-browser is not enough).

---

## 15. Phased roadmap with acceptance gates

Sizes: S ≈ 1–2 focused sessions, M ≈ 3–5, L ≈ 6–10, XL ≈ 10+. Each phase lands as one or more PRs with green CI.

| Phase | Name | Main deliverables | Acceptance gate | Size | Depends on |
|---|---|---|---|---|---|
| **0** | Reset and foundation | `v0-legacy` tag and release; clean slate; uv, ruff, basedpyright, pre-commit, CI skeleton; core (config, ids, as_of clock, points, regulations tables) | CI green; points tables pass property tests incl. the special cases in Appendix D | M | — |
| **1** | Data platform v1 | Connectors: F1DB, Jolpica, FastF1, Open-Meteo; raw→staged→curated; entity resolution incl. lineage, PU supplier, layout versions; status taxonomy; quality gates; `as_of`; the 2026 season fully up to date | All §6.4 gates pass for 1950→now; points reconcile for every season; refresh after a session < 2 min | XL | 0 |
| **2** | Evaluation harness and baselines | Rolling-origin harness for all checkpoints; metrics; bootstrap; baselines incl. **legacy v0 re-scored**; ledger; first Model Lab report (static) | Baseline table published; leakage tests pass | L | 1 |
| **3** | Layer A ratings + A′ hazards | Dynamic PL 1950+; learned reset dynamics; DNF/incident hazards | Beats grid/championship-order baselines at C0/C1; parameter-recovery tests pass | L | 2 |
| **4** | Signals + Layer B pace model | Clean-lap pipeline; stint classification; telemetry circuit geometry; weather alignment; hierarchical pace model with car×circuit factors, tyres, fuel, evolution | Gate G6; posterior checks; SVI vs NUTS agreement | XL | 1, 3 |
| **5** | Layers C + D qualifying and grid | Knockout simulator; penalty model (confirmed + projected) | Gate G5 | M | 4 |
| **6** | Layer E simulator (JAX reference) | Full lap-by-lap engine (§7.6) | Gate G7; performance budget (§2.2) | XL | 4, 5 |
| **7** | Layers F + G + H | Ranking corrector; architecture selection; calibration; conformal; championship simulator; **model freeze → forward test on remaining 2026 races** | Gates G1–G4; ledger running live | L | 6 |
| **8** | Document intelligence | FIA upgrades (LLM-assisted + verified), PU usage, stewards' penalties, Pirelli nominations, aero-testing table; integrated into B/D/F | Ablation shows a non-negative improvement with CI; extraction accuracy ≥ 95% on the verified sample | L | 4, 7 |
| **9** | Publish + API | Mart exporters with JSON Schemas; FastAPI; generated TS client | Contract tests; p95 budget | M | 7 |
| **10** | Web app | Design system → Hub → Predictions → Championship → Profiles → Model Lab → Methodology → Simulation Theatre (2D) → 3D → What-if Lab (server sim first) | Lighthouse and accessibility budgets; Playwright end-to-end; visual review | XL | 9 (the design system can start after 2, using mock marts) |
| **11** | Rust simulator core | `f1sim` crate; PyO3 + WASM bindings; statistical equivalence with the JAX reference; What-if Lab switches to in-browser simulation | Equivalence tests; 10k sims < 1 s in the browser | L | 6, 10 |
| **12** | Live mode | OpenF1 Sponsor feed; particle filter; live pages; Textual TUI | Replay test on recorded 2026 sessions | L | 7, 10 |
| **13** | Polish and extras | Market benchmark (optional), feeder-series rookie priors, OG cards, PWA, documentation, launch | — | M | all |

**Critical path:** 0 → 1 → 2 → 4 → 6 → 7 → 9 → 10. Phases 3, 5 and 8 overlap with it; the web design system runs in parallel from Phase 2.

**Calendar intent:** Phases 0–2 immediately; 3–7 aimed at the final 2026 rounds (US GP 25 Oct → Abu Dhabi) as a forward test where possible; 8–11 over the winter; launch for the 2027 opener; live mode during early 2027.

---

## 16. Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| F1 live-timing / FastF1 access changes | Medium | High | Immutable raw cache; TracingInsights mirror; OpenF1 as an alternative |
| Rate limits (Jolpica, OpenF1) | High | Medium | F1DB bulk download; caching; limiters; off-peak backfills |
| 2026 data quirks (new rules; energy deployment unobserved) | High | Medium | 2026-specific parameters with hierarchical priors; widened uncertainty early in the era |
| Practice fuel loads unknown → biased pace | High | High | Latent fuel per stint; programme classification; validated against race pace; down-weight FP if it doesn't help |
| Bayesian fits too slow | Medium | Medium | SVI + warm starts; minibatches; NUTS only weekly or on subsets |
| Overfitting the backtest | Medium | High | Locked holdout; forward-test ledger; few top-level degrees of freedom; CI-based gates |
| LLM extraction errors in documents | Medium | Low–Medium | Fixed schema; verified sample; extraction only as features gated by ablation |
| Scope creep | High | High | Phase gates; each layer must beat the previous one to ship |
| Rust/WASM complexity | Medium | Medium | Reference JAX simulator first; the Rust port is optional for launch (server-side what-ifs work) |
| Licensing and trademarks | Low–Medium | High | §17 |
| Single maintainer | High | Medium | Automation, documentation, ADRs, `f1p doctor` |

---

## 17. Legal, licensing and ethics

- **Trademarks:** "Formula 1", "F1", team names and logos are trademarks. Use names factually; **no official logos, fonts or imagery**; a clear unofficial / non-affiliated statement in the footer.
- **Data licences:** F1DB CC BY 4.0 → attribution on every page and in the methodology; Open-Meteo **non-commercial** terms → if the project ever becomes commercial, a paid plan is needed; OpenF1 terms; TracingInsights licences verified per repository; FIA and Pirelli documents are public but are cited.
- **Gambling:** no betting advice or affiliate links; probabilities presented as model estimates with calibration evidence. Market data, if used, only as a benchmark or optional feature, in line with the provider's terms.
- **Privacy:** no user accounts needed; analytics privacy-friendly (e.g. Plausible) or none.
- **Code licence:** MIT (as before).

---

## 18. Appendices

### Appendix A — curated entity model (core tables)

| Table | Key columns |
|---|---|
| `seasons` | season, regulation_era, is_reset, points_system_id, sprint_format_id, quali_format_id |
| `circuits` | circuit_id, name, country, lat, lon, tz |
| `circuit_layouts` | layout_id, circuit_id, valid_from, valid_to, length_m, corners, geometry_ref (x/y/z polyline), derived features (§6.6) |
| `events` | event_id, season, round, layout_id, name, format (conventional/sprint), date_local |
| `sessions` | session_id, event_id, type (TEST, FP1–3, SQ, S, Q, R), start_utc, end_utc, scheduled_laps, status_summary |
| `drivers` | driver_id, name, abbreviation, dob, nationality, debut, external IDs (f1db, jolpica, numbers per season) |
| `teams` | team_id (entrant per season), lineage_id, name, pu_supplier_id, chassis, aero-testing allowance |
| `entries` | session_id, driver_id, team_id, car_number, is_reserve, observed_at |
| `results` | session_id, driver_id, position, classified, status_class, status_raw, laps, time_ms, points, grid, observed_at |
| `laps` | session_id, driver_id, lap, lap_ms, s1/s2/s3_ms, compound_c, compound_label, tyre_age, stint, pit_in, pit_out, track_status, deleted, position, gap_ahead_ms, speed_trap_kmh, is_clean, regime (dry/wet), programme (quali_sim/long_run/other) |
| `stints` | session_id, driver_id, stint, compound_c, start_lap, end_lap, tyre_age_start, new_tyre |
| `pit_stops` | session_id, driver_id, lap, lane_ms, stationary_ms (where available) |
| `track_status` | session_id, start_ts, end_ts, status (SC/VSC/RED/YELLOW/GREEN) |
| `race_control` | session_id, ts, category, message, driver_id? |
| `weather_session` | session_id, ts, air_c, track_c, humidity, rainfall, wind_ms, wind_dir |
| `weather_hourly` | layout_id, ts, temperature, humidity, precip_mm, wind_ms, wind_dir, cloud, source (era5/forecast/ensemble member) |
| `upgrades` | event_id, team_id, component, category, reason, description, observed_at |
| `pu_usage` | event_id, driver_id, element (ICE/TC/MGU-K/ES/CE/EX), count, allowance, observed_at |
| `penalties` | event_id, session_id, driver_id, kind (grid/time/drive-through/pit-lane), value, reason, observed_at |
| `compound_nominations` | event_id, hard_c, medium_c, soft_c |
| `points_systems` | points_system_id, session (R/S), position→points, fastest_lap_rule, special multipliers |

### Appendix B — status taxonomy (fixes D1)

| Class | Classified? | Examples (raw statuses) | Used for |
|---|---|---|---|
| `FINISHED` | ✓ | Finished | Ranking |
| `LAPPED` | ✓ | +1 Lap, +2 Laps, …, **Lapped** | Ranking (classified finisher) |
| `DNF_MECHANICAL` | ✗ (✓ if ≥90% distance) | Engine, Gearbox, Hydraulics, Power Unit, Brakes, Suspension, Electrical, Overheating, Fuel pressure, Oil leak, Water leak, Turbo, ERS, … | Mechanical hazard; censored from ranking |
| `DNF_INCIDENT` | ✗ (✓ if ≥90%) | Accident, Collision, Collision damage, Spun off, Damage, Puncture | Incident hazard; censored |
| `DNF_OTHER` | ✗ | Retired (unspecified), Withdrew, Illness | Weak hazard signal |
| `DSQ` | ✗ | Disqualified, Excluded | Penalty propensity; removed |
| `DNS` / `DNQ` / `DNPQ` | — | Did not start / qualify / pre-qualify | Excluded |

"Retired" with no reason is assigned using race-control messages and the lap context where possible (2018+); otherwise `DNF_OTHER`.

### Appendix C — constructor lineage (seed; verify and complete in Phase 1)

| Lineage | Entrant names over time |
|---|---|
| Red Bull junior | Minardi → Toro Rosso → AlphaTauri → RB → Racing Bulls |
| Silverstone | Jordan → Midland → Spyker → Force India → Racing Point → Aston Martin |
| Hinwil | Sauber → BMW Sauber → Sauber → Alfa Romeo → Kick Sauber → Audi (2026) |
| Enstone | Toleman → Benetton → Renault → Lotus → Renault → Alpine |
| Brackley | Tyrrell → BAR → Honda → Brawn → Mercedes |
| Milton Keynes | Stewart → Jaguar → Red Bull |
| New entrants | Haas (2016), Cadillac (2026) |

### Appendix D — points systems (verify every special case in Phase 0)

| Seasons | Race points (P1→P10) | Extras |
|---|---|---|
| 2010–2018 | 25-18-15-12-10-8-6-4-2-1 | **2014 Abu Dhabi: double points** |
| 2019–2024 | Same | **+1 for fastest lap** if finishing in the top 10 |
| 2021 Belgian GP | **Half points** (shortened race) | — |
| 2022+ | Shortened-race scales per the sporting regulations | — |
| 2025+ | Same as above | **Fastest-lap point removed** |
| Sprint 2021 | 3-2-1 | — |
| Sprint 2022+ | 8-7-6-5-4-3-2-1 | — |
| Pre-2010 | Historic tables (1950+) loaded from F1DB/reference data | Used by Layer A only for ordering; points only for reconciliation |

### Appendix E — 2026 regulation notes relevant to modelling (verify against FIA text)

- Power units: MGU-H removed; MGU-K raised to ≈350 kW; nominal ≈50/50 combustion/electric split; energy-limited deployment ("clipping") → bigger gap between one-lap and race pace, and new overtaking dynamics.
- Aero: **active aero** (low-drag straight mode available to all on designated straights); **Overtake Mode** gives extra electrical energy within ~1 s of the car ahead → passing model needs 2026-specific parameters.
- Chassis: shorter, narrower, lighter target; narrower tyres (front −25 mm, rear −30 mm); Pirelli range **C1–C5** (C6 dropped).
- Grid: 11 teams / 22 cars (Cadillac); PU allowances per the sporting regulations (verify), penalties: 10 places for the first excess element, 5 for each further one, >15 → back of the grid.
- 2026 calendar: Bahrain and Saudi Arabian GPs **cancelled** (22 rounds); Madrid added.

### Appendix F — glossary

**PL** Plackett–Luce (probability model for rankings) · **SVI** stochastic variational inference · **NUTS** No-U-Turn Sampler (MCMC) · **PPC** posterior-predictive check · **RPS** Ranked Probability Score · **CRPS** Continuous Ranked Probability Score · **ECE** Expected Calibration Error · **C0–C5** prediction checkpoints · **as_of(t)** time-travel data access that only returns facts knowable at t · **ATR** Aerodynamic Testing Restrictions.

---

## 19. Sources consulted

- OpenF1 — <https://openf1.org/>, <https://openf1.org/auth.html>
- FastF1 on PyPI (v3.8.3, April 2026) — <https://pypi.org/project/fastf1/>; docs — <https://docs.fastf1.dev/>
- Jolpica in FastF1 docs — <https://docs.fastf1.dev/api_reference/jolpica.html>
- F1DB — <https://github.com/f1db/f1db>
- TracingInsights — <https://github.com/TracingInsights/2026>, <https://www.mintlify.com/TracingInsights/RaceData/introduction>
- Open-Meteo Historical Weather API — <https://open-meteo.com/en/docs/historical-weather-api>
- van Kesteren & Bergkamp (2023), *Bayesian analysis of Formula One race results* — <https://arxiv.org/abs/2203.08489>
- FIA Car Presentation Submissions (example) — <https://www.fia.com/system/files/decision-document/2025_british_grand_prix_-_car_presentation_submissions.pdf>
- Motorsport.com on the show-and-tell sessions — <https://www.motorsport.com/all/news/f1-new-show-and-tell-technical-sessions/8310121/>
- 2026 power-unit allocations — <https://www.formula1.com/en/latest/article/how-many-power-unit-components-has-each-driver-used-in-2026-so-far.6DLZYWn0uEpsB0409mFRTF>, <https://racingnews365.com/fia-confirm-future-f1-engine-penalty-rules-after-2026-change>
- 2026 rules overview — <https://www.formula1.com/en/latest/article/everything-you-need-to-know-about-the-new-f1-rules-for-2026.48bv0VTxhIlhrQXmxercXk>
- Pirelli 2026 compounds — <https://www.formula1.com/en/latest/article/pirelli-confirm-2026-tyre-compounds-as-f1-gets-set-for-a-new-era-of.6la0zKVsCYwWk9AAISz4Yw>, <https://press.pirelli.com/the-range-of-compounds-for-the-2026-season-has-been-set/>
- 2026 calendar changes — <https://www.skysports.com/f1/news/13519453/f1-confirms-cancellation-of-bahrain-and-saudi-arabian-grands-prix-due-to-war-in-middle-east-as-2026-calendar-reduced-to-22-races>, <https://www.espn.com/f1/story/_/id/47313408/formula-1-2026-race-calendar-preseason-testing-dates-full-schedule-new-rules>
- Next.js releases — <https://nextjs.org/blog>
- Betfair historic data — <https://github.com/betfair/historic-data-workbook>

Third-party figures (Jolpica rate limits, some 2026 rule numbers, PU allowances beyond 2026) are marked "verify" and are checked against the primary source during the relevant phase.
