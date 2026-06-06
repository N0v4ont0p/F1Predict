# f1predict — Design Notes & Flaw-Correction Log

This document records the deliberate design decisions for f1predict and how this build
addresses weaknesses identified in earlier iterations of the plan.

## Flaws from previous plans → how they are fixed here

1. **Underdeveloped CLI** → A dedicated, fully-implemented Typer + Rich CLI
   (`src/cli/`) with subcommand groups (`data`, `model`), beautiful tables/panels, live
   progress bars, `--config`/`--override`, and `rich`/`json` output modes.
2. **No UI** → A multi-page, F1-broadcast-themed Streamlit dashboard (`src/dashboard/`)
   with 8 pages, dark theme, Plotly charts, cached data access, and session-state what-if
   persistence.
3. **Monte Carlo only "future roadmap"** → First-class vectorised engine
   (`src/simulation/`), 10k–50k sims in <1s, plus CLI command and dedicated UI page.
4. **Light experiment tracking** → Lightweight JSONL tracker (`src/models/experiment.py`)
   logging params, metrics, git commit, data hash, peak RSS and training time; surfaced in
   `model leaderboard` and the Model Comparison page.
5. **Implicit upcoming-race path** → Explicit `upcoming_race` context helper + Upcoming
   Race Intelligence page with auto storylines.
6. **Weak what-if** → A shared `whatif` engine (`src/whatif/`) used by both CLI and UI;
   supports grid, weather, form and car-performance perturbations with delta reporting.
7. **Missing memory profiling** → `src/utils/profiling.py` (`getrusage`-based) integrated
   into training, backtesting, simulation and a `profile-memory` command; Optuna objective
   carries an explicit memory-penalty term.
8. **Insufficient testing** → Property-based tests (Hypothesis) for points/championship
   math, temporal-safety checks on features, simulation invariants, and CLI smoke tests.
9. **Generic docs** → Auto-generated data dictionary + feature glossary
   (`src/reporting/docs.py`) and per-model JSON model cards saved beside each artifact.
10. **Vague extensibility** → Clear seams: `registry.build_model` (add families),
    pluggable data sources behind the canonical schema, and config-driven feature versions.

## Key architectural decisions

- **Regression-to-rank framing.** We predict a continuous finishing position and rank
  within each race. This is robust to variable field sizes and directly yields ranking
  metrics (Spearman, winner/podium accuracy).
- **Temporal safety is enforced in one place** (`features/factory.py`): chronological sort
  + `shift(1)`/expanding windows that exclude the current row. Tested explicitly.
- **jolpica-f1 backbone with synthetic fallback.** Real 1950+ data when online; a seeded
  latent-skill synthetic generator guarantees the platform runs and tests offline.
- **Memory budget (24GB M5).** float32 features, CPU-friendly tree models, vectorised
  simulation, and profiling baked into the training loop. Observed peaks ~250–300MB.
- **Probabilities** come analytically (fast) in the predictor and via full Monte Carlo in
  the simulation engine; both share the model's residual spread for consistency.

## Extension points

- New model family → add a branch to `models/registry.build_model`.
- New data source → produce a frame matching `data_pipeline/schema.RAW_COLUMNS`.
- New feature → add to `features/factory.build_features` (+ glossary entry).
- New dashboard page → add a `render()` module under `dashboard/pages/` and register in
  `dashboard/app.py::PAGES`.

## Regulation-native simulation (upgrade)

F1 races are decided as much by the rule-set as by raw pace, so the Monte-Carlo engine is
**regulation-aware** rather than a one-size-fits-all Gaussian:

- **Rule registry** (`features/regulations.py::_RULES`). Each era carries a curated list of
  key rule changes plus expert-prior modifiers: `chaos` (order variance), `reliability`
  (DNF multiplier), `overtake_difficulty` (0..1 — how much the grid sticks),
  `safety_car_rate`, and `tire_wear_var`. `era_modifiers(season)` amplifies `chaos` and
  `reliability` in and just after a reset year, decaying back to the era baseline via
  `seasons_since_reset` — this is why a 2026 race simulates as genuinely more volatile and
  fragile than a settled 2019 race.
- **Engine dynamics** (`simulation/engine.py::simulate_race`). On top of the base pace draw
  the engine layers: regulation-scaled variance, overtake-difficulty compression (hard-to-
  pass eras shrink position changes), per-driver **retirements** (`empirical_dnf_rates`
  blends a driver's and constructor's recent DNF record, softened by the era reliability
  modifier and capped at a realistic ceiling), and **safety-car bunching** that compresses
  pace gaps in a fraction of races to amplify late upsets. All new behaviour is opt-out and
  backward compatible; `p_dnf` is now reported per driver.
- **CLI surfaces.** `f1predict regulations <season>` renders the era, rule changes and the
  active modifiers; `f1predict simulate` prints the modifiers it applied alongside the
  distribution.

### Explainability without heavy dependencies

`f1predict explain` aggregates impurity-based feature importances across the ensemble's tree
members (RF / ExtraTrees / LightGBM), normalises and groups them into intuitive families
(form, grid/quali, ELO, track, regulations, interactions). This gives an offline, dependency-
light view of *why* the model predicts as it does — including how much weight the regulation-
aware features carry — without pulling in SHAP/XGBoost.

### Calibration metrics

`evaluation/metrics.py` adds proper probabilistic scoring: `brier_score` (P(win) vs. outcome),
`top_n_accuracy`, and `reliability_curve` (predicted vs. observed frequency per bin) for
calibration diagrams.

### Honest data limitation

The jolpica backbone is **results-level** (grid, quali, finishing position, status/DNF,
points) — it carries no per-lap tyre/stint or telemetry data. Physics tyre-degradation curve
fitting is therefore out of scope with the current source; instead we model regulation-driven
tyre-wear *variability* and reliability, which are well-grounded in the available history. A
per-lap source (e.g. FastF1 stints) is the natural extension point for a true tyre model.

## Weather & environment (upgrade)

f1predict now models the *conditions* a race is run in, on two tracks:

**Static circuit physics (researched, no API).** `features/tracks.py::_TRACK_ENV` carries per-circuit
constants for ~37 venues: latitude/longitude, tyre **degradation**, surface **abrasion**, lap length
(km), downforce demand, altitude, and a race-month **climatology** (typical temperature, humidity,
rain probability). These are surfaced as `trk_*` features via `track_stats()` and merged in
`track_profile()`. They are static venue priors — leak-free by construction (no race-outcome info).

**Dynamic weather (Open-Meteo, offline-first).** `data_pipeline/weather.py` wraps the free,
no-API-key Open-Meteo **archive** endpoint for past races (measured/reanalysis daily values) and
caches every `(circuit_id, date)` lookup to `data/cache/weather.parquet`. `get_race_weather()` never
raises and never blocks: anything the API can't cover — future races, offline runs, unknown venues —
falls back to the circuit's climatology. `enrich_weather()` annotates the whole master with the five
`WEATHER_COLUMNS` (`wx_temp_c`, `wx_humidity`, `wx_rain_prob`, `wx_wind_kmh`, `wx_source`), making
exactly one API call per unique race.

**Feature factory.** `features/factory.py` adds nine weather features (`wx_temp_norm`, `wx_humidity`,
`wx_rain_prob`, `wx_wind_norm`, `wx_is_hot`, `wx_is_wet`, `f_tyre_stress`, `wx_rain_x_skill`,
`wx_rain_x_grid`). `f_tyre_stress = trk_degradation · (0.5 + 0.5·wx_temp_norm) · (0.6 + 0.4·trk_abrasion)`
couples heat, abrasion and degradation. Climatology fallback means these are always populated, so the
factory grew 75 → 89 columns **without breaking older models** — `RacePredictor.predict_positions`
selects its own stored `feature_cols`, so a 75-column model still loads and predicts against the
89-column frame.

**Simulation.** `simulation/engine.py` reads `trk_degradation`, `f_tyre_stress` and `wx_rain_prob`
from the race frame (via a robust `_col_mean` helper, defaulting gracefully on older frames) and
modulates the Monte Carlo: rain and high tyre-stress widen the finishing-position σ (more shuffling),
lift the safety-car rate, and raise DNF probability. A `--weather rain` override pins rain high. The
`SimulationResult.meta` now carries `rain_prob`, `tyre_stress` and `track_degradation` for the CLI to
display.

**CLI.** `f1predict weather <race> [--offline]` prints race-day weather (measured or climatology) plus
the track-physics card; `f1predict data weather [--seasons A-B] [--offline]` bulk-enriches and caches
the master; `simulate` surfaces the active rain / tyre-stress / degradation line.

### Honest weather limitation

Open-Meteo archive data is **daily-resolution venue weather**, not session-by-session timing-screen
weather. We use it as a conditions *prior* (was it a hot/wet weekend at this circuit?), not as a
minute-by-minute race feed. Climatology fallback keeps everything fully offline and deterministic for
tests. A session-resolution feed (e.g. FastF1 weather laps for 2018+) is the natural extension point.

## Interactive shell & warm-cache architecture (upgrade)

A traditional `argv`-per-invocation CLI forces the user to retype `f1predict …` and re-pay every
cold-start cost (parquet read, feature engineering, joblib model load) on *each* command. Tools that
feel "smooth" (Claude Code, IPython, aws-shell) are instead a **persistent REPL** with history,
completion and warm in-memory state. f1predict now ships that as a first-class surface.

**REPL (`cli/shell.py`).** Built on `prompt_toolkit`. Running `f1predict` with no args — or
`f1predict shell` — opens a session that reads a line, splits it with `shlex`, and dispatches into the
**existing** Typer app compiled to a Click command (`typer.main.get_command(app)`) with
`standalone_mode=False`. That flag stops Click from calling `sys.exit`, so a finished command, a bad
flag (`UsageError`), or `--help` (`SystemExit`) is caught and the loop continues — one bad command
never kills the session. There is zero duplication of command logic: the shell is a thin driver over
the same commands the one-shot CLI exposes.

**No-args routing.** A root `@app.callback(invoke_without_command=True)` launches the shell **only**
when both stdin and stdout are TTYs; pipes, CI and the pytest `CliRunner` get help text instead, so
nothing ever hangs headless.

**Context-aware completion.** The Click command tree is introspected once into
`{command: {opts, subs}}`. The completer is positional: first token → commands + `/meta`; `-` prefix →
that command's options (Typer's `TyperOption` isn't a `click.Option` subclass, so options are detected
by dash-prefixed `opts` strings rather than `isinstance`); `--preset` → preset values; `model`/`data`
→ their sub-commands (plus live model names for `model <name>`); `predict|simulate|whatif|weather` →
race-reference venue tokens from `racref._ALIASES`.

**Warm caches (the real smoothness win).** Three chokepoints became mtime/identity-keyed caches:
`data_pipeline.build.load_master` (keyed by path + mtime, returns a stable object), `whatif.context.
get_feature_frame` (keyed by that object's identity + feature version), and `models.train.
load_production` (keyed by the resolved model file + mtime). The first `predict` pays ~1–2 s of
feature engineering and the model load **once**; every later command in the session is eff-instant
(measured: feature build 1.2 s → 0 s, model load 0.6 s → 0 s). `clear_caches` / `clear_feature_cache`
/ `clear_model_cache` back the shell's `/reload`. The caches are invalidated automatically when the
underlying files change (e.g. after `train` or `data build`), so correctness is preserved for one-shot
use and tests alike — switching the production pointer changes `production.txt`, which repoints to a
file with a different mtime, busting the model cache.

**Meta-commands** use a leading `/` (`/help`, `/exit`, `/clear`, `/reload`, `/model`) so they can
never collide with real commands, with `exit`/`quit`/Ctrl-D as ergonomic aliases.


