<div align="center">

# 🏎️ f1predict

**Industry-grade Formula 1 prediction platform** — rich CLI, an extravagant F1-broadcast-themed
dashboard, Monte Carlo simulation, what-if analysis, and rigorous walk-forward backtesting.
Optimised to run comfortably on a **MacBook Air M5 (24 GB)**.

</div>

---

## ✨ What it does

- **Predict** full race results with **P(win) / P(podium) / P(points)** and expected points.
- **Backtest** rigorously with expanding-window, **temporal-safe** walk-forward evaluation across eras.
- **Simulate** races with a vectorised **Monte Carlo** engine (10k–50k sims in well under a second).
- **What-if**: change grid, weather, recent form or car performance and instantly see the impact —
  including **championship swing**.
- Two cohesive surfaces: a **Rich Typer CLI** and a **Streamlit dashboard** that share one analysis core.

Primary model: an **optimised RandomForest**, benchmarked against **LightGBM** and
**HistGradientBoosting**, with memory profiling and a memory-aware Optuna objective.

## 🚀 Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # core install
# optional extras:
pip install -e ".[explain,telemetry,dev]"

# 1) Build data (real jolpica-f1, 2014-2026; cached to Parquet)
f1predict data build --source auto

# 2) Train an accurate model with a preset (test ≈ 5-10 min on an M5 Air)
f1predict train --preset test --experiment-name first_run

# 3) Predict a race by shorthand reference
f1predict predict abudhabi2026      # future race, projected from latest form
f1predict predict suzuka26          # or japan26 / madring26 / monaco26 / vegas26
f1predict predict --upcoming        # most recent / next race

# 4) Launch the dashboard
f1predict dashboard

# …or just run `f1predict` with no arguments to drop into the interactive shell
f1predict
```

> First run needs the network to fetch real F1 data (cached afterwards). Fully offline?
> `f1predict data build --source synthetic` produces a deterministic seeded dataset so the
> whole platform (and the test suite) still runs.

### Interactive shell (no more retyping `f1predict …`)

Run `f1predict` with no arguments (or `f1predict shell`) to enter a **persistent session** —
type bare commands, with history, tab-completion and instant responses:

```text
🏎️  f1predict v…  · active model · ensemble_default
f1predict ❯ predict suzuka26
f1predict ❯ simulate vegas26 --weather rain
f1predict ❯ model                 # list / switch trained models
f1predict ❯ /reload               # drop caches & reload fresh
f1predict ❯ /help                 # meta-commands ·  /exit to leave
```

- **Tab** completes commands, sub-commands, options, race references, presets and model names.
- **↑/↓** recall history (persisted to `~/.f1predict/history`); fish-style auto-suggestions.
- **Warm caches** — the dataset, engineered features and model load once and stay in memory, so
  every command after the first is instant.
- **`/` meta-commands**: `/help`, `/exit` (`/quit`, `/q`, Ctrl-D), `/clear`, `/reload`, `/model`.

### Training presets

Each preset runs **real expanding-window time-series-CV tuning** on the full dataset and
assembles a stacked ensemble — there is no toy shortcut, so predictions are genuinely accurate.

| Preset | Approx time (M5 Air) | Ensemble | Optuna trials |
| --- | --- | --- | --- |
| `test`   | ~5-10 min | RF + LightGBM | 15 |
| `light`  | ~15 min   | + ExtraTrees | 30 |
| `medium` | ~40 min   | + HistGB | 60 |
| `deep`   | ~1.5 hr   | + GradientBoosting | 120 |
| `max`    | ~3-5 hr   | full 5-model ensemble | 250 |

### Race references

`predict` / `simulate` accept forgiving shorthand: a country, city, circuit or GP name plus a
2- or 4-digit year. Examples: `abudhabi2026`, `uae2027`, `suzuka26`, `japan27`, `madring26`,
`spain26` (→ Madrid from 2026), `barca26` (→ Barcelona), `vegas26`, `monaco26`.

## 🖥️ CLI

| Command | Purpose |
| --- | --- |
| `f1predict` *(no args)* / `f1predict shell` | Interactive session — bare commands, history, tab-complete, warm caches |
| `f1predict update [--all] [--offline]` | Refresh every dataset to the latest — results, weather & sprint-aware schedule |
| `f1predict train --preset test\|light\|medium\|deep\|max [--session race\|qualifying\|sprint] [--all-sessions]` | Train an accurate cross-validated ensemble (per session) |
| `f1predict train [--tune --n-trials N] [--compare-all] [--profile-memory]` | Manual train / tune / benchmark |
| `f1predict predict abudhabi2026 [race\|qualifying\|sprint]` / `--upcoming` `[--force]` | Probabilistic race/quali/sprint prediction (refs or `--season/--round`) |
| `f1predict backtest --min-races 2 [--export per_race.csv]` | Walk-forward backtest by era |
| `f1predict simulate suzuka26 --n-simulations 10000 [--weather rain]` | Regulation-native Monte Carlo (DNF, safety cars, era chaos) |
| `f1predict whatif --driver "Max Verstappen" --grid 2 --weather rain --recent-form-boost 0.8` | Scenario impact |
| `f1predict regulations 2026` | Era, key rule changes & the simulation modifiers in effect |
| `f1predict weather suzuka26 [--offline]` | Race-day weather (Open-Meteo) + circuit physics (degradation, abrasion, downforce) |
| `f1predict explain [--top 20]` | Which features drive the active model (offline, no SHAP) |
| `f1predict report [--season --round]` | F1-themed HTML race report |
| `f1predict profile-memory --command train` | Peak RSS + time |
| `f1predict data build [--force-refresh] \| data status` | Build / inspect master dataset |
| `f1predict data weather [--seasons 2018-2026] [--offline]` | Enrich master with cached Open-Meteo weather |
| `f1predict model` | List every trained model & show the active one |
| `f1predict model <name>` | Switch the active model (easy/partial names, e.g. `model ensemble`) |
| `f1predict model delete <name> [-y]` | Delete model(s); auto-promotes the newest survivor if the active one is removed |
| `f1predict model delete clear` (or `all`) | Wipe **every** trained model in one go (clears the active pointer) |
| `f1predict model leaderboard` | Experiment history & metrics |
| `f1predict setup` | How to run the CLI from anywhere |
| `f1predict dashboard` | Launch the Streamlit UI |

Every command renders Rich tables/panels with live progress, and most support
`--format json` for scripting plus `--config` / `-o key=value` overrides.

## 📊 Dashboard pages

1. **Race Predictor** — predicted classification, probability bars, key movers.
2. **Championship & What-If Lab** — standings + live perturbation sliders + championship swing.
3. **Monte Carlo Simulator** — winner/podium bars, points violins, downloadable distributions.
4. **Backtesting Explorer** — accuracy over time, by-circuit & by-era breakdowns.
5. **Explainability** — global feature importance + per-driver "why" contributions.
6. **Upcoming Race Intelligence** — auto-generated storylines and predicted vs grid.
7. **Model Comparison** — family benchmark (MAE vs time vs memory) + experiment leaderboard.
8. **Reports & Export** — generate/preview HTML reports, export CSV.

## 🏗️ Architecture

```
src/
├── config/         YAML config loader (defaults merge + dotted overrides)
├── data_pipeline/  jolpica client, synthetic generator, master builder, schema
├── features/       temporal-safe feature factory (versioned, categorised)
├── models/         registry (RF/LGBM/HistGB), predictor, training, experiment tracker
├── evaluation/     ranking metrics + walk-forward backtester
├── simulation/     vectorised Monte Carlo race & championship engine
├── whatif/         shared scenario engine + analysis context (used by CLI & UI)
├── cli/            Typer + Rich CLI
├── dashboard/      Streamlit app, pages, theme, cached data
├── reporting/      HTML reports, model cards, auto data dictionary & glossary
└── utils/          logging, era-aware points, memory/time profiling
```

See [`docs/DESIGN.md`](docs/DESIGN.md) for design decisions and the flaw-correction log,
and the auto-generated [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md) /
[`docs/FEATURE_GLOSSARY.md`](docs/FEATURE_GLOSSARY.md).

## 🧠 How prediction works

We frame the problem as **regression on finishing position** (lower = better) and rank
drivers within each race. The model is a **stacked ensemble** (RandomForest + ExtraTrees +
LightGBM + HistGradientBoosting + GradientBoosting, blended by a Ridge meta-learner), trained
on a rich, strictly **backward-looking** feature set (~75 features):

- **Causal ELO** for drivers *and* constructors, updated race-by-race with pairwise within-race
  results — the dominant skill/car prior, recorded *before* each race is scored (zero leakage).
- **Qualifying** signals: position, gap-to-pole (ms and %), grid penalties.
- **Form & momentum**: multi-window rolling position/points/finish-rate, EWMA momentum, form
  trend, career & season-to-date aggregates, experience/rookie flags.
- **Constructor** pace, reliability (DNF rate) and momentum.
- **Track intelligence**: driver/constructor circuit affinity plus static track-type clusters
  (street / high-speed / technical) and characteristics (overtaking difficulty, power sensitivity).
- **Weather & environment**: race-day air temperature, humidity, rain probability and wind from the
  free [Open-Meteo](https://open-meteo.com) archive (cached, offline-first), plus researched static
  circuit physics — tyre **degradation**, surface **abrasion**, lap length, downforce demand and
  altitude. A derived `f_tyre_stress` term couples heat × abrasion × degradation, and rain interacts
  with driver skill & grid. Future/offline races fall back to each venue's race-month **climatology**.
- **Teammate H2H**: qualifying & race deltas vs the sister car (car-adjusted skill).
- **2026 regulation awareness**: era one-hots, a reset flag, and a `constructor_trust` weight
  that **shrinks stale prior-car history at every rule reset (2014/2017/2022/2026)** so the model
  leans on driver skill when the pecking order is about to change — instead of blindly assuming
  last season's order holds.

Probabilities come from the model's residual spread — analytically in the predictor (fast) and
via full Monte Carlo in the simulation engine (distributional). Features are strictly
backward-looking (chronological sort + windows that exclude the current race), so backtests are
honest and leakage-free. **Future races** (no grid/quali yet) are projected from the latest ELO
& form snapshot, with the expected grid inferred from combined skill ratings.

## ⚡ Performance & memory (24 GB M5)

- float32 features, CPU-friendly tree models, vectorised NumPy simulation.
- Profiling (`utils/profiling.py`) is integrated into training/backtest/sim; the Optuna
  objective includes a **memory-penalty** term.
- Observed peak RSS for training the full synthetic dataset: **~250–300 MB**.

## 🧪 Testing

```bash
pytest -q
```

Includes **property-based** tests (Hypothesis) for points/championship math, **temporal-safety**
checks on features, **simulation invariants** (one winner/sim, probabilities sum to 1), and
CLI smoke tests.

## 🔌 Extending

- New model family → add a branch in `models/registry.build_model`.
- New data source → emit a frame matching `data_pipeline/schema.RAW_COLUMNS`.
- New feature → extend `features/factory.build_features` (+ glossary entry).
- New page → add a `render()` module under `dashboard/pages/` and register in `app.py`.

## 📜 Data sources

- [jolpica-f1](https://github.com/jolpica/jolpica-f1) (Ergast successor) — 1950+ results backbone.
- [FastF1](https://docs.fastf1.dev/) (optional `telemetry` extra) — 2018+ telemetry/weather.
- [Open-Meteo](https://open-meteo.com) — free, no-key weather archive/forecast for race-day
  conditions (cached locally). *Honest limitation:* daily-resolution venue weather, not
  session-by-session timing — used as a conditions prior, with circuit climatology offline.

## License

MIT.
