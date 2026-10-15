# f1predict

Probabilistic Formula 1 forecasting, rebuilt from the ground up. It is built around a generative
world model of the race weekend, fitted to every lap rather than just race results, and tuned to
use every compute unit of an Apple M6.

> **Status: rebuild in progress — Phase 3 (world model v1) landed.** The full blueprint is
> [`docs/plan/REBUILD_PLAN.md`](docs/plan/REBUILD_PLAN.md). The previous implementation is
> preserved in git history at commit `95d8892` (see [`docs/legacy/LEGACY_SNAPSHOT.md`](docs/legacy/LEGACY_SNAPSHOT.md)).

## What exists today

| Module | Purpose |
|---|---|
| `f1p.core.points` | Exact championship points from 1991 to today. Covers sprints, the 2019–2024 fastest-lap bonus, pre-2022 half points, the 2022+ shortened-race scales and 2014 double points |
| `f1p.core.status` | Result-status taxonomy (fixes the legacy "Lapped = DNF" bug) and the 90%-distance classification rule |
| `f1p.core.regulations` | Regulation eras and resets (used as data the model learns from), knockout-qualifying formats (incl. 22-car 2026) |
| `f1p.core.clock` | `AsOf` / `known_at`: the single leakage gate, filtering every row by when it became public |
| `f1p.core.ids` | Canonical ids, `EventKey` (`2026-R05`), `SessionKey` (`2026-R05/quali`) |
| `f1p.core.config` | Typed settings (`F1P_*` env vars), including the 24 GB unified-memory budget |
| `f1p.sources` | Polite HTTP (per-host rate limits, `Retry-After`, immutable raw cache) and the F1DB release connector |
| `f1p.lake` | DuckLake lake: 18 curated tables (1950 → today) with `observed_at`, snapshots and time travel, plus six data-quality gates |
| `f1p.eval` | Rolling-origin harness: contexts cut at each checkpoint's `as_of`, proper scores, calibration, paired bootstrap, baselines |
| `f1p.models` | World model v1: Gaussian-filtered Plackett–Luce ratings, a cause-specific reliability model, and a linear combiner with stage-wise temperatures |
| `f1p.silicon` | Apple-silicon runtime: CPU clusters / GPU / Neural Engine discovery, QoS lanes, unified-memory ledger, MLX race sampler |
| `f1p` CLI | `f1p version`, `f1p doctor`, `f1p points …`, `f1p data update / weather / …`, `f1p eval run / models`, `f1p model forecast / ratings / publish`, `f1p silicon info / bench` |

### The lake today

`f1p data update` downloads the latest [F1DB](https://github.com/f1db/f1db) release and curates
it into 228k rows: events, sessions, entries, results, qualifying, grids, practice, pit stops,
fastest laps, standings and team lineage. It then runs the gates and commits one DuckLake
snapshot. The gates verified against all of F1 history:

| Gate | What it proves |
|---|---|
| `points_reconciliation` | Recomputes **every official race and sprint score since 1991** (15,262 rows) and gets every one exactly right |
| `standings_reconciliation` | Every round's championship table equals the cumulative points (13,505 rows) |
| `classification_integrity` | Positions are contiguous; vacancies only where a disqualification left one |
| `status_taxonomy` | All 198 historical retirement reasons map to a status class |
| `results_coverage` | Every past event has a classification |
| `knowability_order` | Qualifying is public before the race starts, and race results before the next event |

Source errors are corrected and reported, never silently: 12 zero-lap "classified" retirements,
and duplicated pit stops at the 2026 Italian GP (round 13).

## World model v1

Each car's log-strength is a **driver** skill plus a **team-lineage** performance. Both are
Gaussian beliefs that drift between races and across seasons, with a learned extra drift in
regulation-reset years. After each race they are updated by one Newton (Laplace/EKF) step on
the exact Plackett–Luce likelihood of the classified order. Retirements are censored rather
than ranked last, and are forecast separately: mechanical failures are charged to the car,
incidents to the driver, everything else to a field rate.

At **C1** (Thursday before the race) the forecast uses the ratings. At **C4** (just before
the race) it also uses grid slot and qualifying order. A linear combiner turns these into
log-strengths. The finishing order is then drawn place by place, with a **temperature that
falls smoothly down the field**: the winner is far more predictable than one Plackett–Luce
temperature allows, and fitting that curve was the largest single gain. The combiner, the
temperature curve and the P(finish) calibration are refitted each season on earlier seasons
only, and every input is filtered to what was public at the checkpoint.

Rolling-origin backtest, 2014–2026 (269 races; filter hyperparameters fitted on races ≤ 2013 only):

| Checkpoint | Model | Winner log-loss | Podium Brier | Points Brier | Finisher log-lik / car | ECE top-3 |
|---|---|---|---|---|---|---|
| C1 | **world_v1** | **1.560** | **0.078** | **0.171** | **−1.601** | **0.018** |
| C1 | world_v1 without lap pace | 1.584 | 0.079 | 0.172 | −1.606 | 0.023 |
| C1 | world_v1, one temperature | 1.815 | 0.082 | 0.171 | −1.672 | 0.049 |
| C1 | standings baseline | 2.069 | 0.089 | 0.175 | −1.721 | 0.071 |
| C4 | **world_v1** | **1.228** | **0.064** | 0.158 | **−1.520** | **0.018** |
| C4 | world_v1 without lap pace | 1.240 | 0.065 | 0.158 | −1.522 | 0.020 |
| C4 | world_v1, one temperature | 1.386 | 0.067 | **0.156** | −1.582 | 0.039 |
| C4 | qualifying baseline | 1.930 | 0.082 | 0.163 | −1.658 | 0.079 |
| — | uniform | 3.015 | 0.126 | 0.249 | −1.979 | — |

**Lap pace.** For every race since 2018, clean laps are extracted from the F1 live-timing
archive. Lap 1, in-laps, out-laps, safety-car, VSC and red-flag laps, and laps more than 7%
off the car's own median are all excluded. Each car's median then gives its **pace deficit**
to the fastest car, and a recency-weighted team pace plus the driver's delta to their
teammate becomes a model input. This measures speed without the luck in finishing orders.
Over 2019–2026 it significantly improves winner log-loss: −0.038 at C1 and −0.018 at C4,
with 95% intervals excluding 0. The archive does not serve 2022, so that season comes from
Jolpica lap times and pit stops instead (safety-car laps inferred from the whole field
running ≥10% slow); on races both sources cover, the two pace tables agree to r = 0.9999.

**Weather.** Race weather comes from Open-Meteo:

- **C1:** rain climatology for the date, computed from the 30 previous years.
- **C4:** the archived race-window forecast, which is sharp: races forecast at P(wet)=1 were
  wet 87% of the time.

Rain raises the modelled retirement rate. Letting it also flatten the finishing order was
unstable across seasons and did not improve the backtest, so that option ships as an
ablation (`world_v1_wet_temp`), not the default.

Every improvement of world_v1 over the baselines is significant: the paired-bootstrap 95%
interval excludes zero. Against its own one-temperature ablation it trades a little top-10
sharpness at C4 (points Brier +0.0015) for much better winner and podium forecasts.

```bash
uv run f1p model forecast                         # next race, pre-weekend (C1)
uv run f1p model forecast 2026-R17 --checkpoint C4 --json
uv run f1p model ratings
uv run f1p eval run --models world_v1,quali,standings
```

## Web app

`web/` is a statically exported Next.js 16 app (React 19 with the React Compiler, Tailwind v4,
Motion). It renders the JSON marts written by `f1p model publish` in a "pit wall at night"
design:

- **Hub:** the next race with a live "probability river" hero and the probability timing
  tower. Each car's full finishing-position distribution is shown as a heat strip, and the
  rows animate between C1 and C4.
- **Event pages:** the same views for each race, with the result and a post-race score card.
- **Season:** a table of every round's forecast against what actually happened.
- **Ratings:** car and driver beliefs with uncertainty bands.
- **Model Lab:** the leaderboard, reliability diagrams and the stage-temperature curve.

The quickest way in is one command, which installs what's missing and opens your browser:

```bash
./f1p-web.sh              # dev server on the first free port from 3000 (or: just web)
./f1p-web.sh --refresh    # update data + race pace + forecasts first (needs uv)
./f1p-web.sh --prod       # build the static site and serve it
```

Or step by step:

```bash
uv run f1p model publish           # artifacts/marts (≈5 min with the leaderboard)
cd web && pnpm install && pnpm sync-data && pnpm dev
pnpm build                         # static site in web/out
```

## Development

Requires [uv](https://docs.astral.sh/uv/) (it installs Python 3.13 for you) and optionally
[just](https://just.systems).

```bash
just setup        # uv sync --all-groups
just ci           # format check, lint, types, module boundaries, tests
uv run f1p doctor
uv run f1p points 2021 --distance 0.03   # 2021 Belgian GP: half points
uv run f1p data update                   # build the lake (≈30 s)
uv run f1p data show results --event 2026-R17 --session race
uv run f1p data weather --from 2000             # race weather (≈20 min, cached)
uv run f1p data pace --from 2018                # clean-lap race pace (≈20 min, cached)
```

## Roadmap

| Phase | Scope |
|---|---|
| 0 | Reset and foundation ✅ |
| 0.5 | M6 silicon runtime ✅: device discovery, QoS lanes, memory ledger, MLX GPU sampler, `f1p silicon bench` (sustained M6 baselines to record) |
| 1 | Data platform: F1DB → DuckLake ✅; Open-Meteo weather ✅; live-timing lap pace 2018+ ✅ (`f1p data pace`) |
| 2 | Evaluation harness and baselines ✅ |
| 3 | World model v1 (results era): ratings, reliability, combiner, live forecasts ✅ |
| 4–7 | Lap-level world model, qualifying, simulator, challengers, stacking |
| 11 | Publish: JSON marts (`f1p model publish`) ✅ |
| 12 | Web app: Hub, event pages, season, ratings, Model Lab ✅ (first release) |
| 8–15 | Documents, amortized inference, F1-PFN, API, 3D/What-if, Rust/GPU, live |

## Licence

MIT. Data sources carry their own terms (see plan §17); F1DB data is CC BY 4.0 — *Data from F1DB (https://github.com/f1db/f1db)*. Not affiliated with Formula 1, the FIA or
any team.
