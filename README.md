# f1predict

Probabilistic Formula 1 forecasting, rebuilt from the ground up. It is built around a generative
world model of the race weekend, fitted to every lap rather than just race results, and tuned to
use every compute unit of an Apple M6.

> **Status: rebuild in progress — world model v1 with a lap-level pace model, qualifying and
> championship projections, a lap-by-lap race simulator, challengers and stacking, the What-if
> Lab and 3D race theatre.** The full blueprint is
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
| `f1p.lake` | DuckLake lake: 18 curated tables (1950 → today) with `observed_at`, snapshots and time travel, six data-quality gates, plus race pace, every race lap (2018+) and circuit geometry |
| `f1p.eval` | Rolling-origin harness: contexts cut at each checkpoint's `as_of`, proper scores, calibration, paired bootstrap, baselines |
| `f1p.models` | World model v1 (Gaussian-filtered Plackett–Luce ratings, cause-specific reliability, a linear combiner with stage-wise temperatures), the lap-level pace model, `quali_v1`, the season simulator with as-of projections, the lap-by-lap race simulator, LightGBM/CatBoost challengers and stacking |
| `f1p.silicon` | Apple-silicon runtime: CPU clusters / GPU / Neural Engine discovery, QoS lanes, unified-memory ledger, MLX race sampler |
| `f1p` CLI | `f1p version`, `f1p doctor`, `f1p points …`, `f1p data update / weather / pace / laps / circuits / …`, `f1p eval run / title / pace / stack / models`, `f1p model forecast / championship / ratings / publish / ledger`, `f1p silicon info / bench` |

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
| C1 | **world_v1** | **1.537** | **0.077** | **0.170** | **−1.597** | **0.018** |
| C1 | world_v1 with the earlier EWMA pace | 1.556 | 0.078 | 0.171 | −1.599 | 0.018 |
| C1 | world_v1 without lap pace | 1.584 | 0.079 | 0.172 | −1.606 | 0.023 |
| C1 | world_v1, one temperature | 1.771 | 0.080 | 0.170 | −1.668 | 0.046 |
| C1 | standings baseline | 2.069 | 0.089 | 0.175 | −1.721 | 0.071 |
| C4 | **world_v1** | **1.229** | **0.064** | 0.158 | −1.520 | **0.019** |
| C4 | world_v1 with the earlier EWMA pace | 1.231 | 0.065 | 0.158 | **−1.520** | 0.019 |
| C4 | world_v1 without lap pace | 1.240 | 0.065 | 0.158 | −1.522 | 0.020 |
| C4 | world_v1, one temperature | 1.377 | 0.066 | **0.156** | −1.582 | 0.037 |
| C4 | qualifying baseline | 1.930 | 0.082 | 0.163 | −1.658 | 0.079 |
| — | uniform | 3.015 | 0.126 | 0.249 | −1.979 | — |

**Lap pace.** Every Grand Prix lap since 2018 is in the lake (`laps`, 205k rows): lap time,
running position, interval to the car ahead, stint, compound and tyre age, pit and
safety-car flags. Live timing supplies the laps; its tyre feed is withheld for most
seasons, so compounds come from OpenF1 for 2023 onwards. 2022 and a few races the archive
does not serve come from Jolpica (on races both cover, the two agree to r = 0.9999).

The **lap model** explains every clean lap of the last 24 races public at the checkpoint:

```
log lap = event base + car[team, event] + driver + compound[c, event] + degradation[c, event]·tyre age
        + fuel·laps left + traffic·1{< 1 s behind} + Student-t noise
```

Car pace follows a random walk over each team's races (a bigger step across seasons), and
the per-event effects are shrunk towards global ones. It is solved exactly (an IRLS MAP with
its Laplace posterior) in about 0.25 s per window, so it is refitted causally before every
race. Each car's expected pace at the next race is the `lap_pace` input of `world_v1`.

- Held out (2019–2026, 168 races), it predicts each car's race-pace deficit better than the
  earlier exponentially weighted clean-lap deficit: RMSE 0.645 vs 0.685 pp (Δ −0.040,
  95% CI −0.056 to −0.023). A driver's recent median does worse (0.715).
- In the combiner it improves C1 winner log-loss by 0.029 over 2019–2026 (CI −0.055 to
  −0.006) and by 0.019 over the whole 2014–2026 window (CI −0.036 to −0.005), with a better
  podium Brier score (significant); C4 is unchanged. It replaced the EWMA signal.

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
uv run f1p model championship                     # projected final table, now
uv run f1p model championship --after-round 10    # ... as the model saw it after round 10
uv run f1p model championship --season 2021 --as-of 2021-09-01 --json
uv run f1p eval title --from 2010 --to 2025       # backtest of title projections
uv run f1p model forecast --target quali --checkpoint C2   # qualifying, after practice
uv run f1p eval run --target quali --models quali_v1,last_quali,practice
```

**Qualifying.** `quali_v1` forecasts the qualifying order at **C1** (Thursday) and **C2**
(after the last practice session, just before qualifying): pole, front row, P(Q3) and
P(knocked out in Q1/Q2) under the season's knockout format. It is the race model's machinery
fitted on qualifying orders, with the rating filter that also assimilates qualifying,
recent qualifying pace (best-lap gap to pole), this weekend's practice pace and, where it
comes first, the sprint-qualifying order. Its baselines are the last qualifying order, the
championship order and the practice order.

| 2014–2026, 269 sessions | Pole log-loss | Podium Brier | Order log-lik / car |
|---|---|---|---|
| **quali_v1, C2** (after practice) | **1.461** | **0.060** | **−1.580** |
| quali_v1, C1 (Thursday) | 1.558 | 0.066 | −1.662 |
| practice order, C2 | 1.960 | 0.079 | −1.744 |
| championship order | 2.002 | 0.083 | −1.814 |
| last qualifying order | 2.166 | 0.091 | −1.870 |

Against the last qualifying order the gain is −0.61 (C1) and −0.70 (C2) in pole log-loss,
both with 95% intervals far from 0; practice pace and the sprint-qualifying order are what
C2 adds (−0.10 over the same model without them).

**Championship projections.** `f1p model championship` gives the full projected final table:
points now, expected and median final points, 50% and 90% ranges, P(title), P(top 3) and
every final-position probability, for drivers and constructors. Rows are ordered by the
**jointly most likely final order** (an assignment over the position probabilities, solved with
the Hungarian algorithm), and equal points are split on countback. With `--after-round N` or
`--as-of DATE`, every input (results, standings, ratings, race pace, line-up) is cut at that
moment, exactly as in the backtests. `f1p eval title` replays that after every round of every
season and scores it against "current points plus a uniform spread", bootstrapping whole
seasons. Over 2010–2025 (after every round but the last):

| Title projections | world_v1 | uniform spread | Δ (95% CI) |
|---|---|---|---|
| Drivers' title log-loss | **0.605** | 0.921 | −0.317 (−0.469, −0.151) |
| Drivers' final-position Brier | **0.721** | 0.792 | −0.071 (−0.084, −0.059) |
| Constructors' title log-loss | **0.461** | 0.577 | −0.116 (−0.270, +0.114) |
| Constructors' final-position Brier | **0.492** | 0.597 | −0.105 (−0.133, −0.076) |

The constructors' title gain is not significant over 16 seasons. The drivers' title was
called for good (P ≥ 50% from then on) before round 2 in 2015, 2019, 2020, 2023 and 2024,
and never before the finale in 2010 and 2021.

## Web app

`web/` is a statically exported Next.js 16 app (React 19 with the React Compiler, Tailwind v4,
Motion). It renders the JSON marts written by `f1p model publish` in a "pit wall at night"
design:

- **Hub:** the next race with a live "probability river" hero and the probability timing
  tower. Each car's full finishing-position distribution is shown as a heat strip, and the
  rows animate between C1 and C4.
- **Event pages:** the same views for each race, with the result and a post-race score card;
  the tower switches between the race and the qualifying forecast.
- **Season:** a table of every round's forecast against what actually happened.
- **Title:** title odds with final-points fans, plus a round slider over the projected final
  table after every round, a "title odds over the season" river and the full table.
- **What-if Lab:** drag the grid, add tenths, make it rain or take a car out; the forecast
  re-runs in the browser with the model's own sampler (Web Worker), and the scenario lives in
  the URL.
- **Race theatre:** simulated races replayed on the real circuit in 3D (three.js, loaded on
  demand), with the running order, safety cars and lap-by-lap position traces.
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
uv run f1p data laps --from 2018                # every race lap with tyres (≈10 min, cached)
uv run f1p data circuits                        # circuit geometry for the 3D pages
just refresh                                    # after a race weekend: data → publish → site
```

## Roadmap

| Phase | Scope |
|---|---|
| 0 | Reset and foundation ✅ |
| 0.5 | M6 silicon runtime ✅: device discovery, QoS lanes, memory ledger, MLX GPU sampler, `f1p silicon bench` (sustained M6 baselines to record) |
| 1 | Data platform: F1DB → DuckLake ✅; Open-Meteo weather ✅; live-timing lap pace 2018+ ✅ (`f1p data pace`) |
| 2 | Evaluation harness and baselines ✅ |
| 3 | World model v1 (results era): ratings, reliability, combiner, live forecasts ✅ |
| 4–7 | Lap-level pace model ✅, qualifying (C2) ✅, championship as-of ✅, lap-by-lap simulator ✅, challengers + stacking ✅ |
| 11 | Publish: JSON marts (`f1p model publish`) ✅ |
| 12 | Web app: Hub, event pages, season, ratings, Model Lab ✅ (first release) |
| 9 | What-if Lab ✅, 3D race theatre ✅, forecast ledger ✅ |
| 8–15 | Documents, amortized inference, F1-PFN, API, Rust/GPU simulator, live |

## Licence

MIT. Data sources carry their own terms (see plan §17); F1DB data is CC BY 4.0 — *Data from F1DB (https://github.com/f1db/f1db)*. Not affiliated with Formula 1, the FIA or
any team.
