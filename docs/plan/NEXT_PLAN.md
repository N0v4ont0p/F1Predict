# Next plan — after the Phase 0–4a merge

Status: **proposal, not started.** Builds on `REBUILD_PLAN.md` and the merged state of `main`:

- `world_v1` (C1 winner log-loss 1.560, C4 1.228);
- weather, lap pace (2018+, without 2022) and the season simulator;
- the web app, which so far only displays results.

Every item below has a goal, a design, its deliverables and an **acceptance gate**. A model change ships only if the rolling-origin backtest says so. Negative results go in `docs/experiments.md`.

## Order of work

| # | Item | Size | Depends on | Why this order |
|---|---|---|---|---|
| 1 | One-command launcher (`./f1p-web.sh`) | S | — | Immediate quality-of-life |
| 2 | M6 sustained benchmarks | S (your machine) | — | Fixes the compute budget before heavier models |
| 3 | 2022 lap pace from Jolpica | S–M | — | Closes a hole in a feature already in production |
| 4 | Championship table "as of" any date | M | — | Mostly reuses the existing simulator |
| 5 | Qualifying prediction (new checkpoint C2) | M | 3 | Also gives C1 a predicted grid |
| 6 | Lap-level model (Phase 4b) | L | 3 | Provides the pace inputs for the simulator |
| 7 | Lap-by-lap race simulator (Phase 6) | L–XL | 6 | The core of the plan's world model |
| 8 | Challenger models + stacking (Phase 7) | L | 5, 6 | Only worth it once there are several good signals |
| 9 | What-if Lab + 3D pages (Phase 12b/13) | L–XL | 4, 7 | Needs the simulator's parameters in the marts |

---

## 1. One-command launcher

**Goal:** `./f1p-web.sh` in a terminal opens the site in your browser on `http://localhost`.

**Design:** a POSIX shell script at the repository root (also exposed as `just web`).

1. **Checks:** `uv`, `node` ≥ 20 and `pnpm` must be installed. If any is missing, print the one-line install command and exit.
2. **Install:** run `pnpm install` in `web/` only if `node_modules` is missing or `pnpm-lock.yaml` has changed (compared via a lockfile hash).
3. **Data:** `--refresh` runs `f1p data update`, then `f1p data pace`, then `f1p model publish`. Without it, the script reuses the existing marts and warns if they are more than 7 days old.
4. **Sync:** `pnpm sync-data` copies the marts into the site.
5. **Mode:**
   - dev (default) runs `pnpm dev`, which hot-reloads;
   - `--prod` runs `pnpm build`, then serves `web/out` with `python3 -m http.server`.
6. **Port:** try 3000. If it is taken, use the next free port, found with `lsof` on macOS or `ss` on Linux.
7. **Open:** wait until the port answers, then run `open` (macOS) or `xdg-open` (Linux) on the URL.
8. **Exit:** Ctrl-C stops the server cleanly (`trap`).

**Tests:**
- `shellcheck` in CI;
- a CI smoke test that starts `--prod` on a free port, `curl`s `/` for a 200, then kills the server.

**Gate:** on a fresh clone, `./f1p-web.sh` reaches a working page with no other command.

## 2. M6 sustained benchmarks

**Goal:** replace the plan's estimated performance targets (§2.2 / §11a.9) with measured, sustained numbers from your Mac mini.

**Code changes (small):**
- **`--sustained` flag for `f1p silicon bench`:** each workload loops for 10 minutes, and the report gives p50/p95 throughput per minute, so throttling would show.
- **New benchmarks:**
  - JAX on the CPU (an MCMC-sized kernel), if installed;
  - a Core ML Neural Engine round-trip using a toy MLX → Core ML model, with the compute plan checked;
  - MLX bf16 matmul;
  - the Accelerate vs OpenBLAS check.
- **Telemetry:** record `powermetrics` (with sudo, optional) and `ProcessInfo.thermalState` (via a tiny Swift helper) alongside the results.

**What you run:**
```bash
git pull && just setup        # installs the `apple` extra (MLX on Metal)
uv run f1p silicon info       # sanity check: clusters, GPU cores, sampler = mlx
just bench --sustained        # ~45 min; results in artifacts/bench/
```
Then send me the JSON file, or commit it to `docs/bench/m6/`.

**Afterwards (me):**
- lock the targets;
- set `F1P_SAMPLER` defaults and batch sizes from the measurements;
- add a CI comparison that flags a regression of more than 15% against the M6 baseline.

**Gate:** every unit has a sustained number, and the Core ML compute plan confirms the toy model runs on the Neural Engine.

## 3. 2022 lap pace from Jolpica

**Problem:** the live-timing archive returns 403 for every 2022 session, so the 2022–23 training windows have a gap.

**Source:** `api.jolpi.ca/ergast/f1/2022/{round}/laps.json` (paginated) plus `/pitstops.json`.

- **Cost:** about 22 races × about 13 pages = about 300 requests.
- **Rate limit:** 4 per second and 500 per hour, so about 40 minutes, cached for good.

**Design:**
- **New connector:** `sources/jolpica.py`, using the existing `HttpClient` limits for `api.jolpi.ca`.
- **Same output table:** the laps feed the same `pace_table()` logic and land in `race_pace`, with a new `source` column (`livetiming` or `jolpica`).
- **Pit laps:** in-laps and out-laps come from the pit-stop list.
- **Neutralised laps:** Jolpica has no safety-car or flag data, so a lap counts as neutralised when the field's median lap time on it exceeds 110% of the race median. That catches SC and VSC laps. Red-flag gaps show up as missing lap times.

**Validation before use:**
- run the Jolpica path on 2019, 2021 and 2023, where live timing exists, and compare pace deficits per car;
- the gate is a correlation of at least 0.95 and a median absolute difference of at most 0.15 percentage points;
- if it passes, also offer Jolpica as a fallback whenever live timing is missing.

**Gate:**
- the validation above passes;
- `race_pace` covers 2018–2026 without gaps;
- the backtest shows no regression (expected neutral to slightly positive for 2023+).

## 4. Projected championship table at any point in time

**Goal:** "what did / does the model think the final standings will be, as of date X or after round N", as a full table.

**Design:**
- **Command:** `f1p model championship [--season S] [--as-of DATE | --after-round N] [--sims 50k]`.
- **Inputs:** `simulate_season()` gains an `as_of` argument. Every input is cut with `known_at(as_of)`, the same gate the backtests use:
  - standings from that round;
  - ratings and pace as of then;
  - remaining rounds forecast at their C1 from that date.
- **Output, the full projected table:**
  - for each driver and team: current points, expected final points, median, 50% and 90% ranges, P(champion), P(top 3), P(each final position);
  - a **most likely final order**, which solves the assignment problem on the position-probability matrix (Hungarian algorithm). This avoids the inconsistency of simply sorting by expected points.
- **Season history:** `f1p model publish` writes `championship_history.json`, one snapshot per completed round (about 1 MB a season).
- **Page:** the Title page gets a round slider, a "title odds over the season" river chart and the full table view.
- **Tie-break:** countback for equal points (wins, then seconds and so on). This needs per-simulation finishing positions, which the simulator already has.

**Calibration backtest (new, honest):**
- for 2010–2025, take a snapshot after every round and score title predictions (log-loss, and the Brier score of P(final position));
- the baseline is "current points order plus a uniform spread";
- the report shows which seasons were called early.

**Gate:**
- historical title log-loss beats the baseline with a significant interval;
- snapshots are reproducible from the stored lake snapshot.

## 5. Qualifying prediction (checkpoint C2)

**Goal:**
- a separate forecast of the qualifying order: pole odds, P(top 10), P(Q1 knockout) and so on;
- made at **C2**: after the last practice session, before qualifying;
- also available at C1 (Thursday).

**Data:**
- F1DB `qualifying` (positions and times, 1950+);
- `practice` (FP positions);
- live-timing FP long-run and short-run pace (2018+, same parser as race pace, using session `Practice 3` or `Sprint Qualifying`).

**Model `quali_v1`:**
- **Rating:** the existing rating filter with qualifying observations (`quali_obs=True`, already built and tested), read at the checkpoint.
- **Features:** rating, recent qualifying-pace deficit (exponentially weighted gap to pole), FP-best-lap deficit (C2 only), weather (wet qualifying is more random).
- **Order:** the combiner plus staged temperatures, fitted on qualifying orders, a separate fit from the race one.
- **Knockout:** derived from the sampled full order using the season's format (Q1 and Q2 cut sizes from `core.regulations.knockout_format`). Dropping out in Q1 is the same as finishing outside the top N.
- **Grid:** penalties from F1DB grid vs qualifying history give a small "grid drop" model. Then **C1 race forecasts can use a predicted grid**, mixed over sampled grids. That is a candidate improvement for C1, tested separately.

**Evaluation:**
- **New target:** qualifying order.
- **Metrics:** pole log-loss, top-10 Brier, Spearman.
- **Baselines:** last qualifying order, rating-only, standings.

**Product:**
- a `quali` block per event in the marts;
- a "Qualifying" tab on event pages with the same tower UI;
- `f1p model forecast --target quali`.

**Gate:** beats the last-qualifying baseline at C1 and C2 with significant intervals.

## 6. Lap-level model (Phase 4b)

**Goal:** model **lap times** instead of only finishing orders. The pace estimates become sharper, and the race simulator gets everything it needs to run.

**Data (extends `race_pace`):**
- a `laps` table: driver, lap, lap_ms, compound, tyre age, stint, in/out lap, track status, position, gap ahead;
- sources: live-timing `TimingAppData` (stints and compounds) and `TrackStatus`; Jolpica for 2022;
- size: about 200k laps a season, stored as Parquet in the lake.

**Model:** a hierarchical Bayesian lap model in NumPyro (JAX on the CPU, the plan's tier 1/2):
```
lap_ms = circuit_base[event] · (1 + car_pace[team,event] + driver[d] + compound[c,event]
         + deg[c,event]·tyre_age + fuel·laps_remaining + traffic·1{gap<1s}) + ε (Student-t)
```
- **Dynamics:** car pace follows a random walk across events (upgrades).
- **Track type:** a car × circuit-type interaction (circuit embedding later).

**Fitting:**
- fitted after each race;
- SVI for speed, with MCMC overnight as a check;
- parameters stored per snapshot, read causally like the rating filter.

**Uses:**
- a pace signal, in seconds per lap with uncertainty, for the combiner (replacing the current exponentially weighted deficit);
- inputs for the race simulator: degradation, compound offsets, pit loss.

**Gate:**
- held-out lap-time RMSE beats the "driver median" baseline;
- the combiner using the new pace signal beats `world_v1` in the 2019–2026 backtest (paired bootstrap).

## 7. Lap-by-lap race simulator (Phase 6)

**Goal:** a generative race. It returns the same `Forecast` as today, so it plugs straight into the harness and the stacking. It also yields lap traces for the Theatre and What-if pages.

**Simulation, per simulated race:**
1. **Start:** position changes on lap 1 come from an empirical start model (grid slot × car pace).
2. **Each lap:**
   - pace from the lap model plus noise;
   - overtaking when the pace difference is above a circuit-specific threshold, with probability from a fitted logistic curve;
   - tyre wear;
   - pit decisions from a strategy policy, at first the stint lengths observed for the circuit and compound;
   - an SC/VSC hazard per circuit;
   - per-lap DNF hazards from the reliability model.
3. **Finish:** classify the cars and score points.

**Implementation:**
1. A vectorised NumPy reference over races × cars × laps.
2. A JAX version for calibration.
3. The MLX/Metal kernel on the M6, gated by bench item 2 (WGSL/Rust later, per Phase 13).

**Calibration:** fit the start, overtaking and SC parameters to historical distributions:
- positions gained;
- SC laps per race;
- the gap between finishing position and grid.
Use simulated-moment matching, or the differentiable twin later.

**Gate:**
- the simulated distributions match history (KS tests);
- as a standalone forecaster it is within the error band of `world_v1`;
- stacked with `world_v1` it beats `world_v1` significantly.

## 8. Challenger models + stacking (Phase 7)

**Challengers**, all fed by one shared, causal feature table:
- **LightGBM** lambdarank;
- **CatBoost** YetiRank;
- **TabICLv2**, in-context learning, CPU/MPS;
- **TabPFN-2.5** only if its licence allows.

**Stacking:**
- **Pooling:** a log-linear pool of the Plackett–Luce strengths.
- **Weights:** fitted by checkpoint and era, on earlier seasons only, with shrinkage towards equal weights.
- **Calibration:** isotonic calibration of P(win) and conformal intervals for points, as a final check.

**Gate:**
- each challenger is reported honestly, including the ones that lose;
- the stack ships only if it beats the best single model with a significant interval.

## 9. What-if Lab + 3D pages

**What-if Lab (`/lab`), runs in the browser:**
- **Marts:** extended with each car's log-strength, stage temperatures, P(finish) and the "why" components. The data is already computed.
- **Sampler:** the staged Plackett–Luce sampler, ported to TypeScript in a Web Worker. 20k races take under a second in JS; WASM comes later with the Rust simulator.
- **Controls:**
  - drag the grid (C4);
  - toggle rain (uses the fitted rain effect on retirements);
  - knock a driver out (P(finish)=0);
  - adjust a car's pace (tenths of a second, converted with the lap model's scale);
  - swap drivers between cars.
- **Instant updates:** the tower, win odds and title odds recompute on every change. The championship impact uses the season simulator's per-race strengths, also exported.
- **Sharing:** the state lives in the URL (nuqs), so a scenario can be shared.
- **Upgrade path:** when the race simulator (7) lands, the Lab switches to the full simulation (server-side first, WASM later).

**3D pages:**
- **Circuit geometry:** from live-timing `Position.z` car x/y traces. Average the clean laps, smooth the result into a closed track polyline, and store one GeoJSON-like file per layout under `web/public/data/circuits/`.
- **Rendering:** three.js with WebGPU and a WebGL2 fallback, via react-three-fiber:
  - the circuit as an extruded ribbon;
  - instanced cars running a sampled simulated race, with the replay driven by lap traces from the simulator (7);
  - until then, an animated "probability field" of cars spaced by expected gap.
- **Theatre page:** replay controls, "roll again", and position traces for 1,000 sampled races (canvas).
- **Budgets:** under 300 KB of JS for 3D (lazy-loaded); `prefers-reduced-motion` respected; a 2D fallback.

**Gate:**
- What-if results match the Python sampler in a cross-language test (same seed: identical distributions within Monte Carlo error);
- Lighthouse performance ≥ 90 on the Hub;
- Playwright visual tests for every new page.

---

## Cross-cutting

- **Automation:** a `just refresh` target, plus an optional launchd plist on the M6, runs after each race weekend: update, pace, publish, build.
- **Ledger:** every forecast published before a race is hashed and appended to `docs/ledger/`. The forward record cannot be edited afterwards.
- **Docs:** README sections and the experiment ledger are updated with every item. The PR description carries the backtest table for each change.
