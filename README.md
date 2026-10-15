# f1predict

Probabilistic Formula 1 forecasting, rebuilt from the ground up. It is built around a generative
world model of the race weekend, fitted to every lap rather than just race results, and tuned to
use every compute unit of an Apple M6.

> **Status: rebuild in progress — Phase 1 (data platform).** The full blueprint is
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
| `f1p` CLI | `f1p version`, `f1p doctor`, `f1p points …`, `f1p data update / status / check / show` |

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
```

## Roadmap

| Phase | Scope |
|---|---|
| 0 | Reset and foundation ✅ |
| 0.5 | M6 silicon runtime: scheduler, memory budget, per-unit benchmarks |
| 1 | Data platform: F1DB → DuckLake ✅; weather, laps (Jolpica/FastF1) in progress |
| 2 | Evaluation harness and baselines |
| 3–7 | World model, qualifying, simulator, challengers, stacking |
| 8–15 | Documents, amortized inference, F1-PFN, API, web, Rust/GPU, live |

## Licence

MIT. Data sources carry their own terms (see plan §17); F1DB data is CC BY 4.0 — *Data from F1DB (https://github.com/f1db/f1db)*. Not affiliated with Formula 1, the FIA or
any team.
