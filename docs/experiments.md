# Experiment ledger — world model v1

Every idea tried on the results-era model, with its backtest outcome, kept or not. All numbers
come from the same rolling-origin protocol:

- **Window:** 2014–2026, 269 races.
- **Hyperparameters:** filter hyperparameters fitted only on races up to 2013.
- **Refitting:** the combiner and calibrations are refitted each season on the 8 previous
  seasons.
- **Knowability:** every input is cut at the checkpoint's `as_of`.
- **Metrics:** "winner" is winner log-loss (lower is better); "order" is the finisher
  Plackett–Luce log-likelihood per car (higher is better).
- **Significance:** paired bootstrap over events, 95% interval.

## Kept (the default `world_v1`)

| Change | C1 winner | C4 winner | Notes |
|---|---|---|---|
| Baselines: standings (C1), qualifying order (C4) | 2.069 | 1.930 | Rank Plackett–Luce with β fitted causally |
| Gaussian-filtered Plackett–Luce ratings (default hyperparameters) + combiner | 1.882 | 1.394 | Laplace/EKF update; retirements censored |
| + cause-specific retirement model, calibrated with pace signals | 1.880 | 1.386 | Generative order with retirements; Rao-Blackwellised P(win) |
| + bounded prequential hyperparameter fit (≤ 2013) | 1.815 | 1.386 | 2014+ prequential log-lik −8005 → −7795 |
| + **stage-wise temperatures** (smooth 2-parameter curve) | **1.584** | **1.239** | Biggest single gain; ECE top-3 0.049 → 0.022 / 0.039 → 0.019 |
| + rain → retirement rate only | 1.584 | 1.239 | Neutral to slightly positive at C4 |
| + **lap-level race pace** (clean-lap deficit, team EWMA + driver vs teammate) | **1.560** | **1.228** | 2019–2026: −0.038 (C1) / −0.018 (C4), both significant; decay 0.6–0.9 makes little difference (0.8 kept, not tuned on the test window) |

| 2022 lap pace filled from Jolpica (r = 0.9999 vs live timing where both exist) | 1.555 | 1.230 | 2014–2026 rerun: C1 winner −0.005, order +0.002; C4 within noise — no regression |
| **Lap-level pace model** replaces the EWMA deficit (`lap_pace`) | see note | see note | 2019–2026 (169 races) vs the EWMA version: C1 winner −0.029 (−0.055, −0.006), podium Brier −0.0011 (sig.); C4 −0.002 (n.s.). Held-out pace RMSE 0.645 vs 0.685 pp |

### Lap model tuning (held-out pace, 2024 only, before the 2019–2026 check)

| Random-walk step / race | Window | RMSE (pp) | Spearman |
|---|---|---|---|
| 0.006 | 16 | 0.552 | 0.754 |
| 0.003 | 16 | 0.549 | 0.756 |
| 0.0015 | 16 | 0.540 | 0.772 |
| 0.0006 | 24 | 0.521 | 0.783 |
| **0.0004** (kept) | **24** | — | — |
| 0.0003 | 24 | 0.511 | 0.789 |
| EWMA deficit (reference) | 12 | 0.513 | 0.793 |

Car pace moves less from race to race than the per-race noise suggests; the driver prior scale
barely matters (0.002–0.008). Tuned on 2024 only, then confirmed on 2019–2026.

## Qualifying model (`quali_v1`), 2014–2026

| Variant | C1 pole log-loss | C2 pole log-loss | C2 order log-lik |
|---|---|---|---|
| Rating with qualifying assimilated only (`quali_v1_rating`) | 1.584 | 1.588 | −1.683 |
| + race rating + EW qualifying pace (`quali_v1_noprac`) | 1.558 | 1.559 | −1.660 |
| + practice pace + sprint-qualifying order at C2 (**`quali_v1`**) | 1.558 | **1.461** | **−1.580** |
| Last qualifying order (baseline) | 2.166 | 2.166 | −1.870 |

## Tried and not adopted

| Idea | Result | Why it was not adopted |
|---|---|---|
| Unbounded Weng–Lin hyperparameter fit | Collapsed to degenerate values (drift ≈ 10⁶, reset multiplier ≈ 0) | The global `c` normaliser in Weng–Lin can be exploited, which motivated the scale-free Laplace update and bounds |
| Fitted debutant prior mean | Same likelihood at −3 and +1 | Unidentifiable: a gauge freedom. Now fixed at 0, with a test proving the invariance |
| Qualifying + sprint sessions assimilated into the filter (`world_v1_q`) | Order −1.606 → −1.597 (better); winner 1.584 → 1.614 (worse) | Trades the headline metric for full-order sharpness. Refitting hyperparameters with qualifying included overfit pre-2014 (2014+ −7675 → −7706) |
| Both ratings as combiner features (`world_v1_2r`) | Order better, winner 1.584 → 1.595 | Same trade-off |
| Qualifying pace (% gap to pole) at C4 (`world_v1_pace`) | Order and points Brier slightly better, winner 1.386 → 1.454 (before stage temperatures) | Heavy tails (wet qualifying) give over-confident favourites |
| Free per-place temperatures (2, 4, 8, 12, 19 places) | 19 free ≈ smooth curve (1.577 / 1.242) | The smooth 2-parameter curve matches it with far fewer parameters |
| Rain flattens the order (`world_v1_wet_temp`) | 1.586 / 1.244 (slightly worse) | Fitted effect unstable across seasons, even changing sign. Wet races are only about 10% of the sample, and before 2016 the C4 signal is just climatology |
| Circuit affinity: team and driver gain vs season form at this track (`world_v1_aff`) | 1.587 / 1.242 | Finishing-position residuals at a circuit are mostly noise |
| Race-time margins as Gaussian observations in the filter | Worse at every scale/noise tested, e.g. 2014+ prequential −6.6 at best | Log-strength is not linear in time gaps, and the order update already captures most of the information |

## Weather data quality

Archived race-window forecasts (2016+) are sharp: races forecast at P(wet) = 1 were wet 87%
of the time. Races forecast below 5% were wet 2.9% of the time. Climatology is weakly
informative: dates with ≥ 50% historical rain days were wet 26% of the time, against a 10%
base rate.

## Next candidates

- A race-forecast checkpoint at C2 using practice long-run pace (the qualifying C2 shows
  practice is informative).
- Grid-penalty model so C1 race forecasts can mix over predicted grids.
- A driver × wet-skill interaction (rain specialists), now that observed wet labels exist.
- A position-dependent combiner, so grid information decays down the order.
