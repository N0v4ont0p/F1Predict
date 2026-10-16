# Plan — C1 calibration (CAL) and a data-driven reset response (ADR)

Two follow-ups from the reset study (`RESET_PLAN.md`, build log "Regulation resets in the world
model"). Each has hypotheses, a protocol and a decision rule fixed **before** any result is
seen. The holdout (2024–2026) is not touched by either.

---

## CAL — the world model's C1 probabilities

### Evidence

- World_v1's C1 P(win) is **under-confident** in every group of 2014–2023: the temperature
  that best fits the winners is τ = 1.24–1.34, where τ > 1 means "sharper is better".
- The full order's τ is ≈ 0.93.
- C4 is near calibrated (τ ≈ 1.05–1.14).
- These τ were fitted on the evaluation races themselves (in-sample), so they show the
  direction, not a usable value.

**Likely cause.** The C1 combiner's weights are shrunk towards their priors (`l2`), and its
first-place temperature comes from a smooth curve fitted over all places. Both damp the
winner's probability.

### Hypotheses (pre-registered; no others)

| | Change | Fitted on |
|---|---|---|
| **K1** | P(win)^τ, renormalised (`temperature`) | the winners of world_v1's own **out-of-sample** C1 forecasts of the 4 seasons before |
| **K2** | log-strength × τ (`temperature_all`): the whole order is sharpened or flattened | the finishers' Plackett-Luce likelihood of the same forecasts |

- **Bounds and shrinkage:** τ in [0.5, 2], with the penalty of `stacking.TEMPERATURE_L2`, as
  in A1.
- **Causality:** every training forecast was made by the rolling harness before its race, and
  τ for season S uses seasons S−4..S−1 only. So 2014–2023 are all out-of-sample evaluations.
  (This is unlike A1, which had to cross-fit inside the training window.)
- **Checkpoints:** C1 and C4 are fitted and judged separately. The evidence is about C1; C4 is
  included to check it does no harm.

### Decision rule

Over 2014–2023 (204 races per checkpoint; paired bootstrap against world_v1), adopt a variant
at a checkpoint only if **all** of these hold:

1. **Primary:** winner log-loss improves, with a 95% CI below 0.
2. **Guards:**
   - finisher PL log-likelihood is not worse (CI not below 0 for K2; unchanged by
     construction for K1);
   - the ECE of P(win) improves;
   - on the select split alone (2022–23), winner log-loss is not significantly worse.
3. **If both K1 and K2 pass:** take K1, the narrower change, unless K2 is also significantly
   better on the full order.

### Deliverable if adopted

- A `Calibrated` forecaster wrapping world_v1, fitting τ causally per season from its own past
  forecasts, registered as `world_v1_cal`.
- A test asserting that it reproduces the post-hoc study exactly.
- `world_v1` stays the reference until the holdout scoring.

---

## ADR — a data-driven reset response

### Evidence

Fixed reset rules cannot tell a reset that scrambles the order (2009, 2014, 2022) from one that
keeps it (1989, 1995, 2017). H2 helped the former and hurt the latter, and was rejected 4/8.
But the filter sees, race by race, whether its team ratings are being *surprised*.

### Mechanism

Covariance matching, as in adaptive Kalman filtering. After each race, for every team:

- `z_t = Σ g_eff / sqrt(Σ h_eff)`, summed over the team's cars: the standardised score
  (innovation) of the race result for that team.
- Under a correct predictive, E[z²] = 1.
- A team whose results keep surprising the filter (z² ≫ 1) is under-dispersed: inflate its
  variance by `1 + κ·max(0, z_t² − 1)`, capped.
- Only the team component is touched (D3: the movers are teams).

### Hypotheses (pre-registered; no others)

| | Where | Parameter |
|---|---|---|
| **R1** | only in the first 8 races of a reset season | κ_reset ≥ 0 |
| **R2** | every race (form swings within an era too) | κ_all ≥ 0 |

- κ = 0 reproduces the filter exactly; a bit-identity test asserts it.
- **Cap:** the inflation factor is capped at 4 per race.

### Protocol (as in the reset study)

- **Fitting:** the prequential race log-likelihood, with κ shrunk towards 0 (half-normal prior,
  sd 1). Protocol (a): fit on 1980–2013, score 2014+ out of sample. Protocol (b): fit on
  1980–2021, as a robustness check.
- **R1 cross-validation:** leave one reset out over the 8 resets of 1983–2022.
- **R2 cross-validation:** leave one *era* out: fit on the other eras' races and score the
  held-out era's, over the eras starting 1983–2022.
- **Backtest (for a variant that passes the cross-validation):** C1 and C4 on 2014–2023, with
  the reset study's guards G1–G5. For R2, G1 is "non-reset seasons not worse", which can now
  move.

### Decision rule

- **R1:** held-out resets gain on average **and** in at least 5 of 8; the backtest's early
  reset races at C1 are better on average, with no guard failing.
- **R2:** held-out eras gain on average **and** in a majority of them; the backtest's C1
  (all 2014–2023) is better on winner log-loss or full order with a 95% CI excluding 0, and
  not worse on the other; C4 is not worse.
- If both pass, take the one with the larger C1 gain; if they tie, R1, the narrower one.
- If neither passes, nothing changes.

---

## Order of work

1. CAL post-hoc study, on the harness's rolling forecasts 2010–2023.
2. CAL decision. If adopted, the `Calibrated` wrapper, its exactness test and registration.
3. ADR implementation, with the bit-identity test and the innovation formula test.
4. ADR fits, cross-validation, backtest and decision.
5. Build log and the plan's outcome note.
