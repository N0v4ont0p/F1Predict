# Plan — the C1 combiner across regulation resets (W1), to pair with ADR R1

> **Outcome:** done; **not adopted.**
>
> - R1 + W1 keeps the early-reset gain (−0.084 winner log-loss) but still fails G1: normal
>   seasons get worse at C1 by +0.0051 per race (95% CI +0.0014…+0.0086).
> - W1 alone makes the full order slightly worse.
> - R1 lowers the combiner's rating weight whether or not the early reset races are in its
>   training, so W1 does not reach the cause.
>
> The details are in the build log of `M6_COMPUTE_PLAN.md`. World_v1 is unchanged.

Fixed **before** any run. The holdout (2024–2026) is not touched.

## Why

ADR R1 (`CALIBRATION_ADAPTIVE_PLAN.md`) makes world_v1's rating filter learn faster in the first
races of a reset season. It improves those races at C1 significantly (winner log-loss −0.083,
95% CI −0.157…−0.011). But it fails guard G1: non-reset seasons get worse at C1 by +0.0045 per
race.

A diagnostic (build log, "variance give-back") located that loss in the **combiner**, not the
filter:

- At the filter level, R1 does not lose in normal seasons.
- The C1 combiner is refitted each season on the 8 seasons before. With R1, the early reset
  races in that window carry different ratings, and the fitted rating weight falls (2015:
  1.75 → 1.68). A smaller weight flattens every later forecast.

## Hypothesis (pre-registered; no others)

**W1:** the combiner's weights are fitted without the first 8 races (`ADR_ROUNDS`) of reset
seasons. The rating filter is still updated by those races, and the forecasts of those races
are unchanged in form. Only their use as *training samples* for the combiner's weights and
temperatures goes. The retirement calibration keeps every race.

- Reasoning: those races follow a different regime (the old order is wrong and the filter is
  catching up), so they should not set the rating weight used in normal seasons.
- Setting: `WorldV1.combiner_skip_reset_rounds` (default 0, unchanged). It is applied at C1
  and C4 alike.

**Candidate: R1 + W1.** That is `adapt_reset` = 0.066 (protocol (a), fitted on 1980–2013,
unchanged) together with W1 at 8 rounds. **W1 alone** is run as a control. It is adopted only if
it passes the same rule on its own.

## Evaluation

The rolling harness, C1 and C4, 2014–2023, paired bootstrap vs world_v1, with the reset study's
guards. G1 is read as **non-reset seasons not worse**: the 95% CI must not exclude 0 in the bad
direction.

## Decision rule (all must hold for the candidate)

1. **Primary:** early-reset races (rounds ≤ 8 of 2014, 2017, 2022) at C1 are better on winner
   log-loss or full order, with a 95% CI excluding 0. The other measure must not be
   significantly worse.
2. **G1:** non-reset seasons are not significantly worse at C1 or C4, on winner log-loss or
   full order.
3. **G2:** no reset season is significantly worse on C1 full order.
4. **G3:** reset rounds 9+ are not significantly worse.
5. **G4:** C4 overall is not significantly worse.
6. **G5:** the early-reset ECE of P(win) is not worse.
7. **Overall:** all 2014–2023 C1 is not significantly worse on either measure.

If R1+W1 fails, nothing changes. If both R1+W1 and W1 alone pass, take the one with the larger
all-C1 winner gain.

## Honest limits

- W1 was formed after seeing 2014–2023, and the evaluation uses those seasons. A pass is
  therefore weaker evidence than the earlier studies, and the holdout (2026 is a reset) is the
  real test.
- One hypothesis and one control limit the multiple-comparison risk. Nothing is tuned: the
  8-round span is R1's own window, fixed in advance.
