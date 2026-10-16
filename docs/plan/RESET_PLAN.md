# Plan — regulation resets in the world model (C1)

A plan only; no code. Goal: forecast the first races of a regulation-reset season (2014, 2022,
and 2026 in the holdout) better before qualifying (C1), without changing anything else.

---

## 1. Evidence and where the error is

**What the diagnostic found.** `world_v1`, 2014–2023, first 6 rounds of each season:

| | C1 winner log-loss | C4 winner log-loss |
|---|---|---|
| normal seasons (42 races) | 1.35 | 1.20 |
| reset seasons (18 races) | **1.79** | 1.33 |

- At C1 the gap is real but driven by two seasons: 2014 early 1.85 (late 1.55) and 2022 early
  1.95 (late 1.33).
- 2017 went the other way: early 1.58, late 1.62.
- At C4 the gap is small, because qualifying and the grid already reveal the new order. **The
  target is C1 only.**

**How the filter treats a reset today** (`models/ratings.py`, `FITTED_V1`). The parameters
were fitted on 1980–2013 only, so 2014+ is honest out-of-sample.

- At a season boundary, team variance grows by `tau_team_season² = 0.44²`.
- In a reset season, by `(1.11 × 0.44)²`: `reset_multiplier = 1.11`, barely more than normal.
- `team_carryover = 0.79` keeps 79% of a team's rating at every boundary, reset or not.
- In-season, team variance grows by `tau_team² = 0.096²` per race, the same in race 1 as in
  race 20.
- The forecast does use the variance: `predictive_log_strength(mu, var, beta)` flattens
  uncertain teams. Inflating variance therefore both flattens the early forecasts and speeds
  up learning.
- C1's combiner (`rating`, `lap_pace`) is refitted per season on the 8 seasons before. It has
  no notion of "reset", and `lap_pace` at a reset's start is last season's pace, i.e. stale.

**Why 1.11 is plausibly too low for 2014/2022-type resets.**

- Most 1980–2013 resets kept the order largely intact, so averaging over them fits a small
  multiplier.
- The order-scrambling resets are a minority. One multiplier for all resets cannot be right for
  both kinds.

---

## 2. First step: diagnose before changing anything (decides which fix)

Run three diagnostics on 2014, 2017 and 2022, plus the 1980–2013 resets through the filter
alone. Each outcome points to a specific fix.

**D1. Over-confidence or wrong order?**

- For the first 6 rounds of each reset season, decompose C1's loss into:
  - *calibration*: ECE of P(win) and P(top 3);
  - *ranking*: Spearman of mean strength against the finishing order.
- Over-confident with a decent order → the forecast is too sharp; fix with variance (H1/H2).
- Wrong order and not over-confident → the information isn't there; variance helps only a
  little; the fix is faster learning (H2) or new information (H5).

**D2. How fast does the filter catch up?**

- Per reset season, the round at which C1's rolling log-loss returns to the normal-season
  level.
- Compare with the same curve for C4: qualifying shows how fast the truth is observable.
- A long lag at C1 against a short one at C4 points to slow learning (H2) or too much
  carryover (H3).

**D3. Which teams cause it?**

- The forecast-vs-result error per team in early reset races (2014: Mercedes under-rated, Red
  Bull and Lotus over-rated; 2022: Ferrari, Mercedes).
- A few big movers point to per-team variance (H1, H3).
- Diffuse error points to field-wide flattening (H2).

**Output:** a short table in the build log, and the shortlist of hypotheses the data supports.
**Only hypotheses the diagnostics support go forward** (no fishing).

---

## 3. Candidate changes

Each is small, physically meaningful and nested (each reduces to today's model at its default),
and **each acts only in reset seasons**. Normal seasons are bit-for-bit unchanged; a test
asserts this.

| | Change | Parameters | Mechanism |
|---|---|---|---|
| **H1** | Reset multiplier by *kind*: power-unit resets (1989, 1995, 2014, 2026; check whether 2006 should be added) vs chassis/aero resets (1983, 1998, 2009, 2017, 2022) | 2 multipliers, shrunk towards one shared value | Engine changes scramble the order more |
| **H2** | Early-season drift in reset seasons: per-race team drift `tau_team × (1 + (m−1)·e^{−(r−1)/L})` for race r | m ≥ 1, L > 0 (2) | Teams' true pace moves fastest in the first races of a new formula (development curves); the filter learns faster then |
| **H3** | Separate carryover in reset seasons, `team_carryover_reset` | 1 | Last year's order is worth less after a reset; means revert more towards the field |
| **H4** | Reset awareness in the C1 combiner: shrink `lap_pace`'s weight in the first k races of a reset season (stale pace) | k; or a fitted interaction, shrunk | Last season's lap pace mis-ranks the new cars |
| **H5** | *(Optional, data)* Pre-season testing pace as a C1 signal in reset seasons | needs a new data source | The only real new information before race 1 |

**Notes and reservations.**

- **Classifying the resets is a modelling decision, made before fitting and documented once.**
  It must not be tuned to the results. Borderline seasons are classified by the regulation
  text, not by outcomes: 2006 (V8 engines) is not in `MAJOR_RESETS`; 1998 changed both.
- H2 adds two parameters fitted on few events. Shrinkage (a prior centred on m = 1) is
  mandatory.
- H4 touches the combiner, not the filter, so it also changes C1 in non-reset seasons unless
  it is gated to reset seasons. It stays gated.
- H5: live-timing archives have testing only from about 2018, so of the evaluation resets
  only 2022 has it, plus 2026 in the holdout. It cannot be validated properly and stays
  optional (§7).

---

## 4. Fitting protocol (no peeking)

**What fits the parameters.** Same as `FITTED_V1`: the filter's prequential (one-step-ahead)
race log-likelihood.

**Which data.** Resets are rare: 1980–2013 has 1983, 1989, 1995, 1998 and 2009. Two options:

- **(a)** Fit on 1980–2013 only, exactly like `FITTED_V1`. Then 2014, 2017 and 2022 are clean
  out-of-sample tests, but the fit sees only 5 resets.
- **(b)** Fit on 1980–2021 (the tune split). Then 2014 and 2017 are in-sample, and only 2022
  (select) and 2026 (holdout) test it.

**Plan: run (a) as primary, (b) as a robustness check.** The decision rests on (a), because it
keeps 2014, 2017 and 2022 out of sample.

**Parameter estimation.**

- Only the new parameters are fitted; all `FITTED_V1` values stay fixed. That isolates the
  effect and avoids re-tuning the whole filter.
- Within each reset, the score counts the first 8 races, where the parameters act. The whole
  season is also reported.

**Leave-one-reset-out cross-validation (the main evidence, given so few events).**

- Over all resets 1983–2022 (8 events): fit on 7, score the 8th's prequential likelihood, then
  rotate.
- Report the gain per held-out reset and the number of resets that improve.
- This is not used for the 2014/2022 headline numbers, which come from protocol (a).

**Pre-registered variants (no others):**

1. H1;
2. H2;
3. H3;
4. H1 + H3;
5. H2 + H3;
6. the best of 1–5 plus H4, only if D2 showed stale pace.

That is six configurations, each counted in a multiple-comparison note.

---

## 5. Evaluation

**Primary.** C1 on the tune and select seasons' reset years (2014, 2017, 2022), first 8 rounds:

- winner log-loss;
- finisher PL log-likelihood;
- paired bootstrap against the current `world_v1` over those ~24 races.

**Guards. Any one failing kills the variant.**

- **G1.** Normal seasons are bit-for-bit identical (automated test).
- **G2.** No reset season is worse on C1 PL log-likelihood by more than its own bootstrap
  noise. 2017 is the trap: the cure must not hurt the reset that didn't scramble.
- **G3.** Rounds 9+ of reset seasons are no worse (the filter must re-converge).
- **G4.** C4 is no worse (C4 uses `rating` too).
- **G5.** Calibration: the early-reset ECE of P(win) is no worse.

**Decision rule (pre-registered).** Adopt the **simplest** variant that:

- improves held-out prequential likelihood in LORO on average *and* in at least 5 of 8 resets;
- improves C1 PL log-likelihood on the early 2014/2017/2022 races with a 95% CI not below 0 on
  the pooled ~24 races. A significant gain is unlikely at this sample size, so the requirement
  is "not negative, and positive on average".
- passes G1–G5.

If no variant qualifies, nothing changes, and the result is recorded.

**Final confirmation.** 2026 in the holdout, scored once with everything else. It is the real
test of whether this generalises.

---

## 6. Expected size of the gain, and risks

**Ceiling.**

- Early-reset C1 races are about 9% of races, and their excess winner log-loss is about 0.44
  nats.
- Closing all of it: about 0.04 off C1's overall winner log-loss.
- Realistic (25–50%): 0.01–0.02 overall, 0.1–0.2 within those races.
- No effect at C4, and none in normal seasons by construction.

**Risks and their guards.**

| Risk | Guard |
|---|---|
| Fitting to two dramatic resets (2014, 2022) | Protocol (a), LORO, shrinkage, 2017 guard (G2) |
| Too many variants tried on few events | Six pre-registered configurations; the simplest one wins |
| Flattening hurts resets that didn't scramble | G2, and H1's split by reset kind |
| Leaking into normal seasons | All changes gated to reset seasons; G1 test |
| The C1 combiner undoing the flattening (its rating weight is fitted on mostly normal seasons) | Check in D1 that predictive flattening survives the combiner; else consider H4-style gating of the combiner weight |
| Reset classification chosen to fit results | Fixed from regulation history before any fit (§3) |

---

## 7. Optional: pre-season testing (H5)

- **Source.** FastF1 / live-timing testing sessions (about 2018+). Per team: best and long-run
  pace in testing.
- **Use.** A C1 feature only in reset seasons and only for the first rounds. Its weight is
  shrunk and fitted on 2022 only, so it is very weakly identified.
- **Honest status.** It cannot be validated on more than one reset before the holdout. Build it
  only if §5's variants leave a large C1 gap in 2022. Expect a judgement call, not a statistical
  proof.

---

## 8. Steps and effort

| Step | What | Where | Effort |
|---|---|---|---|
| 1 | Diagnostics D1–D3 and their table | this cloud (minutes of compute) | ½ day |
| 2 | Freeze the reset classification and the variant list | doc | 1 h |
| 3 | Implement H1–H3 (+ H4 if supported), gated to reset seasons; G1 test | code | ½–1 day |
| 4 | Fit by protocol (a) and (b); LORO over 8 resets | this cloud (filter only, CPU minutes) | ½ day |
| 5 | Harness evaluation on 2014/2017/2022, guards G1–G5 | this cloud | ½ day |
| 6 | Decide by §5's rule, write the build log, update `FITTED_V1` only if adopted | doc | 1 h |
| 7 | 2026 confirmation, with the holdout scoring | M6 (B12) | — |

Nothing here needs the M6 except the final holdout scoring.
