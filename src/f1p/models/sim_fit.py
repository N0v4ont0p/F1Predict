"""Sub-models of ``race_sim_v2`` fitted from the lap tables (plan M6 §B.2).

Each sub-model is fitted on races whose laps are public at the forecast's checkpoint, and is
a small, explicit parametric piece the GPU simulator evaluates every lap:

* **overtaking** — the probability that a car within a second of the car ahead at the line
  is ahead of it one lap later, from observed position swaps between consecutive laps:
  ``logit p = a_circuit + b·Δpace + g·gap + d·Δtyre_age + e·Δsoftness``;
* **strategy** — the probability of pitting on a lap, from the actual pit decisions:
  tyre age (and its square), compound, stops made, laps left, neutralised lap;
* more sub-models (incidents and neutralisations, reliability split, starts) follow the same
  pattern.

Fitting uses one tool: a logistic regression with an L2 penalty on the slopes and partially
pooled group intercepts (circuit), solved exactly with L-BFGS on the analytic gradient —
the maximum a posteriori fit of a hierarchical logistic model with fixed scales.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy.optimize import minimize

from f1p.eval.context import Tables

FloatArray = NDArray[np.float64]

SOFTNESS = {"SOFT": 2.0, "MEDIUM": 1.0, "HARD": 0.0}  # wet tyres and unknown: no contrast


# -- penalised logistic regression with pooled group intercepts ------------------------------------
@dataclass
class LogitFit:
    names: list[str]
    coef: FloatArray  # intercept first, then one per feature
    groups: dict[str, float] = field(default_factory=dict)  # pooled intercept offsets
    n: int = 0
    events: int = 0

    def logit(self, x: FloatArray, group: list[str] | None = None) -> FloatArray:
        eta = self.coef[0] + x @ self.coef[1:]
        if group is not None:
            eta = eta + np.array([self.groups.get(g, 0.0) for g in group])
        return eta

    def prob(self, x: FloatArray, group: list[str] | None = None) -> FloatArray:
        return 1.0 / (1.0 + np.exp(-self.logit(x, group)))

    def offset(self, group: str) -> float:
        """A group's intercept offset (0: unseen groups get the pooled mean)."""
        return self.groups.get(group, 0.0)


def fit_logit(
    x: FloatArray,
    y: NDArray[np.bool_] | FloatArray,
    names: list[str],
    *,
    groups: list[str] | None = None,
    l2: float = 1.0,
    group_sd: float = 0.5,
) -> LogitFit:
    """MAP fit of ``logit p = c + x·b + u[group]`` with ``b ~ N(0, 1/l2)`` and
    ``u ~ N(0, group_sd²)`` (the intercept is flat)."""
    x = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    if not (np.isfinite(x).all() and np.isfinite(yv).all()):
        bad = [names[j] for j in range(x.shape[1]) if not np.isfinite(x[:, j]).all()]
        raise ValueError(f"non-finite inputs to fit_logit: {bad or ['y']}")
    n, k = x.shape
    labels = sorted(set(groups)) if groups is not None else []
    g_idx = (
        np.array([labels.index(g) for g in groups], dtype=np.int64)
        if groups is not None
        else np.zeros(n, dtype=np.int64)
    )
    n_g = len(labels)
    base = float(np.clip(yv.mean(), 1e-6, 1 - 1e-6))

    def unpack(theta: FloatArray) -> tuple[float, FloatArray, FloatArray]:
        return theta[0], theta[1 : 1 + k], theta[1 + k :]

    def objective(theta: FloatArray) -> tuple[float, FloatArray]:
        c, b, u = unpack(theta)
        eta = c + x @ b + (u[g_idx] if n_g else 0.0)
        # log(1 + e^eta) stably
        nll = float(np.sum(np.logaddexp(0.0, eta) - yv * eta))
        nll += 0.5 * l2 * float(b @ b) + 0.5 * float(u @ u) / group_sd**2
        r = 1.0 / (1.0 + np.exp(-eta)) - yv
        grad = np.empty_like(theta)
        grad[0] = r.sum()
        grad[1 : 1 + k] = x.T @ r + l2 * b
        if n_g:
            grad[1 + k :] = np.bincount(g_idx, weights=r, minlength=n_g) + u / group_sd**2
        return nll, grad

    theta0 = np.zeros(1 + k + n_g)
    theta0[0] = np.log(base / (1 - base))
    res = minimize(objective, theta0, jac=True, method="L-BFGS-B", options={"maxiter": 1000})
    if not res.success and np.linalg.norm(res.jac) > 1e-3 * max(n, 1):
        raise RuntimeError(f"fit_logit did not converge: {res.message}")
    c, b, u = unpack(res.x)
    return LogitFit(
        names,
        np.r_[c, b],
        dict(zip(labels, u.tolist(), strict=True)),
        n=n,
        events=int(yv.sum()),
    )


# -- the laps we learn from ------------------------------------------------------------------------
def _racing_laps(tables: Tables, laps: pl.DataFrame) -> pl.DataFrame:
    """Laps with their circuit and each car's typical clean-lap pace (median, s) in that race."""
    ev = tables["events"].select("event_id", "circuit_id", "season")
    pace = (
        laps.filter(pl.col("clean"))
        .group_by("event_id", "driver_id")
        .agg((pl.col("lap_ms").median() / 1000.0).alias("pace_s"))
    )
    return (
        laps.join(ev, on="event_id")
        .join(pace, on=["event_id", "driver_id"], how="left")
        .with_columns(
            pl.col("lap").max().over("event_id").alias("race_laps"),
            pl.col("compound").replace_strict(SOFTNESS, default=None).alias("softness"),
        )
    )


# -- overtaking ------------------------------------------------------------------------------------
OVERTAKE_FEATURES = ["pace_adv_s", "gap_s", "age_adv", "soft_adv"]


def overtaking_data(tables: Tables, laps: pl.DataFrame) -> pl.DataFrame:
    """One row per (lap, pair of cars consecutive on the road with the follower within 1 s
    at the line), on green-flag racing laps: did the follower lead the pair one lap later?

    Features, all from the follower's side: typical pace advantage (s/lap, + = faster),
    gap (s), tyre-age advantage (laps fresher), softness advantage (softer compound). Pairs
    where either car pits on either lap, laps under neutralisation and lap 1 are excluded."""
    df = _racing_laps(tables, laps).filter(pl.col("position").is_not_null())
    cols = [
        "event_id",
        "circuit_id",
        "lap",
        "driver_id",
        "position",
        "gap_ahead_s",
        "tyre_age",
        "softness",
        "pace_s",
        "in_lap",
        "out_lap",
        "neutral",
    ]
    cur = df.select(cols)
    nxt = df.select(
        "event_id",
        (pl.col("lap") - 1).alias("lap"),
        "driver_id",
        pl.col("position").alias("pos_next"),
        pl.col("in_lap").alias("in_next"),
        pl.col("out_lap").alias("out_next"),
        pl.col("neutral").alias("neutral_next"),
    )
    car = cur.join(nxt, on=["event_id", "lap", "driver_id"], how="inner")
    ahead = car.select(
        "event_id",
        "lap",
        (pl.col("position") + 1).alias("position"),
        pl.col("driver_id").alias("ahead_id"),
        pl.col("pos_next").alias("ahead_pos_next"),
        pl.col("tyre_age").alias("ahead_age"),
        pl.col("softness").alias("ahead_soft"),
        pl.col("pace_s").alias("ahead_pace"),
        (pl.col("in_lap") | pl.col("out_lap") | pl.col("in_next") | pl.col("out_next")).alias(
            "ahead_pit"
        ),
    )
    pairs = car.join(ahead, on=["event_id", "lap", "position"], how="inner").filter(
        (pl.col("lap") >= 2)
        & (pl.col("gap_ahead_s") < 1.0)
        & ~pl.col("neutral")
        & ~pl.col("neutral_next")
        & ~(pl.col("in_lap") | pl.col("out_lap") | pl.col("in_next") | pl.col("out_next"))
        & ~pl.col("ahead_pit")
        & pl.col("pace_s").is_not_null()
        & pl.col("ahead_pace").is_not_null()
    )
    return pairs.select(
        "event_id",
        "circuit_id",
        "lap",
        (pl.col("ahead_pace") - pl.col("pace_s")).alias("pace_adv_s"),
        pl.col("gap_ahead_s").alias("gap_s"),
        (pl.col("ahead_age") - pl.col("tyre_age")).cast(pl.Float64).fill_null(0.0).alias("age_adv"),
        (pl.col("softness") - pl.col("ahead_soft")).fill_null(0.0).alias("soft_adv"),
        (pl.col("pos_next") < pl.col("ahead_pos_next")).alias("passed"),
    )


def fit_overtaking(tables: Tables, laps: pl.DataFrame, **kw: float) -> LogitFit:
    d = overtaking_data(tables, laps)
    x = d.select(OVERTAKE_FEATURES).to_numpy().astype(np.float64)
    return fit_logit(
        x, d["passed"].to_numpy(), OVERTAKE_FEATURES, groups=d["circuit_id"].to_list(), **kw
    )


# -- strategy --------------------------------------------------------------------------------------
STRATEGY_FEATURES = [
    "age",
    "age2",
    "left",
    "stops",
    "neutral",
    "soft",
    "hard",
    "wet",
    "unknown",
]


def strategy_data(tables: Tables, laps: pl.DataFrame) -> pl.DataFrame:
    """One row per racing lap of a car that ran to the end of the race: did it pit (an
    in-lap)? Features: tyre age (/20, and squared), share of the race left, stops made,
    neutralised lap, compound (SOFT/HARD/wet vs MEDIUM; unknown before 2022)."""
    df = _racing_laps(tables, laps)
    finishers = df.group_by("event_id", "driver_id").agg(pl.col("lap").max().alias("done"))
    df = df.join(finishers, on=["event_id", "driver_id"]).filter(
        (pl.col("done") >= pl.col("race_laps") - 1) & (pl.col("lap") >= 1) & ~pl.col("out_lap")
    )
    return df.select(
        "event_id",
        "circuit_id",
        (pl.col("tyre_age").cast(pl.Float64) / 20.0).alias("age"),
        ((pl.col("tyre_age").cast(pl.Float64) / 20.0) ** 2).alias("age2"),
        ((pl.col("race_laps") - pl.col("lap")) / pl.col("race_laps")).alias("left"),
        (pl.col("stint") - 1).clip(0, 4).cast(pl.Float64).alias("stops"),
        pl.col("neutral").cast(pl.Float64),
        (pl.col("compound") == "SOFT").fill_null(False).cast(pl.Float64).alias("soft"),
        (pl.col("compound") == "HARD").fill_null(False).cast(pl.Float64).alias("hard"),
        pl.col("compound")
        .is_in(["INTERMEDIATE", "WET"])
        .fill_null(False)
        .cast(pl.Float64)
        .alias("wet"),
        # 2018-2021: no compound in the timing feed
        pl.col("compound").is_null().cast(pl.Float64).alias("unknown"),
        pl.col("in_lap").alias("pit"),
    )


def fit_strategy(tables: Tables, laps: pl.DataFrame, **kw: float) -> LogitFit:
    d = strategy_data(tables, laps)
    x = d.select(STRATEGY_FEATURES).to_numpy().astype(np.float64)
    return fit_logit(
        x, d["pit"].to_numpy(), STRATEGY_FEATURES, groups=d["circuit_id"].to_list(), **kw
    )
