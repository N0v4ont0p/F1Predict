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


def _fmean(s: pl.Series, default: float) -> float:
    """Mean of a numeric or boolean series (``default`` when empty)."""
    v = s.cast(pl.Float64).mean() if s.len() else None
    return float(v) if isinstance(v, int | float) else default


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


# -- starts ----------------------------------------------------------------------------------------
def run_to_turn1(tables: Tables, *, radius: float = 150.0) -> dict[str, float]:
    """Metres from the start/finish line (the first geometry point) to the first corner
    tighter than ``radius`` (a sustained run of three points), per circuit with geometry."""
    if "circuit_geometry" not in tables or tables["circuit_geometry"].height == 0:
        return {}
    out = {}
    for (cid,), g in tables["circuit_geometry"].group_by("circuit_id"):
        g = g.sort("point")
        xy = np.column_stack([g["x"].to_numpy(), g["y"].to_numpy()]) / 10.0  # dm → m
        seg = np.diff(np.vstack([xy, xy[:1]]), axis=0)
        ds = np.hypot(seg[:, 0], seg[:, 1])
        ang = np.arctan2(seg[:, 1], seg[:, 0])
        dh = (np.roll(ang, -1) - ang + np.pi) % (2 * np.pi) - np.pi
        w = 3
        turn = np.abs(sum(np.roll(dh, k) for k in range(-w, w)))
        span = sum(np.roll(ds, k) for k in range(-w, w))
        tight = span / np.maximum(turn, 1e-6) < radius
        run = np.convolve(tight.astype(int), np.ones(3, dtype=int), mode="same") == 3
        first = int(np.argmax(run)) if run.any() else len(ds) // 4
        out[str(cid)] = float(ds[:first].sum())
    return out


@dataclass
class StartModel:
    """Lap-1 order as a Plackett-Luce (Gumbel) ranking of start utilities
    ``u = w_slot·slot + w_clean·clean + w_run·slot·run + κ_circuit·slot + λ_team``:
    a negative slot weight keeps the grid order, and the smaller its magnitude the more the
    field shuffles. Cars retiring on lap 1 are censored in the fit; they come from
    ``lap1_dnf`` (per circuit, shrunk to the global rate)."""

    w_slot: float
    w_clean: float
    w_run: float
    circuit: dict[str, float]
    team: dict[str, float]
    run_mean: float
    run_sd: float
    run: dict[str, float]
    lap1_dnf: dict[str, float]
    lap1_dnf_global: float

    def utilities(self, slot: FloatArray, circuit: str, teams: list[str]) -> FloatArray:
        """Start utilities of a grid (``slot`` 0 = pole)."""
        z = (self.run.get(circuit, self.run_mean) - self.run_mean) / max(self.run_sd, 1e-9)
        slope = self.w_slot + self.w_run * z + self.circuit.get(circuit, 0.0)
        clean = (np.asarray(slot) % 2 == 0).astype(np.float64)  # odd places (P1, P3, …)
        lam = np.array([self.team.get(t, 0.0) for t in teams])
        return slope * np.asarray(slot, dtype=np.float64) + self.w_clean * clean + lam

    def p_lap1_dnf(self, circuit: str) -> float:
        return self.lap1_dnf.get(circuit, self.lap1_dnf_global)


def fit_start(
    tables: Tables, laps: pl.DataFrame, *, l2: float = 1.0, shrink: float = 200.0
) -> StartModel:
    """Fit on every race with lap-1 positions: grid slot from the results (pit-lane starts
    go to the back), lap-1 order from the laps, team = lineage x season."""
    from f1p.models import linear_pl

    ev = tables["events"].select("event_id", "circuit_id", "season")
    res = (
        tables["results"]
        .filter((pl.col("session") == "race") & pl.col("grid").is_not_null())
        .select("event_id", "driver_id", "constructor_id", "grid", "status", "laps")
    )
    lap1 = laps.filter(pl.col("lap") == 1).select("event_id", "driver_id", "position")
    runs = run_to_turn1(tables)
    run_vals = np.array(list(runs.values())) if runs else np.array([300.0])
    run_mean, run_sd = float(run_vals.mean()), float(run_vals.std() or 1.0)
    circuits: list[str] = []
    teams: list[str] = []
    rows = []
    started = res.join(ev, on="event_id").filter(
        pl.col("event_id").is_in(lap1["event_id"].unique().to_list())
    )
    for _, g in started.group_by("event_id", maintain_order=True):
        g = g.join(lap1, on=["event_id", "driver_id"], how="left").sort("grid")
        # slot: grid order, pit-lane starters (grid 0) at the back
        grid = g["grid"].to_numpy().astype(np.float64)
        grid = np.where(grid <= 0, grid.max() + 1, grid)
        slot = np.argsort(np.argsort(grid, kind="stable"), kind="stable").astype(np.float64)
        circ = str(g["circuit_id"][0])
        team = [f"{c}:{g['season'][0]}" for c in g["constructor_id"].to_list()]
        pos = g["position"].to_numpy()
        ok = ~np.isnan(pos.astype(np.float64))
        fin = np.flatnonzero(ok)[np.argsort(pos[ok], kind="stable")]
        rows.append((circ, team, slot, fin))
        circuits.append(circ)
        teams.extend(team)
    c_names = sorted(set(circuits))
    t_names = sorted(set(teams))
    c_i = {c: i for i, c in enumerate(c_names)}
    t_i = {t: i for i, t in enumerate(t_names)}
    p = 3 + len(c_names) + len(t_names)
    samples = []
    for circ, team, slot, fin in rows:
        n = len(slot)
        x = np.zeros((n, p))
        z = (runs.get(circ, run_mean) - run_mean) / run_sd
        x[:, 0] = slot
        x[:, 1] = (slot % 2 == 0).astype(np.float64)
        x[:, 2] = slot * z
        x[:, 3 + c_i[circ]] = slot
        for k, t in enumerate(team):
            x[k, 3 + len(c_names) + t_i[t]] = 1.0
        samples.append((x, fin.astype(np.intp)))
    prior = np.zeros(p)
    w = linear_pl.fit(samples, prior=prior, l2=l2)
    # lap-1 retirements per circuit, shrunk to the global rate
    first = res.join(ev, on="event_id").with_columns(
        (pl.col("status").str.starts_with("dnf") & (pl.col("laps") == 0)).alias("l1")
    )
    g_rate = _fmean(first["l1"], 0.0)
    per = first.group_by("circuit_id").agg(pl.col("l1").sum().alias("k"), pl.len().alias("n"))
    lap1_dnf = {
        str(c): (float(k) + shrink * g_rate) / (float(n) + shrink) for c, k, n in per.iter_rows()
    }
    return StartModel(
        float(w[0]),
        float(w[1]),
        float(w[2]),
        {c: float(w[3 + i]) for c, i in c_i.items()},
        {t: float(w[3 + len(c_names) + i]) for t, i in t_i.items()},
        run_mean,
        run_sd,
        runs,
        lap1_dnf,
        g_rate,
    )


# -- incidents, neutralisations, retirements ----------------------------------------------------
INCIDENT_FEATURES = ["lap1", "restart", "progress"]


def neutral_periods(laps: pl.DataFrame) -> pl.DataFrame:
    """Neutralised periods (SC, VSC, red flag: most of the field on a neutral lap), as
    (event_id, start, length) in laps."""
    by_lap = (
        laps.group_by("event_id", "lap")
        .agg((pl.col("neutral").mean() >= 0.5).alias("n"))
        .sort("event_id", "lap")
    )
    out = []
    for (eid,), g in by_lap.group_by("event_id", maintain_order=True):
        n = g["n"].to_numpy()
        lap = g["lap"].to_numpy()
        i = 0
        while i < len(n):
            if n[i]:
                j = i
                while j + 1 < len(n) and n[j + 1]:
                    j += 1
                out.append((str(eid), int(lap[i]), int(j - i + 1)))
                i = j + 1
            else:
                i += 1
    return pl.DataFrame(out, schema=["event_id", "start", "length"], orient="row")


@dataclass
class IncidentModel:
    """Per-lap incidents. An incident on a lap is a retirement in an incident then (with the
    neutralisation it causes, which starts that lap or the next), or a neutralisation that
    no retirement explains. Given an incident: how many cars retire (``dnf_counts``,
    P(0), P(1), P(2), P(3+)), whether the race is neutralised (always when nobody retires —
    otherwise the incident is unseen), for how many laps (``lengths``), and who is hit
    (``position_weights``, by running position decile). ``mechanical_share`` splits a car's
    retirement probability into mechanical (pace-independent) and incident parts."""

    hazard: LogitFit
    dnf_counts: FloatArray
    p_neutral_given_dnf: float
    lengths: NDArray[np.int64]
    position_weights: FloatArray
    mechanical_share: float


def fit_incidents(tables: Tables, laps: pl.DataFrame, **kw: float) -> IncidentModel:
    ev = tables["events"].select("event_id", "circuit_id")
    race_laps = laps.group_by("event_id").agg(pl.col("lap").max().alias("race_laps"))
    periods = neutral_periods(laps)
    res = tables["results"].filter(
        (pl.col("session") == "race") & pl.col("event_id").is_in(race_laps["event_id"].to_list())
    )
    crash = res.filter(pl.col("status") == "dnf_incident").select(
        "event_id", "driver_id", (pl.col("laps") + 1).alias("lap")
    )
    per_lap_dnf = crash.group_by("event_id", "lap").agg(pl.len().alias("dnfs"))
    # a crash during lap L usually neutralises the field from lap L or L + 1: such a
    # neutralisation belongs to the crash (one incident), the rest stand alone
    crash_laps = per_lap_dnf.select("event_id", "lap")
    explained = (
        pl.concat([crash_laps, crash_laps.with_columns(pl.col("lap") + 1)])
        .unique()
        .with_columns(pl.lit(True).alias("x"))
    )
    neutral = (
        periods.select("event_id", pl.col("start").alias("lap"))
        .join(explained, on=["event_id", "lap"], how="left")
        .with_columns(pl.col("x").fill_null(False))
    )
    # neutralisations explained by a crash are credited to the crash's lap
    credited = (
        neutral.filter(pl.col("x"))
        .join(
            crash_laps.with_columns(pl.lit(True).alias("same")), on=["event_id", "lap"], how="left"
        )
        .with_columns(
            pl.when(pl.col("same").fill_null(False))
            .then(pl.col("lap"))
            .otherwise(pl.col("lap") - 1)
            .alias("lap")
        )
        .select("event_id", "lap")
    )
    starts = (
        pl.concat([credited, neutral.filter(~pl.col("x")).select("event_id", "lap")])
        .unique()
        .with_columns(pl.lit(True).alias("sc"))
    )
    ends = periods.select(
        "event_id", (pl.col("start") + pl.col("length")).alias("lap"), pl.lit(True).alias("rs")
    )
    grid = (
        race_laps.join(ev, on="event_id")
        .with_columns(pl.int_ranges(1, pl.col("race_laps") + 1).alias("lap"))
        .explode("lap")
        .join(per_lap_dnf, on=["event_id", "lap"], how="left")
        .join(starts, on=["event_id", "lap"], how="left")
        .join(ends, on=["event_id", "lap"], how="left")
        .with_columns(
            pl.col("dnfs").fill_null(0),
            pl.col("sc").fill_null(False),
            pl.col("rs").fill_null(False),
        )
        .sort("event_id", "lap")
        .with_columns(
            # restart: the lap a period ends or the one after
            (pl.col("rs") | pl.col("rs").shift(1).over("event_id").fill_null(False)).alias(
                "restart"
            ),
            ((pl.col("dnfs") > 0) | pl.col("sc")).alias("incident"),
        )
    )
    x = np.column_stack(
        [
            (grid["lap"] == 1).cast(pl.Float64).to_numpy(),
            grid["restart"].cast(pl.Float64).to_numpy(),
            (grid["lap"] / grid["race_laps"]).to_numpy().astype(np.float64),
        ]
    )
    hazard = fit_logit(
        x, grid["incident"].to_numpy(), INCIDENT_FEATURES, groups=grid["circuit_id"].to_list(), **kw
    )
    inc = grid.filter(pl.col("incident"))
    k = np.clip(inc["dnfs"].to_numpy(), 0, 3)
    counts = np.bincount(k, minlength=4).astype(np.float64)
    with_dnf = inc.filter(pl.col("dnfs") > 0)
    p_neutral = _fmean(with_dnf["sc"], 0.5)
    lengths = periods["length"].to_numpy().astype(np.int64)
    # who is hit: running position (lap before) as a decile of the field
    pos = laps.select("event_id", "driver_id", (pl.col("lap") + 1).alias("lap"), "position")
    size = laps.group_by("event_id").agg(pl.col("driver_id").n_unique().alias("field"))
    hit = (
        crash.join(pos, on=["event_id", "driver_id", "lap"], how="left")
        .join(size, on="event_id")
        .filter(pl.col("position").is_not_null())
    )
    dec = np.clip(
        ((hit["position"].to_numpy() - 1) / hit["field"].to_numpy() * 10).astype(int), 0, 9
    )
    weights = ((np.bincount(dec, minlength=10) + 1.0) / (len(dec) + 10.0) * 10.0).astype(np.float64)
    dnfs = res.filter(pl.col("status").is_in(["dnf_mechanical", "dnf_incident"]))
    mech = _fmean(dnfs["status"] == "dnf_mechanical", 0.5)
    return IncidentModel(
        hazard,
        counts / counts.sum() if counts.sum() else np.array([1.0, 0, 0, 0]),
        p_neutral,
        lengths if len(lengths) else np.array([4], dtype=np.int64),
        weights,
        mech,
    )


# -- compound choice -------------------------------------------------------------------------------
DRY = ("SOFT", "MEDIUM", "HARD")
LEFT_BUCKETS = (0.25, 0.5)  # share of the race left at the stop: <0.25, 0.25-0.5, >0.5


@dataclass
class CompoundModel:
    """Which dry compound a car starts on, and which it fits at a stop given the stint it
    starts and the share of the race left (bucketed), from dry races with compound data
    (2022+). Probabilities in :data:`DRY` order; Laplace-smoothed."""

    start: FloatArray  # (3,)
    next: FloatArray  # (stint 2/3/4+, left bucket, 3)

    def left_bucket(self, left: FloatArray) -> NDArray[np.int64]:
        return np.searchsorted(np.asarray(LEFT_BUCKETS), np.asarray(left), side="right").astype(
            np.int64
        )


def fit_compounds(laps: pl.DataFrame, *, smooth: float = 1.0) -> CompoundModel:
    stints = (
        laps.filter(pl.col("compound").is_not_null())
        .with_columns(pl.col("lap").max().over("event_id").alias("race_laps"))
        .group_by("event_id", "driver_id", "stint")
        .agg(
            pl.col("compound").first(),
            pl.col("lap").min().alias("first"),
            pl.col("race_laps").first(),
        )
    )
    wet_races = stints.filter(~pl.col("compound").is_in(list(DRY)))["event_id"].unique()
    dry = stints.filter(~pl.col("event_id").is_in(wet_races.to_list()))
    idx = {c: i for i, c in enumerate(DRY)}
    start = np.full(3, smooth)
    for c in dry.filter(pl.col("stint") == 1)["compound"].to_list():
        start[idx[c]] += 1
    nxt = np.full((3, 3, 3), smooth)
    later = dry.filter(pl.col("stint") >= 2)
    left = 1.0 - (later["first"] - 1).cast(pl.Float64) / later["race_laps"].cast(pl.Float64)
    buckets = np.searchsorted(np.asarray(LEFT_BUCKETS), left.to_numpy(), side="right")
    for st, b, c in zip(
        later["stint"].to_list(), buckets.tolist(), later["compound"].to_list(), strict=True
    ):
        nxt[min(st, 4) - 2, b, idx[c]] += 1
    return CompoundModel(start / start.sum(), nxt / nxt.sum(axis=2, keepdims=True))
