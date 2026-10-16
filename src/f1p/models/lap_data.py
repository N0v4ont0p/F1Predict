"""Design arrays for the Bayesian lap model ``lap_v2`` (plan M6 §A.1).

Everything the sampler needs for one window of races is built here, once, in NumPy: integer
indices per lap, the spline bases and the Hilbert-space GP basis over race order. The model
itself (:mod:`f1p.models.lap_v2`) only does arithmetic on these arrays, so the same window
feeds every sampler, the SBC simulator and the sequential (SMC) updates.

Terms and their bases:

* **car pace** per team: a level per (team, season) moving as a random walk across seasons
  (the regulation/new-car reset), plus a Matérn-3/2 GP over the window's race order through
  ``m`` Hilbert-space basis functions (Solin & Särkkä; Riutort-Mayol et al. 2023), plus
  team-specific slopes on circuit layout features (straights, slow corners, street circuit,
  km per corner);
* **driver**: a level per (driver, season), again a random walk across seasons, a rookie
  offset, and a driver x wet-lap effect;
* **tyres**: compound offsets (contrasts within a race, against its most-used compound) and a
  **monotone I-spline in tyre age** per compound,
  scaled by a wear rate per compound x race (wear and the cliff);
* **fuel**: laps remaining, with a slope per season;
* **track evolution**: a smooth curve in race progress per event, made orthogonal to a
  straight line so it cannot trade off against fuel burn (fuel *is* a straight line in race
  progress within an event; the fuel slope therefore also carries any linear evolution);
* **traffic**: a hinge basis in the gap to the car ahead (zero beyond 3 s) plus a DRS term
  (gap under 1 s at the line, the best public proxy for the detection point);
* **in- and out-laps**: their own offsets per event (pit loss), with a wider noise scale;
* **noise**: a scale per driver x event.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy.interpolate import BSpline

from f1p.eval.context import Tables

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

COMPOUNDS = ("MEDIUM", "SOFT", "HARD", "INTERMEDIATE", "WET", "UNKNOWN")
# Compound offsets are contrasts *within* a race, against the race's most-used compound (its
# base carries that one): a race run on a single compound — every 2018-2021 race (UNKNOWN: no
# compound in the timing feed) or a full-wet race — has no offset to confuse with its base.
# Each contrast is shrunk towards the difference of global compound levels, MEDIUM = 0.
OFFSET_COMPOUNDS = COMPOUNDS[1:]  # the global levels that are free
WET_COMPOUNDS = ("INTERMEDIATE", "WET")
AGE_KNOTS = (0.0, 4.0, 12.0, 25.0, 70.0)  # laps; the I-spline is flat beyond 70
GAP_HINGES = (0.5, 1.0, 2.0, 3.0)  # s
FUEL_SCALE = 50.0  # laps remaining / 50: the fuel slope is the effect of 50 laps of fuel
EVO_BASIS = 5
HSGP_M = 30
HSGP_C = 2.5  # boundary factor L = c · max|x|: the forecast race sits at the edge, where a
# small c underestimates the variance (Riutort-Mayol et al. 2023: c ≥ 4.5 l/S, m ≥ 3.42 c S/l for
# Matérn-3/2; c = 2.5, m = 30 hold down to l ≈ 3.5 races on a 24-race window)
CIRCUIT_FEATURES = ("straight", "slow", "street", "km_per_turn")


# -- bases ---------------------------------------------------------------------------------------
def ispline_basis(
    x: FloatArray, knots: tuple[float, ...] = AGE_KNOTS, degree: int = 2
) -> FloatArray:
    """Monotone I-spline basis: each column rises from 0 to 1, so non-negative weights give a
    non-decreasing curve that starts at 0 (Ramsay 1988). Built as the reversed cumulative sum
    of a B-spline basis of one degree higher."""
    lo, hi = knots[0], knots[-1]
    x = np.clip(np.asarray(x, dtype=np.float64), lo, hi)
    k = degree + 1
    t = np.r_[[lo] * (k + 1), knots[1:-1], [hi] * (k + 1)]
    n_basis = len(t) - k - 1
    b = BSpline.design_matrix(x, t, k, extrapolate=False).toarray()
    if b.shape[1] != n_basis:  # pragma: no cover - scipy contract
        raise AssertionError("unexpected basis size")
    i = np.cumsum(b[:, ::-1], axis=1)[:, ::-1]  # I_j = sum_{l >= j} B_l
    # I_0 ≡ 1 is the intercept. I_1 = 1 - B_0 saturates within the first knot interval (4 laps):
    # for nearly every racing lap it would act as a second, positive-only compound offset,
    # identified only by the few laps at ages 0-3. Dropping it, the curve starts flat and bends
    # over ~12 laps; the out-lap offset carries the warm-up lap.
    return i[:, 2:]


def evolution_basis(u: FloatArray, n: int = EVO_BASIS) -> FloatArray:
    """Cubic B-splines in race progress ``u`` ∈ [0, 1] with their constant and linear parts
    removed (least squares on a uniform grid), so the curve is pure curvature: ``n - 2``
    columns, each with unit RMS over the race."""
    k = 3
    inner = np.linspace(0.0, 1.0, n - k + 1)[1:-1]
    t = np.r_[[0.0] * (k + 1), inner, [1.0] * (k + 1)]
    grid = np.linspace(0.0, 1.0, 201)
    lin = np.column_stack([np.ones_like(grid), grid])
    bg = BSpline.design_matrix(grid, t, k).toarray()
    coef, *_ = np.linalg.lstsq(lin, bg, rcond=None)  # (2, n): each basis' straight line
    _, sv, vt = np.linalg.svd(bg - lin @ coef, full_matrices=False)
    rot = vt[: n - 2].T / sv[: n - 2] * np.sqrt(len(grid))  # (n, n - 2)
    uc = np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0)
    b = BSpline.design_matrix(uc, t, k).toarray()
    return (b - np.column_stack([np.ones_like(uc), uc]) @ coef) @ rot


def gap_basis(gap: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Hinge functions ``(h - gap)+ / h`` for the hinges in :data:`GAP_HINGES` (zero for a
    car more than 3 s behind, or with nobody ahead) and the DRS indicator."""
    g = np.asarray(gap, dtype=np.float64)
    g = np.where(np.isfinite(g), g, 99.0)
    hinges = np.column_stack([np.clip(h - g, 0.0, None) / h for h in GAP_HINGES])
    return hinges, (g < 1.0).astype(np.float64)


@dataclass(frozen=True, slots=True)
class HSGPBasis:
    """Hilbert-space approximation of a stationary GP on ``[-L, L]``."""

    phi: FloatArray  # (points, m)
    sqrt_eig: FloatArray  # (m,) √λ_j: the frequencies at which the spectral density is read
    center: float
    half_width: float

    def at(self, x: FloatArray) -> FloatArray:
        return hsgp_phi(
            np.asarray(x, dtype=np.float64) - self.center, self.half_width, len(self.sqrt_eig)
        )


def hsgp_phi(xc: FloatArray, half_width: float, m: int) -> FloatArray:
    j = np.arange(1, m + 1)
    return np.sqrt(1.0 / half_width) * np.sin(
        np.pi * j * (xc[:, None] + half_width) / (2.0 * half_width)
    )


def hsgp_basis(x: FloatArray, m: int = HSGP_M, c: float = HSGP_C) -> HSGPBasis:
    x = np.asarray(x, dtype=np.float64)
    center = float((x.min() + x.max()) / 2.0)
    half = c * max(float(np.max(np.abs(x - center))), 1.0)
    sqrt_eig = np.pi * np.arange(1, m + 1, dtype=np.float64) / (2.0 * half)
    return HSGPBasis(hsgp_phi(x - center, half, m), sqrt_eig, center, half)


def matern32_sqrt_psd(sqrt_eig: FloatArray, amplitude: float, lengthscale: float) -> FloatArray:
    """√S(ω) for the 1-D Matérn-3/2 kernel (NumPy twin of the model's tensor version)."""
    s = (
        amplitude**2
        * 12.0
        * np.sqrt(3.0)
        / lengthscale**3
        / (3.0 / lengthscale**2 + sqrt_eig**2) ** 2
    )
    return np.sqrt(s)


def helmert(n: int) -> FloatArray:
    """(n, n-1) orthonormal basis of the zero-sum subspace, scaled so that ``H @ z`` with
    ``z ~ N(0, I)`` has unit marginal variance and sums to exactly zero."""
    h = np.zeros((n, max(n - 1, 0)))
    for j in range(1, n):
        h[:j, j - 1] = 1.0 / np.sqrt(j * (j + 1))
        h[j, j - 1] = -j / np.sqrt(j * (j + 1))
    return h * np.sqrt(n / (n - 1)) if n > 1 else h


def zero_sum_basis(groups: IntArray) -> FloatArray:
    """Block-diagonal :func:`helmert` bases: rows of one group sum to zero (a group of one
    is pinned at zero). Removes directions the likelihood cannot see — the common level of
    car or driver effects that every event re-centres — from the sampler's space, where they
    would be long, non-axis-aligned ridges that a diagonal mass matrix cannot adapt to."""
    groups = np.asarray(groups)
    labels = list(dict.fromkeys(groups.tolist()))
    blocks = []
    for g in labels:
        rows = np.flatnonzero(groups == g)
        h = np.zeros((len(groups), max(len(rows) - 1, 0)))
        h[rows] = helmert(len(rows))
        blocks.append(h)
    return np.hstack(blocks) if blocks else np.zeros((len(groups), 0))


def walk_matrix(entity: IntArray, season: IntArray) -> FloatArray:
    """For (entity, season) rows sorted by entity then season: the 0/1 matrix ``C`` with
    ``level = C @ steps``, so each row's level is its entity's first draw plus every later
    step up to that season (a random walk across seasons)."""
    n = len(entity)
    c = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            if entity[j] == entity[i] and season[j] <= season[i]:
                c[i, j] = 1.0
    return c


# -- circuits --------------------------------------------------------------------------------------
def circuit_features(tables: Tables) -> pl.DataFrame:
    """Layout features per circuit, standardised across circuits: the share of the lap on
    straights (radius > 400 m) and in slow corners (radius < 60 m) from the circuit geometry
    (0 = field average where the geometry is unknown), a street-circuit flag, and km per
    corner from the circuit table."""
    circuits = tables["circuits"].select("circuit_id", "circuit_type", "length_km", "turns")
    geo_rows = []
    if "circuit_geometry" in tables and tables["circuit_geometry"].height:
        for (cid,), g in tables["circuit_geometry"].group_by("circuit_id"):
            g = g.sort("point")
            xy = np.column_stack([g["x"].to_numpy(), g["y"].to_numpy()]) / 10.0  # dm → m
            seg = np.diff(np.vstack([xy, xy[:1]]), axis=0)  # closed loop
            ds = np.hypot(seg[:, 0], seg[:, 1])
            ang = np.arctan2(seg[:, 1], seg[:, 0])
            dh = (np.roll(ang, -1) - ang + np.pi) % (2 * np.pi) - np.pi  # turn at each point
            w = 3  # heading change over ±3 points (~70 m) smooths sensor noise
            turn = np.abs(sum(np.roll(dh, k) for k in range(-w, w)))
            span = sum(np.roll(ds, k) for k in range(-w, w))
            radius = span / np.maximum(turn, 1e-6)
            total = ds.sum()
            geo_rows.append(
                {
                    "circuit_id": str(cid),
                    "straight": float(ds[radius > 400.0].sum() / total),
                    "slow": float(ds[radius < 60.0].sum() / total),
                }
            )
    geo = (
        pl.DataFrame(geo_rows)
        if geo_rows
        else pl.DataFrame(
            schema={"circuit_id": pl.String, "straight": pl.Float64, "slow": pl.Float64}
        )
    )
    df = circuits.join(geo, on="circuit_id", how="left").with_columns(
        (pl.col("circuit_type") == "STREET").cast(pl.Float64).alias("street"),
        (pl.col("length_km") / pl.col("turns").cast(pl.Float64).clip(lower_bound=1)).alias(
            "km_per_turn"
        ),
    )
    out = [pl.col("circuit_id")]
    for f in CIRCUIT_FEATURES:
        col = pl.col(f)
        out.append(((col - col.mean()) / col.std()).fill_nan(0.0).fill_null(0.0).alias(f))
    return df.select(out)


# -- the window --------------------------------------------------------------------------------
@dataclass
class WindowData:
    """One window of races, as arrays. Lap arrays have one entry per modelled lap."""

    events: list[str]
    event_season: IntArray
    event_center: FloatArray  # median clean log lap time per event (y is centred by it)
    event_circuit: list[str]
    teams: list[str]
    drivers: list[str]
    seasons: list[int]
    # laps
    y: FloatArray
    train: NDArray[np.bool_]  # in the likelihood (False: held-out laps of the forecast race)
    ev: IntArray
    team: IntArray
    driver: IntArray
    ts: IntArray  # team-season row
    ds: IntArray  # driver-season row
    de: IntArray  # driver-event row (noise scale)
    off: IntArray  # compound-offset row per lap (-1: reference compound)
    wear: IntArray  # compound-event row of the wear curve
    age: FloatArray  # (laps, k) I-spline basis
    tyre_age: FloatArray  # laps on the tyre (raw)
    fuel: FloatArray
    fuel_season: IntArray
    evo: FloatArray  # (laps, k) track-evolution basis
    gap: FloatArray  # (laps, k) hinge basis
    drs: FloatArray
    wet: FloatArray
    pit: IntArray  # 0 racing lap, 1 in-lap, 2 out-lap
    # structure
    ts_team: IntArray
    ts_season: IntArray
    ts_walk: FloatArray
    ds_driver: IntArray
    ds_season: IntArray
    ds_walk: FloatArray
    ds_rookie: FloatArray
    off_compound: IntArray  # offset row → COMPOUNDS index of the compound
    off_ref: IntArray  # offset row → COMPOUNDS index of the race's reference compound
    off_event: IntArray
    wear_compound: IntArray  # wear row → COMPOUNDS index
    wear_event: IntArray
    de_driver: IntArray
    de_event: IntArray
    team_present: FloatArray  # (teams, events)
    ds_present: FloatArray  # (driver-seasons, events)
    hsgp: HSGPBasis
    circuit_x: FloatArray  # (events, features)
    entries: list[tuple[str, str]]  # forecast race: (driver, team lineage) per car
    entry_team: IntArray
    entry_ds: IntArray
    next_index: int | None  # the forecast race's event index (the last), if any
    meta: dict[str, object] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.y)

    def summary(self) -> dict[str, int]:
        return {
            "laps": int(self.train.sum()),
            "holdout_laps": int((~self.train).sum()),
            "events": len(self.events),
            "teams": len(self.teams),
            "drivers": len(self.drivers),
            "team_seasons": len(self.ts_team),
            "driver_seasons": len(self.ds_driver),
            "wear_curves": len(self.wear_compound),
        }


def _index(values: list[str]) -> dict[str, int]:
    return {v: i for i, v in enumerate(values)}


def lap_frame(laps: pl.DataFrame, tables: Tables) -> pl.DataFrame:
    """Modelled laps — clean racing laps plus in- and out-laps (not lap 1, not neutralised) —
    with team lineage, season order and the raw regressors."""
    ev = tables["events"].select("event_id", "season", "round", "circuit_id")
    res = (
        tables["results"]
        .filter(pl.col("session") == "race")
        .select("event_id", "driver_id", "constructor_id")
    )
    cs = tables["constructor_seasons"].select("season", "constructor_id", "lineage_id")
    wet_events: set[str] = set()
    if "weather_observed" in tables and tables["weather_observed"].height:
        wo = tables["weather_observed"]
        wet_events = set(wo.filter(pl.col("wet").fill_null(False))["event_id"].to_list())
    df = (
        laps.with_columns(
            pl.col("lap").max().over("event_id").alias("race_laps"),
        )
        .filter(
            pl.col("clean")
            | ((pl.col("in_lap") | pl.col("out_lap")) & ~pl.col("neutral") & (pl.col("lap") > 1))
        )
        .join(ev, on="event_id")
        .join(res, on=["event_id", "driver_id"])
        .join(cs, on=["season", "constructor_id"], how="left")
        .with_columns(
            pl.coalesce("lineage_id", "constructor_id").alias("team"),
            pl.col("compound").fill_null("UNKNOWN"),
            pl.col("tyre_age").fill_null(pl.col("lap")).cast(pl.Float64),
            (pl.col("lap_ms").cast(pl.Float64) / 1000.0).log().alias("log_s"),
            ((pl.col("race_laps") - pl.col("lap")).cast(pl.Float64) / FUEL_SCALE).alias("fuel"),
            (pl.col("lap").cast(pl.Float64) / pl.col("race_laps").cast(pl.Float64)).alias("u"),
            pl.when(pl.col("in_lap"))
            .then(1)
            .when(pl.col("out_lap"))
            .then(2)
            .otherwise(0)
            .alias("pit"),
        )
    )
    unknown_wet = (pl.col("compound") == "UNKNOWN") & pl.col("event_id").is_in(list(wet_events))
    return df.with_columns(
        (pl.col("compound").is_in(list(WET_COMPOUNDS)) | unknown_wet).cast(pl.Float64).alias("wet")
    ).sort("season", "round", "driver_id", "lap")


def first_seasons(tables: Tables) -> dict[str, int]:
    """Each driver's first season with a race start (for the rookie offset)."""
    ev = tables["events"].select("event_id", "season")
    res = tables["results"].filter(pl.col("session") == "race").select("event_id", "driver_id")
    first = res.join(ev, on="event_id").group_by("driver_id").agg(pl.col("season").min())
    return dict(first.iter_rows())


def _lineage(tables: Tables, season: int) -> dict[str, str]:
    cs = tables["constructor_seasons"].filter(pl.col("season") == season)
    return {
        str(c): str(lin or c) for c, lin in cs.select("constructor_id", "lineage_id").iter_rows()
    }


def build_window(
    laps: pl.DataFrame,
    tables: Tables,
    *,
    next_event: str | None = None,
    entries: list[tuple[str, str]] | None = None,
    holdout: pl.DataFrame | None = None,
) -> WindowData | None:
    """Arrays for one window of races and, optionally, the race it forecasts.

    ``next_event`` becomes the last event of the window, with no laps in the likelihood: its
    car pace is the GP extrapolated one race on (plus a season step when it opens a season),
    and ``entries`` — its (driver, constructor) pairs — decide which team- and driver-season
    rows exist there. ``holdout`` (that race's laps, for validation) is attached with
    ``train = False``: scored by the posterior predictive, never fitted."""
    df = lap_frame(laps, tables)
    # an event needs racing laps (its centre and base come from them)
    racing = df.filter(pl.col("pit") == 0)["event_id"].unique().to_list()
    df = df.filter(pl.col("event_id").is_in(racing))
    if df.filter(pl.col("pit") == 0).height < 50:
        return None
    order = df.select("event_id", "season", "round", "circuit_id").unique().sort("season", "round")
    next_season: int | None = None
    if next_event is not None:
        nxt = tables["events"].filter(pl.col("event_id") == next_event)
        if nxt.height == 0:
            raise ValueError(f"unknown event {next_event!r}")
        row = nxt.select("event_id", "season", "round", "circuit_id")
        next_season = int(row["season"][0])
        order = pl.concat([order.filter(pl.col("event_id") != next_event), row])
        df = df.filter(pl.col("event_id") != next_event)
    df = df.with_columns(pl.lit(True).alias("train"))
    if holdout is not None and next_event is not None:
        hold = lap_frame(holdout.filter(pl.col("event_id") == next_event), tables)
        if hold.height:
            df = pl.concat(
                [df, hold.with_columns(pl.lit(False).alias("train"))], how="vertical_relaxed"
            )
            if entries is None:
                res = tables["results"].filter(
                    (pl.col("event_id") == next_event) & (pl.col("session") == "race")
                )
                # every starter is forecast, including cars with no laps to score (a lap-1 DNF)
                entries = [
                    (str(d), str(c))
                    for d, c in res.select("driver_id", "constructor_id").iter_rows()
                ]
    entry_rows: list[tuple[str, str]] = []
    if next_event is not None and entries:
        lin = _lineage(tables, next_season or 0)
        entry_rows = [(d, lin.get(c, c)) for d, c in entries]

    events = order["event_id"].to_list()
    ev_i = _index(events)
    seasons = sorted({*order["season"].to_list()})
    teams = sorted({*df["team"].to_list(), *(t for _, t in entry_rows)})
    drivers = sorted({*df["driver_id"].to_list(), *(d for d, _ in entry_rows)})
    team_i, drv_i = _index(teams), _index(drivers)
    train = df["train"].to_numpy()

    # event centring (median racing log lap) keeps the event bases near 0 for the sampler; a
    # held-out race is centred on its own median (both models are scored on relative times)
    centre = dict(
        df.filter(pl.col("pit") == 0).group_by("event_id").agg(pl.col("log_s").median()).iter_rows()
    )
    event_center = np.array([centre.get(e, 0.0) for e in events])
    ev_arr = np.array([ev_i[e] for e in df["event_id"].to_list()], dtype=np.int64)
    y = df["log_s"].to_numpy() - event_center[ev_arr]

    team_arr = np.array([team_i[t] for t in df["team"].to_list()], dtype=np.int64)
    drv_arr = np.array([drv_i[d] for d in df["driver_id"].to_list()], dtype=np.int64)
    season_arr = df["season"].to_numpy().astype(np.int64)
    e_next = len(events) - 1 if next_event is not None else None
    ent_team = np.array([team_i[t] for _, t in entry_rows], dtype=np.int64)
    ent_drv = np.array([drv_i[d] for d, _ in entry_rows], dtype=np.int64)
    ent_season = np.full(len(entry_rows), next_season or 0, dtype=np.int64)

    # team-season and driver-season rows (sorted by entity, then season) with their walks
    ts_rows = sorted(
        {*zip(team_arr.tolist(), season_arr.tolist(), strict=True)}
        | {*zip(ent_team.tolist(), ent_season.tolist(), strict=True)}
    )
    ts_i = {r: i for i, r in enumerate(ts_rows)}
    ds_rows = sorted(
        {*zip(drv_arr.tolist(), season_arr.tolist(), strict=True)}
        | {*zip(ent_drv.tolist(), ent_season.tolist(), strict=True)}
    )
    ds_i = {r: i for i, r in enumerate(ds_rows)}
    ts_team = np.array([r[0] for r in ts_rows], dtype=np.int64)
    ts_season = np.array([r[1] for r in ts_rows], dtype=np.int64)
    ds_driver = np.array([r[0] for r in ds_rows], dtype=np.int64)
    ds_season = np.array([r[1] for r in ds_rows], dtype=np.int64)
    first = first_seasons(tables)
    ds_rookie = np.array(
        [float(first.get(drivers[d], s) >= s) for d, s in ds_rows], dtype=np.float64
    )
    ds_lap = np.array(
        [ds_i[r] for r in zip(drv_arr.tolist(), season_arr.tolist(), strict=True)], dtype=np.int64
    )

    # compound offsets and wear rates per (compound, event)
    comp = df["compound"].to_list()
    comp_i = {c: i for i, c in enumerate(COMPOUNDS)}
    pairs = list(zip(comp, ev_arr.tolist(), strict=True))
    racing = df["pit"].to_numpy() == 0
    counts: dict[int, dict[str, int]] = {}
    for (c, e), r in zip(pairs, racing, strict=True):
        if r:
            counts.setdefault(e, {}).setdefault(c, 0)
            counts[e][c] += 1
    # the reference: the most-used compound on racing laps (ties: COMPOUNDS order)
    ref = {e: min(cnt, key=lambda c: (-cnt[c], comp_i[c])) for e, cnt in counts.items()}
    off_rows = sorted({(comp_i[c], e) for c, e in pairs if c != ref.get(e, c)})
    off_i = {r: i for i, r in enumerate(off_rows)}
    off = np.array([off_i.get((comp_i[c], e), -1) for c, e in pairs], dtype=np.int64)
    wear_rows = sorted({(comp_i[c], e) for c, e in pairs})
    wear_i = {r: i for i, r in enumerate(wear_rows)}
    wear = np.array([wear_i[(comp_i[c], e)] for c, e in pairs], dtype=np.int64)

    de_rows = sorted(set(zip(drv_arr.tolist(), ev_arr.tolist(), strict=True)))
    de_i = {r: i for i, r in enumerate(de_rows)}
    de = np.array(
        [de_i[r] for r in zip(drv_arr.tolist(), ev_arr.tolist(), strict=True)], dtype=np.int64
    )

    season_i = _index([str(s) for s in seasons])
    fuel_season = np.array([season_i[str(s)] for s in season_arr], dtype=np.int64)

    # who races where: from the laps, and for the forecast race from its entry list
    present = np.zeros((len(teams), len(events)))
    present[team_arr[train], ev_arr[train]] = 1.0
    ds_present = np.zeros((len(ds_rows), len(events)))
    ds_present[ds_lap[train], ev_arr[train]] = 1.0
    entry_ds = np.array(
        [ds_i[r] for r in zip(ent_drv.tolist(), ent_season.tolist(), strict=True)], dtype=np.int64
    )
    if e_next is not None:
        present[ent_team, e_next] = 1.0
        ds_present[entry_ds, e_next] = 1.0

    # race order: events at 0..E-1 (the forecast race, if any, is the last)
    hsgp = hsgp_basis(np.arange(len(events), dtype=np.float64))
    circuits = order["circuit_id"].to_list()
    feats = circuit_features(tables)
    fmap = {str(r[0]): np.array(r[1:], dtype=np.float64) for r in feats.iter_rows()}
    zero = np.zeros(len(CIRCUIT_FEATURES))
    circuit_x = np.vstack([fmap.get(c, zero) for c in circuits])

    hinges, drs = gap_basis(df["gap_ahead_s"].to_numpy())
    return WindowData(
        events=events,
        event_season=order["season"].to_numpy().astype(np.int64),
        event_center=event_center,
        event_circuit=circuits,
        teams=teams,
        drivers=drivers,
        seasons=seasons,
        y=y,
        train=train,
        ev=ev_arr,
        team=team_arr,
        driver=drv_arr,
        ts=np.array(
            [ts_i[r] for r in zip(team_arr.tolist(), season_arr.tolist(), strict=True)],
            dtype=np.int64,
        ),
        ds=ds_lap,
        de=de,
        off=off,
        wear=wear,
        age=ispline_basis(df["tyre_age"].to_numpy()),
        tyre_age=df["tyre_age"].to_numpy(),
        fuel=df["fuel"].to_numpy(),
        fuel_season=fuel_season,
        evo=evolution_basis(df["u"].to_numpy()),
        gap=hinges,
        drs=drs,
        wet=df["wet"].to_numpy(),
        pit=df["pit"].to_numpy().astype(np.int64),
        ts_team=ts_team,
        ts_season=ts_season,
        ts_walk=walk_matrix(ts_team, ts_season),
        ds_driver=ds_driver,
        ds_season=ds_season,
        ds_walk=walk_matrix(ds_driver, ds_season),
        ds_rookie=ds_rookie,
        off_compound=np.array([r[0] for r in off_rows], dtype=np.int64),
        off_ref=np.array([comp_i[ref[r[1]]] for r in off_rows], dtype=np.int64),
        off_event=np.array([r[1] for r in off_rows], dtype=np.int64),
        wear_compound=np.array([r[0] for r in wear_rows], dtype=np.int64),
        wear_event=np.array([r[1] for r in wear_rows], dtype=np.int64),
        de_driver=np.array([r[0] for r in de_rows], dtype=np.int64),
        de_event=np.array([r[1] for r in de_rows], dtype=np.int64),
        team_present=present,
        ds_present=ds_present,
        hsgp=hsgp,
        circuit_x=circuit_x,
        entries=[(drivers[d], teams[t]) for d, t in zip(ent_drv, ent_team, strict=True)],
        entry_team=ent_team,
        entry_ds=entry_ds,
        next_index=e_next,
        meta={"next_event": next_event, "next_season": next_season},
    )
