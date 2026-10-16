"""``lap_v2`` — the full Bayesian lap model (plan M6 §A), sampled with MCMC.

The model is written once in PyMC over the arrays of :mod:`f1p.models.lap_data` and can be
sampled by several engines (the bake-off of §A.5):

* ``nutpie`` — Rust NUTS on the compiled PyMC graph (``nutpie-flow`` adds its normalising-flow
  mass-matrix adaptation, ``nutpie-mclmc`` its microcanonical Langevin sampler);
* ``numpyro`` — NUTS on JAX with vectorised chains;
* ``blackjax`` — NUTS on JAX through BlackJAX;
* ``pymc`` — PyMC's own NUTS (reference).

Every parameterisation is non-centred (hierarchical scales multiply standard normals), the
likelihood is Student-t with a learned nu, and computation is ``float64``. The model exposes a
few deterministic quantities the rest of the system reads (car pace and driver levels at the
end of the window, pit loss, wear curves); :func:`car_pace_draws` turns them into each car's
pace at the next race — a posterior *distribution*, not a point and a Laplace sd.

The samplers are an optional extra: ``uv sync --extra bayes``.
"""

from __future__ import annotations

import importlib
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from f1p.models.lap_data import (
    CIRCUIT_FEATURES,
    COMPOUNDS,
    GAP_HINGES,
    OFFSET_COMPOUNDS,
    WindowData,
    helmert,
    zero_sum_basis,
)

if TYPE_CHECKING:  # pragma: no cover
    import pymc as pm

FloatArray = NDArray[np.float64]
Sampler = Literal["nutpie", "nutpie-flow", "nutpie-mclmc", "numpyro", "blackjax", "pymc"]
SAMPLERS: tuple[str, ...] = ("nutpie", "nutpie-flow", "nutpie-mclmc", "numpyro", "blackjax", "pymc")


@dataclass(frozen=True, slots=True)
class V2Priors:
    """Prior scales on the log-lap-time scale (0.01 = 1% of a lap). Hyper-scales are
    half-normal with these scales; fixed values are marked."""

    base: float = 0.02  # event base around the event's median lap (y is centred)
    car0: float = 0.02  # fixed: a team's first level in the window
    car_reset: float = 0.012  # season-to-season step of a team's level (new cars)
    gp_amplitude: float = 0.004  # within-window car form (upgrades, development)
    gp_ls_alpha: float = 5.0  # inverse-gamma lengthscale prior, in races: mean beta/(alpha-1) = 10
    gp_ls_beta: float = 40.0
    car_event: float = 0.004  # a team's race-specific deviation (track fit beyond the features)
    circuit: float = 0.003  # team x circuit-feature slopes
    driver0: float = 0.006  # driver level at first appearance in the window
    driver_step: float = 0.003  # season-to-season driver step
    rookie: float = 0.004
    driver_wet: float = 0.01
    compound: float = 0.03
    compound_event: float = 0.01
    wear_log_mean: float = -5.5  # log of each I-spline weight: e^-5.5 ≈ 0.4% of a lap
    wear_log_sd: float = 1.0
    wear_event: float = 0.3  # log-scale spread of an event's wear rate (track abrasiveness)
    fuel_mean: float = 0.03  # effect of 50 laps of fuel
    fuel_sd: float = 0.03
    fuel_season: float = 0.01  # fixed
    evolution: float = 0.005
    evolution_event: float = 0.003
    # plan §A.5: the first term to cut if the sampler cannot resolve it (it trades off against
    # the wear of the first stint, where tyre age and race lap move together)
    track_evolution: bool = True
    traffic: float = 0.01
    pit_in: tuple[float, float] = (0.05, 0.05)  # (mean, sd) of the in-lap offset
    pit_out: tuple[float, float] = (0.20, 0.10)
    pit_event: float = 0.03
    sigma_log_mean: float = -5.1  # e^-5.1 ≈ 0.6% of a lap
    sigma_log_sd: float = 0.5
    sigma_spread: float = 0.3
    pit_noise_log_mean: float = 1.4  # pit laps are ~4x noisier
    nu_rate: float = 0.1  # nu = 2 + Exponential(rate)


# -- the model -------------------------------------------------------------------------------
def _team_event_rows(w: WindowData) -> tuple[NDArray[np.int64], FloatArray]:
    """For every (team, event): the team-season row (0 where the team has no laps that
    season) and a mask of the pairs that exist."""
    t_n, e_n = w.team_present.shape
    rows = np.zeros((t_n, e_n), dtype=np.int64)
    ok = np.zeros((t_n, e_n))
    lookup = {
        (int(t), int(s)): i for i, (t, s) in enumerate(zip(w.ts_team, w.ts_season, strict=True))
    }
    for t in range(t_n):
        for e in range(e_n):
            i = lookup.get((t, int(w.event_season[e])))
            if i is not None:
                rows[t, e], ok[t, e] = i, 1.0
    return rows, ok


def _structure_labels(members: list[str], n: int) -> list[str]:
    import hashlib

    tag = hashlib.sha256("|".join(members).encode()).hexdigest()[:8]
    return [f"{tag}:{j}" for j in range(n)]


def _first_mask(entity: NDArray[np.int64]) -> FloatArray:
    return np.r_[1.0, (np.diff(entity) != 0).astype(np.float64)]


def build_model(w: WindowData, priors: V2Priors | None = None) -> pm.Model:
    # PyTensor's operator overloads are untyped: work with the modules as ``Any``
    pm: Any = importlib.import_module("pymc")
    pt: Any = importlib.import_module("pytensor.tensor")

    pr = priors or V2Priors()
    m = len(w.hsgp.sqrt_eig)
    te_rows, te_ok = _team_event_rows(w)
    present = w.team_present
    ts_first = _first_mask(w.ts_team)
    # zero-sum parameterisations: car and driver levels are re-centred at every event, so
    # their common level is invisible to the likelihood; sampling it would be a ridge
    h_team = helmert(len(w.teams))
    ts_basis = zero_sum_basis(w.ts_season)
    ds_basis = zero_sum_basis(w.ds_season)
    ts_labels = [f"{w.teams[t]}:{s}" for t, s in zip(w.ts_team, w.ts_season, strict=True)]
    ds_labels = [f"{w.drivers[d]}:{s}" for d, s in zip(w.ds_driver, w.ds_season, strict=True)]
    ds_first = _first_mask(w.ds_driver)
    off_idx = np.where(w.off < 0, len(w.off_compound), w.off)  # -1 → the appended zero
    pit_idx = np.clip(w.pit - 1, 0, 1)
    # per-race mean over racing laps (held-out laps included: their covariates are known in
    # advance, their times never enter), as an O(n) segment sum — a race's laps are contiguous
    starts = np.flatnonzero(np.r_[True, np.diff(w.ev) != 0])
    if len(starts) != len(np.unique(w.ev)):
        raise ValueError("laps must be grouped by race")
    ends = np.r_[starts[1:], len(w.ev)]
    racing_w = (w.pit == 0).astype(np.float64)
    seg_n = np.add.reduceat(racing_w, starts)
    seg_ev = w.ev[starts]  # the race of each segment
    is_pit = (w.pit > 0).astype(np.float64)
    coords = {
        "event": w.events,
        "team": w.teams,
        "driver": w.drivers,
        "season": [str(s) for s in w.seasons],
        # zero-sum bases: a coordinate means something only within one team/season structure,
        # so the labels carry that structure (a warm start transfers them only when it matches)
        "team_free": _structure_labels(w.teams, len(w.teams) - 1),
        "ts_free": _structure_labels(ts_labels, ts_basis.shape[1]),
        "ds_free": _structure_labels(ds_labels, ds_basis.shape[1]),
        "team_season": ts_labels,
        "driver_season": ds_labels,
        "basis": [f"phi{j}" for j in range(m)],
        "circuit_feature": list(CIRCUIT_FEATURES),
        "offset_compound": list(OFFSET_COMPOUNDS),
        "offset_row": [
            f"{COMPOUNDS[c]}-{COMPOUNDS[r]}:{w.events[e]}"
            for c, r, e in zip(w.off_compound, w.off_ref, w.off_event, strict=True)
        ],
        "tyre": list(COMPOUNDS),
        "wear_row": [
            f"{COMPOUNDS[c]}:{w.events[e]}"
            for c, e in zip(w.wear_compound, w.wear_event, strict=True)
        ],
        "age_basis": np.arange(w.age.shape[1]),
        "evo_basis": np.arange(w.evo.shape[1]),
        "gap_hinge": [f"{h}s" for h in GAP_HINGES],
        "pit_kind": ["in", "out"],
        "driver_event": [
            f"{w.drivers[d]}:{w.events[e]}" for d, e in zip(w.de_driver, w.de_event, strict=True)
        ],
    }
    train = np.flatnonzero(w.train)
    hold = np.flatnonzero(~w.train)
    if len(hold):
        coords["holdout_lap"] = np.arange(len(hold))
    if len(w.entries):
        coords["entry"] = [f"{d}:{t}" for d, t in w.entries]
    with pm.Model(coords=coords) as model:
        y = pm.Data("y", w.y[train])

        base = pm.Normal("base", 0.0, pr.base, dims="event")

        # car: team-season levels walking across seasons + GP over race order + circuit fit
        car_reset = pm.HalfNormal("car_reset", pr.car_reset)
        ts_z = pt.dot(ts_basis, pm.Normal("ts_z", 0.0, 1.0, dims="ts_free"))
        ts_scale = ts_first * pr.car0 + (1.0 - ts_first) * car_reset
        ts_level = pm.Deterministic(
            "ts_level", pt.dot(w.ts_walk, ts_scale * ts_z), dims="team_season"
        )
        gp_amp = pm.HalfNormal("gp_amplitude", pr.gp_amplitude)
        gp_ls = pm.InverseGamma("gp_lengthscale", pr.gp_ls_alpha, pr.gp_ls_beta)
        lam = w.hsgp.sqrt_eig
        psd = gp_amp**2 * 12.0 * np.sqrt(3.0) / gp_ls**3 / (3.0 / gp_ls**2 + lam**2) ** 2
        gp_z = pt.dot(h_team, pm.Normal("gp_z", 0.0, 1.0, dims=("team_free", "basis")))
        gp = pt.dot(gp_z * pt.sqrt(psd), w.hsgp.phi.T)  # (team, event)
        circ_sd = pm.HalfNormal("circuit_sd", pr.circuit)
        circ_z = pt.dot(
            h_team, pm.Normal("circuit_z", 0.0, 1.0, dims=("team_free", "circuit_feature"))
        )
        circ = pt.dot(circ_sd * circ_z, w.circuit_x.T)  # (team, event)
        ce_sd = pm.HalfNormal("car_event_sd", pr.car_event)
        ce_z = pt.dot(h_team, pm.Normal("car_event_z", 0.0, 1.0, dims=("team_free", "event")))
        car = ts_level[te_rows] + gp + circ + ce_sd * ce_z
        field = (car * present).sum(axis=0) / np.maximum(present.sum(axis=0), 1.0)
        car_c = pm.Deterministic("car", (car - field) * te_ok, dims=("team", "event"))

        # driver: driver-season levels walking across seasons, rookie offset, wet skill
        d0 = pm.HalfNormal("driver0", pr.driver0)
        d_step = pm.HalfNormal("driver_step", pr.driver_step)
        ds_z = pt.dot(ds_basis, pm.Normal("ds_z", 0.0, 1.0, dims="ds_free"))
        rookie = pm.Normal("rookie", 0.0, pr.rookie)
        ds_scale = ds_first * d0 + (1.0 - ds_first) * d_step
        ds_level = pm.Deterministic(
            "ds_level",
            pt.dot(w.ds_walk, ds_scale * ds_z) + rookie * w.ds_rookie,
            dims="driver_season",
        )
        # centred per event like the car (the event base carries the field's level)
        d_field = pt.dot(ds_level, w.ds_present) / np.maximum(w.ds_present.sum(axis=0), 1.0)
        wet_sd = pm.HalfNormal("driver_wet_sd", pr.driver_wet)
        wet_z = pm.Normal("driver_wet_z", 0.0, 1.0, dims="driver")

        # tyres: compound offsets and monotone wear curves per compound x event
        # offsets: within-race contrasts against the race's reference compound, shrunk towards
        # the difference of global levels (MEDIUM = 0)
        comp_g = pm.Normal("compound", 0.0, pr.compound, dims="offset_compound")
        level = pt.concatenate([pt.zeros(1), comp_g])  # COMPOUNDS order, MEDIUM first
        comp_sd = pm.HalfNormal("compound_event_sd", pr.compound_event)
        comp_z = pm.Normal("compound_z", 0.0, 1.0, dims="offset_row")
        contrast = level[w.off_compound] - level[w.off_ref] + comp_sd * comp_z
        offsets = pt.concatenate([contrast, pt.zeros(1)])
        wear_g = pm.Normal("wear_log", pr.wear_log_mean, pr.wear_log_sd, dims=("tyre", "age_basis"))
        # one curve shape per compound, scaled per event by a wear rate (track abrasiveness)
        wear_sd = pm.HalfNormal("wear_event_sd", pr.wear_event)
        wear_z = pm.Normal("wear_z", 0.0, 1.0, dims="wear_row")
        wear_w = pm.Deterministic(
            "wear_weights",
            pt.exp(wear_g[w.wear_compound] + (wear_sd * wear_z)[:, None]),
            dims=("wear_row", "age_basis"),
        )

        # fuel, track evolution, traffic, pit laps
        # per-season fuel slopes around a global one with a *fixed* spread: a window holds one
        # to three seasons, too few to learn a variance (a learned one makes a funnel)
        fuel_g = pm.Normal("fuel", pr.fuel_mean, pr.fuel_sd)
        fuel_s = fuel_g * pt.ones(len(w.seasons))
        if len(w.seasons) > 1:  # season deviations sum to zero around the global slope
            fuel_z = pm.Normal("fuel_z", 0.0, 1.0, shape=len(w.seasons) - 1)
            fuel_s = fuel_s + pr.fuel_season * pt.dot(helmert(len(w.seasons)), fuel_z)
        evolution = pt.zeros(len(w.y))
        if pr.track_evolution:
            evo_g = pm.Normal("evolution", 0.0, pr.evolution, dims="evo_basis")
            evo_sd = pm.HalfNormal("evolution_event_sd", pr.evolution_event)
            evo_z = pm.Normal("evolution_z", 0.0, 1.0, dims=("event", "evo_basis"))
            evolution = (w.evo * (evo_g + evo_sd * evo_z)[w.ev]).sum(axis=1)
        gap_b = pm.Normal("gap", 0.0, pr.traffic, dims="gap_hinge")
        drs_b = pm.Normal("drs", 0.0, pr.traffic)
        pit_g = pm.Normal(
            "pit",
            np.array([pr.pit_in[0], pr.pit_out[0]]),
            np.array([pr.pit_in[1], pr.pit_out[1]]),
            dims="pit_kind",
        )
        pit_sd = pm.HalfNormal("pit_event_sd", pr.pit_event)
        pit_z = pm.Normal("pit_z", 0.0, 1.0, dims=("pit_kind", "event"))
        pit_e = pm.Deterministic(
            "pit_event", pit_g[:, None] + pit_sd * pit_z, dims=("pit_kind", "event")
        )

        # everything that varies within a race, centred on the race's racing laps: the event
        # base is then the race's mean level, orthogonal to every global slope (otherwise each
        # slope moves all bases together — a ridge). An exact reparameterisation of the base.
        within = (
            car_c[w.team, w.ev]
            + ds_level[w.ds]
            - d_field[w.ev]
            + w.wet * wet_sd * wet_z[w.driver]
            + offsets[off_idx]
            + (w.age * wear_w[w.wear]).sum(axis=1)
            + w.fuel * fuel_s[w.fuel_season]
            + evolution
            + pt.dot(w.gap, gap_b)
            + w.drs * drs_b
            + is_pit * pit_e[pit_idx, w.ev]
        )
        cs = pt.concatenate([pt.zeros(1), pt.cumsum(within * racing_w)])
        seg_mean = (cs[ends] - cs[starts]) / np.maximum(seg_n, 1.0)
        race_mean = pt.zeros(len(w.events))[seg_ev].set(seg_mean)
        mu = base[w.ev] + within - race_mean[w.ev]

        # noise: a scale per driver x event, wider on pit laps; Student-t with learned nu
        sig_mu = pm.Normal("sigma_log", pr.sigma_log_mean, pr.sigma_log_sd)
        sig_sd = pm.HalfNormal("sigma_spread", pr.sigma_spread)
        sig_z = pm.Normal("sigma_z", 0.0, 1.0, dims="driver_event")
        pit_noise = pm.Normal("pit_noise_log", pr.pit_noise_log_mean, 0.5)
        sigma = pt.exp(sig_mu + sig_sd * sig_z[w.de] + is_pit * pit_noise)
        nu_extra = pm.Exponential("nu_minus_2", pr.nu_rate)
        pm.StudentT(
            "obs", nu=2.0 + nu_extra, mu=mu[train], sigma=sigma[train], observed=y, shape=y.shape
        )

        # the forecast race: each entry's pace (car + driver, centred over the field) and, for
        # validation, the predictive location and scale of its held-out laps
        if w.next_index is not None and len(w.entries):
            e = w.next_index
            pm.Deterministic(
                "pace_next",
                car_c[w.entry_team, e] + ds_level[w.entry_ds] - d_field[e],
                dims="entry",
            )
        if len(hold):
            pm.Deterministic("holdout_mu", mu[hold], dims="holdout_lap")
            pm.Deterministic("holdout_sigma", sigma[hold], dims="holdout_lap")
    return model


# -- sampling ------------------------------------------------------------------------------------
@dataclass
class SampleConfig:
    sampler: str = "nutpie"
    draws: int = 1000
    tune: int = 1000
    chains: int = 4
    cores: int | None = None
    seed: int = 7
    target_accept: float = 0.9
    adaptation: str = "diag"  # nutpie mass matrix: diag or low_rank


@dataclass
class Diagnostics:
    max_rhat: float
    min_ess_bulk: float
    min_ess_tail: float
    divergences: int
    min_bfmi: float
    wall_s: float
    sampler: str
    ess_per_s: float  # min bulk ESS of the key quantities per second of sampling
    worst: list[str] = field(default_factory=list)
    compile_s: float = 0.0
    key_rhat: float = float("nan")  # max R-hat over KEY_VARS (what forecasts read)
    rhat_by_var: dict[str, float] = field(default_factory=dict)  # max R-hat per KEY_VARS name

    @property
    def ok(self) -> bool:
        """Plan §A.4: R-hat < 1.01, bulk/tail ESS > 400, no divergences, E-BFMI > 0.3."""
        return (
            self.max_rhat < 1.01
            and self.min_ess_bulk > 400
            and self.min_ess_tail > 400
            and self.divergences == 0
            and self.min_bfmi > 0.3
        )

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "ok": self.ok}


def _enable_x64() -> None:
    import jax

    jax.config.update("jax_enable_x64", True)


def sample(
    model: pm.Model,
    cfg: SampleConfig | None = None,
    timings: dict[str, float] | None = None,
    *,
    warm: Unconstrained | None = None,
) -> Any:
    """Posterior draws (an arviz ``DataTree``) from the chosen engine. ``timings`` receives
    ``compile_s`` and ``sample_s`` (ESS/s is reported on sampling time only)."""
    pm: Any = importlib.import_module("pymc")
    cfg = cfg or SampleConfig()
    if cfg.sampler not in SAMPLERS:
        raise ValueError(f"unknown sampler {cfg.sampler!r}; choose from {SAMPLERS}")
    times = timings if timings is not None else {}
    t0 = time.perf_counter()
    if cfg.sampler.startswith("nutpie"):
        nutpie: Any = importlib.import_module("nutpie")
        kwargs: dict[str, Any] = {}
        if cfg.sampler == "nutpie-flow":
            _enable_x64()
            compiled = nutpie.compile_pymc_model(model, backend="jax", gradient_backend="jax")
            kwargs["adaptation"] = "flow"
        else:
            compiled = nutpie.compile_pymc_model(model, backend="numba")
            kwargs["adaptation"] = cfg.adaptation
        if cfg.sampler == "nutpie-mclmc":
            kwargs["sampler"] = "mclmc"
        else:
            kwargs["target_accept"] = cfg.target_accept
        if warm is not None:
            kwargs["init_mean"] = warm_init(compiled, model, warm)
        t1 = time.perf_counter()
        times["compile_s"] = t1 - t0
        trace = nutpie.sample(
            compiled,
            store_unconstrained=True,
            draws=cfg.draws,
            tune=cfg.tune,
            chains=cfg.chains,
            cores=cfg.cores or cfg.chains,
            seed=cfg.seed,
            progress_bar=False,
            save_warmup=False,
            **kwargs,
        )
        times["sample_s"] = time.perf_counter() - t1
        return trace
    common: dict[str, Any] = {
        "draws": cfg.draws,
        "tune": cfg.tune,
        "chains": cfg.chains,
        "random_seed": cfg.seed,
        "progressbar": False,
    }
    if cfg.sampler in ("numpyro", "blackjax"):
        _enable_x64()
        nuts_kwargs: dict[str, Any] = {"target_accept": cfg.target_accept}
        if cfg.sampler == "numpyro":
            nuts_kwargs["chain_method"] = "vectorized"
        common |= {"nuts_sampler": cfg.sampler, "nuts_sampler_kwargs": nuts_kwargs}
    else:
        common |= {"cores": cfg.cores or cfg.chains, "target_accept": cfg.target_accept}
    with model:
        trace = pm.sample(**common)
    # JAX and PyMC compile inside ``pm.sample``: their time is all reported as sampling
    times["compile_s"] = 0.0
    times["sample_s"] = time.perf_counter() - t0
    return trace


# -- warm starts (plan §A.3, sequential tier) ----------------------------------------------------
# value var → (labels per axis, posterior mean in the sampler's space)
Unconstrained = dict[str, tuple[list[list[str]], FloatArray]]


def summarise_unconstrained(trace: Any, model: pm.Model) -> Unconstrained:
    """Posterior mean of every free variable in the sampler's (unconstrained) space, with the
    coordinate labels of each axis."""
    out: Unconstrained = {}
    unc = trace.get("unconstrained_posterior", None)
    post = trace["posterior"]
    for value in model.value_vars:
        rv = model.values_to_rvs[value]
        name = value.name
        if unc is not None and name in unc.data_vars:
            arr = np.asarray(unc[name], dtype=np.float64)
        elif rv.name in post.data_vars:
            arr = np.asarray(post[rv.name], dtype=np.float64)
        else:
            continue
        dims = model.named_vars_to_dims.get(rv.name, ())
        labels = [[str(c) for c in model.coords[d] or ()] for d in dims]
        out[name] = (labels, arr.mean(axis=(0, 1)))
    return out


def _map_labels(
    new: list[list[str]], old: list[list[str]], values: FloatArray, default: FloatArray
) -> FloatArray:
    """``default`` with every entry whose labels exist in ``old`` replaced by its value."""
    if len(new) != len(old) or values.ndim != len(old) or default.ndim != len(new):
        return default
    out = default.copy()
    src, dst = [], []
    for n_lab, o_lab in zip(new, old, strict=True):
        pos = {lab: i for i, lab in enumerate(o_lab)}
        pairs = [(i, pos[lab]) for i, lab in enumerate(n_lab) if lab in pos]
        if not pairs:
            return default
        dst.append(np.array([p[0] for p in pairs]))
        src.append(np.array([p[1] for p in pairs]))
    out[np.ix_(*dst)] = values[np.ix_(*src)]
    return out


def warm_init(compiled: Any, model: pm.Model, prev: Unconstrained) -> FloatArray:
    """nutpie's ``init_mean`` for ``model``: the earlier posterior's means where the labels
    match (a race or driver in both windows keeps its value), the model's own initial point
    elsewhere. A mismatch only costs speed: it is a starting point, not a constraint."""
    names, slices, _ = compiled.shape_info
    ip = model.initial_point()
    init = np.zeros(compiled.n_dim)
    for name, sl in zip(names, slices, strict=True):
        if sl.stop > compiled.n_dim:
            continue  # beyond the free parameters (expanded deterministics)
        default = np.asarray(ip[name], dtype=np.float64)
        if name in prev:
            labels, values = prev[name]
            value = next(v for v in model.value_vars if v.name == name)
            rv = model.values_to_rvs[value]
            dims = model.named_vars_to_dims.get(rv.name, ())
            new_labels = [[str(c) for c in model.coords[d] or ()] for d in dims]
            default = _map_labels(new_labels, labels, values, default)
        init[sl] = default.ravel()
    return init


KEY_VARS = ("pace_next", "car_reset", "gp_amplitude", "car_event_sd", "sigma_log")


def diagnose(
    trace: Any, wall_s: float, sampler: str, *, var_names: tuple[str, ...] | None = None
) -> Diagnostics:
    """Convergence checks over every free parameter and ESS/s on the car-pace quantities."""
    import arviz as az

    post = trace["posterior"]
    names = list(var_names) if var_names else [v for v in post.data_vars if not v.endswith("_z")]
    rhat = az.rhat(trace, var_names=names)
    ess_b = az.ess(trace, var_names=names, method="bulk")
    ess_t = az.ess(trace, var_names=names, method="tail")

    def _flat(ds: Any) -> dict[str, float]:
        out = {}
        for v in ds.data_vars:
            arr = np.asarray(ds[v]).ravel()
            arr = arr[np.isfinite(arr)]
            if arr.size:
                out[str(v)] = float(arr.max() if ds is rhat else arr.min())
        return out

    r, eb, et = _flat(rhat), _flat(ess_b), _flat(ess_t)
    div = 0
    bfmi = float("nan")
    if "sample_stats" in trace:
        stats = trace["sample_stats"]
        for key in ("diverging", "divergences"):
            if key in stats.data_vars:
                div = int(np.asarray(stats[key]).sum())
                break
        try:
            b = az.bfmi(trace)  # a Dataset: one energy value per chain
            vals = [np.asarray(b[v]) for v in b.data_vars] if hasattr(b, "data_vars") else [b]
            bfmi = float(np.min(np.concatenate([np.ravel(v) for v in vals])))
        except Exception:  # samplers without energy (MCLMC)
            bfmi = float("nan")
    key = [v for v in KEY_VARS if v in eb]
    key_ess = min((eb[v] for v in key), default=float("nan"))
    key_rhat = max((r[v] for v in key if v in r), default=float("nan"))
    worst = sorted(r, key=lambda v: -r[v])[:3]
    return Diagnostics(
        max_rhat=max(r.values(), default=float("nan")),
        min_ess_bulk=min(eb.values(), default=float("nan")),
        min_ess_tail=min(et.values(), default=float("nan")),
        divergences=div,
        min_bfmi=bfmi,
        wall_s=round(wall_s, 2),
        sampler=sampler,
        ess_per_s=round(key_ess / max(wall_s, 1e-9), 3),
        worst=worst,
        key_rhat=key_rhat,
        rhat_by_var={v: r[v] for v in key if v in r},
    )


@dataclass
class LapPosterior:
    """Thinned posterior draws of the quantities downstream models need."""

    data: WindowData
    draws: dict[str, FloatArray]  # name → (draws, *shape)
    diagnostics: Diagnostics
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_draws(self) -> int:
        return len(next(iter(self.draws.values())))


KEEP = (
    "pace_next",
    "holdout_mu",
    "holdout_sigma",
    "nu_minus_2",
    "car",
    "car_reset",
    "car_event_sd",
    "gp_amplitude",
    "gp_lengthscale",
    "driver0",
    "driver_step",
    "rookie",
    "pit",
    "pit_event_sd",
    "pit_event",
    "wear_log",
    "wear_event_sd",
    "wear_weights",
    "compound",
    "compound_event_sd",
    "fuel",
    "sigma_log",
    "sigma_spread",
    "pit_noise_log",
    "gap",
    "drs",
    "base",
)


def fit(
    w: WindowData,
    cfg: SampleConfig | None = None,
    *,
    priors: V2Priors | None = None,
    keep: int = 1000,
    warm: LapPosterior | None = None,
) -> LapPosterior:
    """Sample the window and keep ``keep`` evenly thinned draws of :data:`KEEP`.

    ``warm`` (the posterior of an earlier, overlapping window) starts the sampler at that
    posterior's mean, matched by label (plan §A.3, sequential tier): with nutpie, tuning can
    then be much shorter."""
    cfg = cfg or SampleConfig()
    model = build_model(w, priors)
    timings: dict[str, float] = {}
    start = warm.meta.get("unconstrained") if warm is not None else None
    trace = sample(model, cfg, timings, warm=start)
    diag = diagnose(trace, timings["sample_s"], cfg.sampler)
    diag.compile_s = round(timings["compile_s"], 2)
    post = trace["posterior"]
    draws: dict[str, FloatArray] = {}
    for name in KEEP:
        if name in post.data_vars:
            arr = np.asarray(post[name], dtype=np.float64)
            flat = arr.reshape(-1, *arr.shape[2:])
            idx = np.linspace(0, len(flat) - 1, min(keep, len(flat))).round().astype(int)
            draws[name] = flat[idx]
    meta: dict[str, Any] = {"sampler": cfg.sampler, "draws": cfg.draws, "warm": start is not None}
    if cfg.sampler.startswith("nutpie"):
        meta["unconstrained"] = summarise_unconstrained(trace, model)
    return LapPosterior(w, draws, diag, meta)


# -- what the forecast reads ----------------------------------------------------------------------
def car_pace_draws(post: LapPosterior, drivers: list[str]) -> FloatArray:
    """Posterior draws (draws x cars) of each car's log-lap-time effect at the forecast race
    (car + driver, centred over the field; lower is faster), in the order of ``drivers``.
    NaN for a driver not on the window's entry list."""
    w = post.data
    pace = post.draws["pace_next"]
    col = {d: i for i, (d, _) in enumerate(w.entries)}
    out = np.full((post.n_draws, len(drivers)), np.nan)
    for k, d in enumerate(drivers):
        if d in col:
            out[:, k] = pace[:, col[d]]
    return out


def holdout_log_density(post: LapPosterior) -> FloatArray:
    """Log pointwise predictive density of every held-out lap: log of the mean over draws of
    the Student-t density (the elpd terms of plan §A.4)."""
    from scipy.special import logsumexp
    from scipy.stats import t as student_t

    w = post.data
    y = w.y[~w.train]
    mu, sd = post.draws["holdout_mu"], post.draws["holdout_sigma"]
    nu = 2.0 + post.draws["nu_minus_2"][:, None]
    lp = student_t.logpdf(y[None, :], nu, loc=mu, scale=sd)
    return logsumexp(lp, axis=0) - np.log(lp.shape[0])


def pit_loss_draws(post: LapPosterior, lap_s: float = 90.0) -> FloatArray:
    """Seconds lost to a stop at the window's average event (in-lap + out-lap offsets on a
    ``lap_s`` lap), per draw."""
    pe = post.draws["pit_event"]  # (draws, 2, events)
    return (np.expm1(pe[:, 0]) + np.expm1(pe[:, 1])).mean(axis=1) * lap_s


# -- persistence (plan §A.6) ----------------------------------------------------------------------
SAVED = tuple(k for k in KEEP if k not in ("holdout_mu", "holdout_sigma", "car", "base"))


def save_posterior(post: LapPosterior, path: Path, *, priors: V2Priors | None = None) -> Path:
    """Write the thinned draws downstream models need as ``<path>.parquet`` (one column per
    scalar, one row per draw) and the window's labels and diagnostics as ``<path>.json``."""
    import json

    import polars as pl

    cols: dict[str, FloatArray] = {}
    shapes: dict[str, list[int]] = {}
    for name in SAVED:
        if name not in post.draws:
            continue
        arr = post.draws[name]
        shapes[name] = list(arr.shape[1:])
        flat = arr.reshape(len(arr), -1)
        for j in range(flat.shape[1]):
            cols[f"{name}[{j}]" if flat.shape[1] > 1 or arr.ndim > 1 else name] = flat[:, j]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(cols).write_parquet(path.with_suffix(".parquet"))
    w = post.data
    meta = {
        "events": w.events,
        "entries": w.entries,
        "teams": w.teams,
        "drivers": w.drivers,
        "seasons": w.seasons,
        "next_event": w.meta.get("next_event"),
        "shapes": shapes,
        "diagnostics": post.diagnostics.to_dict(),
        "priors": asdict(priors or V2Priors()),
        **{k: v for k, v in post.meta.items() if isinstance(v, str | int | float)},
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=str))
    return path.with_suffix(".parquet")


def load_draws(path: Path) -> tuple[dict[str, FloatArray], dict[str, Any]]:
    """Read :func:`save_posterior` output back: draws by name (with their shapes) and the
    metadata."""
    import json

    import polars as pl

    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    df = pl.read_parquet(path.with_suffix(".parquet"))
    draws: dict[str, FloatArray] = {}
    for name, shape in meta["shapes"].items():
        if not shape:
            draws[name] = df[name].to_numpy()
            continue
        k = int(np.prod(shape))
        block = np.column_stack([df[f"{name}[{j}]"].to_numpy() for j in range(k)])
        draws[name] = block.reshape(len(df), *shape)
    return draws, meta


# -- the lap_pace v2 feature (plan §A.6/§A.7) -----------------------------------------------------
class PosteriorStore:
    """Saved ``lap_v2`` posteriors, one per forecast race (``<root>/<event_id>.parquet``),
    read lazily: each car's centred pace at that race as a posterior mean and sd, in % of a
    lap (lower = faster)."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._cache: dict[str, dict[str, tuple[float, float]] | None] = {}

    def fingerprint(self) -> str:
        """Changes when posteriors are added or refitted (a cache key for prepared signals)."""
        import hashlib

        if not self.root.exists():
            return "empty"
        h = hashlib.sha256()
        for p in sorted(self.root.glob("*.parquet")):
            h.update(f"{p.name}:{p.stat().st_mtime_ns}".encode())
        return h.hexdigest()[:16]

    def pace(self, event_id: str) -> dict[str, tuple[float, float]] | None:
        if event_id not in self._cache:
            self._cache[event_id] = self._read(event_id)
        return self._cache[event_id]

    def _read(self, event_id: str) -> dict[str, tuple[float, float]] | None:
        import json

        import polars as pl

        path = self.root / f"{event_id}.parquet"
        if not path.exists():
            return None
        meta = json.loads(path.with_suffix(".json").read_text())
        entries = [str(e[0]) for e in meta["entries"]]
        cols = [f"pace_next[{j}]" for j in range(len(entries))]
        draws = 100.0 * pl.read_parquet(path, columns=cols).to_numpy()
        draws = draws - draws.mean(axis=1, keepdims=True)  # centred per draw
        return {
            d: (float(draws[:, j].mean()), float(draws[:, j].std())) for j, d in enumerate(entries)
        }


def lap_pace_v2_signal(
    drivers: list[str], event_id: str, store: PosteriorStore
) -> tuple[FloatArray, FloatArray]:
    """Two combiner columns for the cars of a race: the posterior mean pace advantage (% of a
    lap, centred, higher = faster) and that mean times its posterior sd, so a linear
    combiner can learn to trust uncertain pace less (effective weight ``w1 + w2 * sd``).
    Zeros when the race has no stored posterior (before 2018, or not fitted yet); a car
    missing from the posterior gets the field's mean and the median sd."""
    n = len(drivers)
    pace = store.pace(event_id)
    if not pace:
        return np.zeros(n), np.zeros(n)
    sds = np.array([v[1] for v in pace.values()])
    med_sd = float(np.median(sds))
    mean = np.array([-pace[d][0] if d in pace else np.nan for d in drivers])
    sd = np.array([pace[d][1] if d in pace else med_sd for d in drivers])
    fill = float(np.nanmean(mean)) if np.isfinite(mean).any() else 0.0
    mean = np.where(np.isnan(mean), fill, mean)
    mean = mean - mean.mean()
    return mean, mean * sd
