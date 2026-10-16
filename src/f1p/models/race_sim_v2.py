"""``race_sim_v2`` as a :class:`~f1p.eval.context.Forecaster` (plan M6 §B.5).

Per race and checkpoint:

* **sub-models** (start, overtaking, strategy, incidents, compounds) fitted on the laps of
  earlier seasons only — refitted once a season, so every forecast is causal;
* **pace** from the stored ``lap_v2`` posterior of the race when there is one (one draw per
  simulated race), else from the Laplace lap model (mean and sd per car, linear wear);
* **grid**: the real grid at C4; at C1, grids sampled from ``quali_v1``'s C1 forecast;
* **reliability** from the world model's filter; **circuit** laps and base lap from the
  circuit's recent visits.

The forecast is the simulated finishing-order distribution, with its Plackett-Luce projection
for scoring and pooling, and the Monte-Carlo error of P(win) in the meta.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from f1p.eval.baselines import History, grid_ranks
from f1p.eval.context import EventContext
from f1p.eval.forecast import Forecast, position_matrix, stable_seed
from f1p.models import sim_fit as sf
from f1p.models import sim_v2 as sv
from f1p.models.lap_model import LapModelCache, car_pace
from f1p.models.race_sim import RaceSim, circuit_profile, implied_log_strength


@dataclass(frozen=True, slots=True)
class SimV2Signal:
    circuit: str
    laps: int
    base_lap_s: float
    drivers: list[str]
    teams: list[str]  # start-model labels: constructor:season
    p_finish: Any  # (n,)
    grid: Any | None  # C4: slots (n,)
    pace_mean: Any  # (n,) Laplace fallback
    pace_sd: Any
    deg: dict[str, float]
    offset: dict[str, float]
    fuel: float
    pit_loss_s: float
    posterior: dict[str, Any] | None  # lap_v2 draws for this race, if stored
    order: list[int] | None  # car → posterior entry
    wet: bool


@dataclass
class RaceSimV2:
    name: str = "race_sim_v2"
    checkpoints: frozenset[str] = frozenset({"C1", "C4"})
    targets: frozenset[str] = frozenset({"race"})
    n_sims: int = 20_000
    knobs: sv.Knobs = field(default_factory=sv.Knobs)
    # lap_v2 posteriors (relative to the artifacts directory unless absolute)
    lap_v2: str | None = "posteriors/lap_v2/default"
    _laps: LapModelCache = field(default_factory=LapModelCache, repr=False)
    _models: dict[int, sv.SubModels] = field(default_factory=dict, repr=False)
    _v1: RaceSim = field(default_factory=RaceSim, repr=False)
    _store: Any = field(default=None, repr=False)

    def prepare_signature(self) -> tuple[Any, ...]:
        store = self._lap_store()
        return (
            "race_sim_v2",
            self._laps.window,
            self._laps.priors,
            self._v1.prepare_signature(),
            self.lap_v2,
            store.fingerprint() if store is not None else None,
        )

    def _lap_store(self) -> Any:
        if self.lap_v2 is None:
            return None
        if self._store is None:
            from f1p.core.config import get_settings
            from f1p.models.lap_v2 import PosteriorStore

            root = Path(self.lap_v2)
            if not root.is_absolute():
                root = get_settings().paths.artifacts / root
            self._store = PosteriorStore(root)
        return self._store

    # -- sub-models, refitted once a season on earlier seasons ----------------------------------
    def sub_models(self, ctx: EventContext) -> sv.SubModels | None:
        if ctx.season not in self._models:
            ev = ctx.tables["events"].select("event_id", "season")
            laps = ctx.tables["laps"].join(ev, on="event_id")
            laps = laps.filter(pl.col("season") < ctx.season).drop("season")
            if laps.height < 5_000:
                return None
            tables = {k: v for k, v in ctx.tables.items() if k != "laps"}
            self._models[ctx.season] = sv.SubModels(
                sf.fit_overtaking(tables, laps),
                sf.fit_strategy(tables, laps),
                sf.fit_start(tables, laps),
                sf.fit_incidents(tables, laps),
                sf.fit_compounds(laps),
            )
        return self._models[ctx.season]

    # -- Forecaster protocol ---------------------------------------------------------------------
    def prepare(self, ctx: EventContext) -> SimV2Signal | None:
        if "laps" not in ctx.tables:
            return None
        fitted = self._laps.fit_for(ctx)
        if fitted is None:
            return None
        mean, var = car_pace(fitted, ctx, self._laps.priors)
        has = ~np.isnan(mean)
        if has.sum() < max(10, ctx.n // 2):
            return None
        fill, worst = float(np.nanmean(mean)), float(np.nanmax(mean))
        mean = np.where(has, mean, 0.5 * (fill + worst))  # newcomers: the slower half's average
        var = np.where(has, var, np.nanmax(var))
        grid = None
        if ctx.checkpoint == "C4":
            ranks = grid_ranks(ctx)
            if ranks is not None:
                grid = np.argsort(np.argsort(ranks, kind="stable"), kind="stable")
        known = ctx.known("laps")
        profile = circuit_profile(known, ctx.tables["events"], ctx.circuit_id)
        posterior, order = None, None
        store = self._lap_store()
        if store is not None and (store.root / f"{ctx.event_id}.parquet").exists():
            from f1p.models.lap_v2 import load_draws

            draws, meta = load_draws(store.root / ctx.event_id)
            entries = [str(e[0]) for e in meta["entries"]]
            col = {d: i for i, d in enumerate(entries)}
            if all(d in col for d in ctx.drivers):
                posterior, order = draws, [col[d] for d in ctx.drivers]
        from f1p.models.world_v1 import rain_signal

        world = self._v1._world_model()
        return SimV2Signal(
            ctx.circuit_id,
            profile.laps,
            profile.base_lap_s,
            list(ctx.drivers),
            [f"{c}:{ctx.season}" for c in ctx.constructors],
            world.raw_p_finish(ctx),
            grid,
            mean,
            np.sqrt(var),
            {c: fitted.degradation(c) for c in ("SOFT", "MEDIUM", "HARD")},
            {c: fitted.compound_offset(c) for c in ("SOFT", "HARD")},
            float(np.clip(fitted.value("fuel") * 50.0, 0.0, 0.1)),
            profile.pit_loss_s,
            posterior,
            order,
            rain_signal(ctx) > 0.5,
        )

    def forecast(self, ctx: EventContext, signal: Any, history: History) -> Forecast | None:
        del history  # sub-models come from earlier seasons' laps, not the score history
        if not isinstance(signal, SimV2Signal):
            return None
        models = self.sub_models(ctx)
        if models is None:
            return None
        rng = np.random.default_rng(stable_seed(ctx.event_id, ctx.checkpoint, self.name))
        s, n = self.n_sims, ctx.n
        spec, draws = self.race(ctx, signal, s, rng)
        res = sv.simulate(spec, draws, models, self.knobs, rng)
        pos = position_matrix(res.orders, n)
        wins = np.bincount(res.orders[:, 0], minlength=n).astype(float)
        p_win = (wins + 0.5) / (wins.sum() + 0.5 * n)
        mc_se = float(np.max(np.sqrt(p_win * (1 - p_win) / s)))
        return Forecast(
            ctx.event_id,
            ctx.checkpoint,
            self.name,
            ctx.drivers,
            pos,
            implied_log_strength(res.orders, n),
            {
                "sims": float(s),
                "p_win_mc_se": mc_se,
                "posterior": float(signal.posterior is not None),
                "neutral_periods": float(res.neutral_periods.mean()),
                "overtakes": float(res.overtakes.mean()),
            },
            p_win,
            res.finished.mean(axis=0),
        )

    def race(
        self, ctx: EventContext, signal: SimV2Signal, s: int, rng: np.random.Generator
    ) -> tuple[sv.RaceSpec, sv.Draws]:
        """The simulator's inputs for ``s`` races of ``ctx`` (also used by calibration)."""
        n = ctx.n
        if signal.grid is not None:
            grid = np.broadcast_to(np.asarray(signal.grid, dtype=np.int64), (s, n)).copy()
        else:
            grid = self._v1._grids(ctx, s, rng).astype(np.int64)
        if signal.posterior is not None and signal.order is not None:
            draws = sv.draws_from_posterior(
                signal.posterior, signal.order, s, signal.base_lap_s, rng
            )
        else:
            draws = sv.draws_from_laplace(
                np.asarray(signal.pace_mean),
                np.asarray(signal.pace_sd),
                s,
                rng,
                deg=signal.deg,
                offset=signal.offset,
                fuel=signal.fuel,
                pit_loss_s=signal.pit_loss_s,
            )
        spec = sv.RaceSpec(
            signal.circuit,
            signal.laps,
            signal.base_lap_s,
            signal.teams,
            np.asarray(signal.p_finish),
            grid,
            dry=not signal.wet,
        )
        return spec, draws


def load_knobs(path: Path | None = None) -> sv.Knobs:
    """The calibrated knobs (``f1p sim calibrate``) if saved, else the defaults."""
    import json

    from f1p.core.config import get_settings

    p = path or get_settings().paths.artifacts / "sim_v2" / "knobs.json"
    if not p.exists():
        return sv.Knobs()
    saved = json.loads(p.read_text())["knobs"]
    fields = set(sv.Knobs.__dataclass_fields__)
    return sv.Knobs(**{k: float(v) for k, v in saved.items() if k in fields})
