"""``lapseq`` as a forecaster (plan M6 §D.2): pre-race (C4) from the grid token, in the race
(C5) from the lap-``k`` token.

One network per forecast season, trained on the real races of the ``training_seasons``
before it and pretrained on ``race_sim_v2`` simulations of those same races (each simulated
with the sub-models and pace known before it — causal). Trained networks are kept under
``artifacts/lapseq/<signature>/<season>/``, keyed by the configuration and the tables'
content, and shared by worker processes (the first trains, the others wait and load).

A forecast's input never reads the race's classification: :func:`race_seq` builds it with
``outcome=False`` from the context's grid, the pace known before the race and, in the
race, the laps up to ``k`` (:func:`f1p.eval.in_race.race_so_far`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from f1p.core.logging import get_logger
from f1p.eval.baselines import History, grid_ranks
from f1p.eval.context import IN_RACE, EventContext, Tables, build_contexts, is_in_race
from f1p.eval.forecast import Forecast
from f1p.models import lapseq as lq
from f1p.models.lapseq_data import RaceSeq, Tokens, real_race, recent_pace, simulated_races
from f1p.models.lapseq_data import tokens as to_tokens

log = get_logger("models.lapseq_model")

MIN_TRAINING_RACES = 20  # fewer: no network for the season


@dataclass(frozen=True, slots=True)
class LapSeqSignal:
    x: np.ndarray  # (T, n, F)
    running: np.ndarray  # (T, n)

    def tokens(self) -> Tokens:
        t, n = self.running.shape
        nan = np.full((t, n), np.nan, dtype=np.float32)
        return Tokens(
            self.x,
            self.running,
            nan,
            nan,
            nan,
            np.full(t, np.nan, dtype=np.float32),
            np.full(n, -1, dtype=np.int16),
        )


def race_laps(ctx: EventContext) -> int:
    if "race_laps" in ctx.meta:
        return int(ctx.meta["race_laps"])
    ev = ctx.tables["events"].filter(pl.col("event_id") == ctx.event_id)
    return int(ev["laps"][0] or 0) if ev.height else 0


def race_seq(ctx: EventContext, *, outcome: bool) -> RaceSeq | None:
    """The race of ``ctx`` as a sequence: with ``outcome`` (training) all its laps and its
    classification; without (a forecast) the laps public at the checkpoint and no result."""
    total = race_laps(ctx)
    if total < 2 or "laps" not in ctx.tables:
        return None
    if outcome:
        laps = ctx.tables["laps"].filter(pl.col("event_id") == ctx.event_id)
        upto = None
    elif is_in_race(ctx.checkpoint):
        from f1p.eval.in_race import race_so_far

        laps = race_so_far(ctx)
        upto = int(ctx.meta["lap"])
    else:
        laps = ctx.tables["laps"].clear()
        upto = 1
    if outcome and laps.is_empty():
        return None
    results = ctx.tables["results"].filter(
        (pl.col("event_id") == ctx.event_id) & (pl.col("session") == "race")
    )
    ranks = grid_ranks(ctx)
    pace = (
        recent_pace(
            ctx.known("race_pace"), ctx.tables["events"], ctx.drivers, ctx.season, ctx.round
        )
        if "race_pace" in ctx.tables
        else {}
    )
    return real_race(
        laps,
        results,
        ctx.drivers,
        total,
        pace=pace,
        upto=upto,
        event_id=ctx.event_id,
        grid=None if ranks is None else ranks - 1.0,
        outcome=outcome,
    )


def training_contexts(tables: Tables, season: int, seasons: int) -> list[EventContext]:
    """C4 contexts of the races with lap timing in the ``seasons`` before ``season``."""
    if "laps" not in tables:
        return []
    timed = set(tables["laps"]["event_id"].unique().to_list())
    ctxs = build_contexts(tables, seasons=(season - seasons, season - 1), checkpoints=("C4",))
    return [c for c, _, _ in ctxs if c.event_id in timed]


def simulate_corpus(
    ctxs: list[EventContext], per_race: int, seed: int, knobs: Any = None
) -> list[RaceSeq]:
    """``per_race`` traced ``race_sim_v2`` races for each context (the pretraining set)."""
    from f1p.models import sim_v2 as sv
    from f1p.models.race_sim_v2 import RaceSimV2, SimV2Signal

    sim = RaceSimV2(knobs=knobs or sv.Knobs(), n_sims=per_race)
    out: list[RaceSeq] = []
    for i, ctx in enumerate(ctxs):
        signal = sim.prepare(ctx)
        models = sim.sub_models(ctx)
        if not isinstance(signal, SimV2Signal) or models is None:
            continue
        rng = np.random.default_rng([seed, i])
        spec, draws = sim.race(ctx, signal, per_race, rng)
        res = sv.simulate(spec, draws, models, sim.knobs, rng, trace=True)
        out += simulated_races(
            res.trace, res.orders, res.finished, spec.grid, draws.pace, spec.laps, rng
        )
    return out


@dataclass
class LapSeq:
    name: str = "lapseq"
    checkpoints: frozenset[str] = frozenset({"C4", *IN_RACE})
    targets: frozenset[str] = frozenset({"race"})
    config: lq.LapSeqConfig = field(default_factory=lq.LapSeqConfig)
    training_seasons: int = 8
    pretrain: bool = True
    min_races: int = MIN_TRAINING_RACES
    root: str = "lapseq"  # under the artifacts directory unless absolute
    _nets: dict[int, Any] = field(default_factory=dict, repr=False)
    _fingerprint: dict[int, str] = field(default_factory=dict, repr=False)

    def prepare_signature(self) -> tuple[Any, ...]:
        return ("lapseq-input", 1)  # the input does not depend on the network's settings

    def prepare(self, ctx: EventContext) -> LapSeqSignal | None:
        seq = race_seq(ctx, outcome=False)
        if seq is None:
            return None
        upto = int(ctx.meta["lap"]) if is_in_race(ctx.checkpoint) else 1
        tok = to_tokens(seq, upto)
        return LapSeqSignal(tok.x, tok.running)

    # -- the network for a season ------------------------------------------------------------
    def _dir(self, tables: Tables, season: int) -> Path:
        from f1p.core.config import get_settings
        from f1p.eval.runner import table_fingerprint

        key = id(tables)
        if key not in self._fingerprint:
            self._fingerprint[key] = table_fingerprint(tables)
        root = Path(self.root)
        if not root.is_absolute():
            root = get_settings().paths.artifacts / root
        sig = lq.digest(
            self.config,
            self.training_seasons,
            self.pretrain,
            self.min_races,
            self._fingerprint[key],
        )
        return root / sig / str(season)

    def network(self, tables: Tables, season: int) -> Any:
        if season in self._nets:
            return self._nets[season]
        where = self._dir(tables, season)
        with lq.FileLock(where.parent / f"{season}.lock"):
            if not (where / "weights.npz").exists():
                trained = self.fit(tables, season)
                if trained is None:
                    self._nets[season] = None
                    return None
                lq.save(trained, where, {"season": season, "model": self.name})
            backend, _, _ = lq.load(where)
        self._nets[season] = backend
        return backend

    def fit(self, tables: Tables, season: int) -> lq.Trained | None:
        ctxs = training_contexts(tables, season, self.training_seasons)
        real = [s for c in ctxs if (s := race_seq(c, outcome=True)) is not None]
        if len(real) < self.min_races:
            log.info("lapseq %d: %d training races, too few", season, len(real))
            return None
        sims: list[RaceSeq] = []
        if self.pretrain and self.config.sims_per_race > 0:
            sims = simulate_corpus(ctxs, self.config.sims_per_race, self.config.seed)
        log.info("lapseq %d: %d real races, %d simulated", season, len(real), len(sims))
        return lq.train(real, sims, self.config)

    # -- Forecaster protocol ---------------------------------------------------------------------
    def forecast(self, ctx: EventContext, signal: Any, history: History) -> Forecast | None:
        del history  # the network is trained on earlier seasons' races, not the score history
        if not isinstance(signal, LapSeqSignal):
            return None
        net = self.network(ctx.tables, ctx.season)
        if net is None:
            return None
        p = lq.predict(net, signal.tokens())
        p_finish = np.where(p.running, p.p_classified, 0.0)
        return Forecast.from_log_strength(
            ctx.event_id,
            ctx.checkpoint,
            self.name,
            ctx.drivers,
            p.strength,
            p_finish=p_finish,
            meta={
                "p_neutral_next": p.p_neutral,
                "lap": float(ctx.meta.get("lap", 0)),
            },
        )
