"""Challenger models (plan item 8, Phase 7): gradient-boosted rankers on one causal feature
table.

* **LightGBM** with the lambdarank objective;
* **CatBoost** with YetiRank.

Both see the same per-car features, built from what is public at the checkpoint (the
world model's beliefs and signals, standings, the last race, reliability, circuit
affinities, rain; at C4 the grid and qualifying). Rankers output scores, not probabilities,
so each score is mapped to a Plackett-Luce log-strength by a scale and stage temperatures
fitted **out of sample**: the ranker is trained on seasons ``S-8 .. S-2``, the mapping is
fitted on its predictions for season ``S-1`` (the latest fifth of the races when that season
is thin), and the ranker is then retrained on ``S-8 .. S-1`` for season ``S``. Retirements
use the world model's calibrated P(finish), so the comparison is about the order, not about
who finishes.

TabICLv2 and TabPFN are not included: both need PyTorch, which is not a dependency here.
They would plug in through :class:`RankerModel` the same way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from f1p.eval.baselines import History, last_race_ranks, standings_ranks
from f1p.eval.context import EventContext
from f1p.eval.forecast import Forecast
from f1p.eval.metrics import Outcome
from f1p.models import linear_pl
from f1p.models.features import circuit_affinity
from f1p.models.lap_model import lap_pace_signal
from f1p.models.world_v1 import Signal, WorldV1, rain_signal

FloatArray = NDArray[np.float64]

FEATURES = {
    "C1": ("rating", "race_pace"),
    "C4": ("rating", "grid", "quali", "race_pace", "pace"),
}
EXTRA = (
    "rating_sd",
    "standings",
    "last_race",
    "finish_logit",
    "lap_pace",
    "team_affinity",
    "driver_affinity",
    "rain",
    "field",
)


@dataclass(frozen=True, slots=True)
class RankSignal:
    x: FloatArray  # n x p features
    world: Signal  # the world model's own signal (for P(finish))


def _log_rank(r: FloatArray | None, n: int) -> FloatArray:
    return np.zeros(n) if r is None else -np.log(r)


def feature_names(checkpoint: str) -> tuple[str, ...]:
    return (*FEATURES[checkpoint], *EXTRA)


def relevance(ctx: EventContext, outcome: Outcome) -> NDArray[np.int64]:
    """Ranking labels: classified cars by finishing order (winner highest), others 0."""
    rel = np.zeros(ctx.n, dtype=np.int64)
    idx = ctx.index()
    m = len(outcome.classified)
    for k, d in enumerate(outcome.classified):
        if d in idx:
            rel[idx[d]] = m - k
    return rel


@dataclass
class RankerModel:
    """A gradient-boosted ranker as a forecaster. ``kind`` is ``lgbm`` or ``catboost``."""

    kind: str = "lgbm"
    name: str = ""
    checkpoints: frozenset[str] = frozenset({"C1", "C4"})
    targets: frozenset[str] = frozenset({"race"})
    training_seasons: int = 8
    seed: int = 7
    _world: WorldV1 = field(default_factory=lambda: WorldV1(features=dict(FEATURES)), repr=False)
    _fits: dict[tuple[int, str], tuple[Any, linear_pl.PLFit]] = field(
        default_factory=dict, repr=False
    )

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"{self.kind}_rank"
        if self.kind not in ("lgbm", "catboost"):
            raise ValueError(f"unknown ranker {self.kind!r}")

    # -- features ---------------------------------------------------------------------------
    def prepare(self, ctx: EventContext) -> RankSignal:
        world = self._world.prepare(ctx)
        _, var = self._world.beliefs(ctx)
        st = self._world._state(ctx.tables)
        extra = {
            "rating_sd": np.sqrt(var),
            "standings": _log_rank(standings_ranks(ctx), ctx.n),
            "last_race": _log_rank(last_race_ranks(ctx), ctx.n),
            "finish_logit": world.finish_logit,
            "lap_pace": lap_pace_signal(ctx, self._world._laps),
            "team_affinity": circuit_affinity(ctx, by="team", lineage=st.lineage),
            "driver_affinity": circuit_affinity(ctx, by="driver"),
            "rain": np.full(ctx.n, rain_signal(ctx)),
            "field": np.full(ctx.n, float(ctx.n)),
        }
        x = np.column_stack([world.x, *(extra[k] for k in EXTRA)])
        return RankSignal(x, world)

    # -- training ---------------------------------------------------------------------------
    def _train(self, rows: list[tuple[EventContext, RankSignal, Outcome]]) -> Any:
        x = np.concatenate([sig.x for _, sig, _ in rows])
        y = np.concatenate([relevance(c, o) for c, _, o in rows])
        groups = [c.n for c, _, _ in rows]
        if self.kind == "lgbm":
            import lightgbm as lgb

            params = {
                "objective": "lambdarank",
                "learning_rate": 0.05,
                "num_leaves": 15,
                "min_data_in_leaf": 20,
                "bagging_fraction": 0.8,
                "bagging_freq": 1,
                "feature_fraction": 0.8,
                "label_gain": list(range(int(y.max()) + 1)),
                "seed": self.seed,
                "num_threads": 4,
                "verbose": -1,
            }
            return lgb.train(params, lgb.Dataset(x, y, group=groups), num_boost_round=300)
        from catboost import CatBoostRanker

        group_id = np.repeat(np.arange(len(groups)), groups)
        model = CatBoostRanker(
            loss_function="YetiRank",
            iterations=400,
            depth=4,
            learning_rate=0.08,
            random_seed=self.seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=4,
        )
        model.fit(x, y, group_id=group_id)
        return model

    def _scale(
        self, model: Any, rows: list[tuple[EventContext, RankSignal, Outcome]]
    ) -> linear_pl.PLFit:
        """Score → log-strength: a PL scale and smooth stage temperatures fitted on held-out
        races."""
        samples = []
        for c, sig, o in rows:
            score = np.asarray(model.predict(sig.x), dtype=np.float64)
            idx = c.index()
            fin = np.array([idx[d] for d in o.classified if d in idx], dtype=np.intp)
            samples.append(((score - score.mean())[:, None], fin))
        n_stages = linear_pl.MAX_STAGES
        return linear_pl.fit_model(
            samples,
            prior=np.array([1.0]),
            n_stages=n_stages,
            l2=0.1,
            bounds=[(0.0, None)],
            basis=linear_pl.smooth_stage_basis(n_stages),
        )

    def _fit(self, ctx: EventContext, history: History) -> tuple[Any, linear_pl.PLFit] | None:
        key = (ctx.season, ctx.checkpoint)
        if key in self._fits:
            return self._fits[key]
        rows = [
            (c, sig, o)
            for c, sig, o in history
            if isinstance(sig, RankSignal)
            and ctx.season - self.training_seasons <= c.season < ctx.season
            and len(o.classified) >= 2
        ]
        rows.sort(key=lambda r: (r[0].season, r[0].round))
        if len(rows) < 20:
            return None
        # the score→strength mapping is fitted on the most recent season the ranker did not
        # see (or, if that season is thin, the latest fifth of the races)
        last = rows[-1][0].season
        holdout = [r for r in rows if r[0].season == last]
        if len(holdout) < 5 or len(holdout) > len(rows) - 15:
            holdout = rows[-max(5, len(rows) // 5) :]
        inner = rows[: len(rows) - len(holdout)]
        scale = self._scale(self._train(inner), holdout)
        fitted = (self._train(rows), scale)
        self._fits[key] = fitted
        return fitted

    def forecast(self, ctx: EventContext, signal: Any, history: History) -> Forecast | None:
        if not isinstance(signal, RankSignal):
            return None
        fitted = self._fit(ctx, history)
        if fitted is None:
            return None
        model, scale = fitted
        score = np.asarray(model.predict(signal.x), dtype=np.float64)
        ls = scale.log_strength((score - score.mean())[:, None])
        world_hist = [(c, s.world, o) for c, s, o in history if isinstance(s, RankSignal)]
        p_finish = self._world.forecast(ctx, signal.world, world_hist).p_finish
        return Forecast.from_log_strength(
            ctx.event_id,
            ctx.checkpoint,
            self.name,
            ctx.drivers,
            ls,
            meta={"scale": float(scale.w[0])},
            p_finish=p_finish,
            stage_scales=np.exp(scale.theta),
        )
