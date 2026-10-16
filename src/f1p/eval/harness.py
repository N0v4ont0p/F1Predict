"""Rolling-origin evaluation (plan §10.1).

For each event and checkpoint, a model sees the context (data public at ``as_of``) plus the
*history* of earlier events whose results were public by then, and returns a forecast that is
scored against the outcome. Nothing a model receives was unknowable at the time.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from f1p.core.logging import get_logger
from f1p.eval.bootstrap import Comparison, paired_bootstrap
from f1p.eval.context import (
    EventContext,
    Forecaster,
    Tables,
    build_contexts,
    live_context,
    target_checkpoints,
)
from f1p.eval.forecast import Forecast
from f1p.eval.metrics import (
    HIGHER_IS_BETTER,
    expected_calibration_error,
)

log = get_logger("eval.harness")

METRICS = (
    "winner_logloss",
    "winner_hit",
    "podium_brier",
    "points_brier",
    "spearman",
    "finisher_pl_loglik",
)


@dataclass
class EvalResult:
    per_event: pl.DataFrame
    calibration: dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict
    )
    forecasts: dict[tuple[str, str, str], Forecast] = field(default_factory=dict)

    def summary(self) -> pl.DataFrame:
        agg = self.per_event.group_by("model", "checkpoint").agg(
            pl.len().alias("events"), *(pl.col(m).mean() for m in METRICS)
        )
        ece = [
            {"model": m, "checkpoint": c, f"ece_top{k}": expected_calibration_error(p, y)}
            for (m, c, k), (p, y) in self.calibration.items()
        ]
        if ece:
            e = pl.DataFrame(ece).group_by("model", "checkpoint").agg(pl.all().drop_nulls().first())
            agg = agg.join(e, on=["model", "checkpoint"], how="left")
        return agg.sort("checkpoint", "winner_logloss")

    def compare(self, model: str, reference: str, checkpoint: str, metric: str) -> Comparison:
        """Paired bootstrap of ``model - reference`` on events both forecast."""
        df = self.per_event.filter(pl.col("checkpoint") == checkpoint)
        a = df.filter(pl.col("model") == model).select("event_id", pl.col(metric).alias("a"))
        b = df.filter(pl.col("model") == reference).select("event_id", pl.col(metric).alias("b"))
        j = a.join(b, on="event_id")
        return paired_bootstrap(j["a"].to_numpy(), j["b"].to_numpy())


def run(
    tables: Tables,
    models: Sequence[Forecaster],
    *,
    seasons: tuple[int, int],
    history_from: int | None = None,
    checkpoints: tuple[str, ...] | None = None,
    keep_forecasts: bool = False,
    target: str = "race",
    workers: int = 1,
    cache_dir: Path | None = None,
) -> EvalResult:
    """Evaluate ``models`` on every event in ``seasons`` (history starts at ``history_from``)
    against ``target`` — the race classification or the qualifying order. Models that do
    not forecast ``target`` are skipped.

    ``workers`` > 1 fans both stages out over processes (same results); ``cache_dir`` keeps
    each model's prepared signals on disk, keyed by the tables' content and the model's
    prepare signature (see :mod:`f1p.eval.runner`)."""
    from f1p.eval import runner

    lo, hi = seasons
    start = history_from if history_from is not None else lo - 8
    checkpoints = checkpoints or target_checkpoints(target)
    rows: list[dict[str, Any]] = []
    calibration: dict[tuple[str, str, int], tuple[list[np.ndarray], list[np.ndarray]]] = {}
    forecasts: dict[tuple[str, str, str], Forecast] = {}
    for model in models:
        if target not in model_targets(model):
            continue
        cps = runner.model_checkpoints(model, checkpoints, target)
        if not cps:
            continue  # the model forecasts none of the requested checkpoints
        with runner.WorkerPool(workers, tables, model, (start, hi), cps, target) as pool:
            prepared = runner.prepare(
                tables,
                model,
                span=(start, hi),
                checkpoints=cps,
                target=target,
                workers=workers,
                cache_dir=cache_dir,
                pool=pool,
            )
            scored = runner.score_prepared(
                tables,
                model,
                prepared,
                seasons=seasons,
                keep_forecasts=keep_forecasts,
                workers=workers,
                pool=pool,
            )
        rows.extend(scored.rows)
        for (cp, k), (p, y) in scored.calibration.items():
            bucket = calibration.setdefault((model.name, cp, k), ([], []))
            bucket[0].extend(p)
            bucket[1].extend(y)
        for (eid, cp), fc in scored.forecasts.items():
            forecasts[(model.name, cp, eid)] = fc
        log.info("evaluated %s", model.name)
    per_event = pl.DataFrame(rows) if rows else pl.DataFrame()
    cal = {k: (np.concatenate(p), np.concatenate(y)) for k, (p, y) in calibration.items()}
    return EvalResult(per_event, cal, forecasts)


class EventForecaster:
    """Forecasts any number of events (raced or not) at one checkpoint, as the harness would.

    The history — every earlier event's context, prepared signal and outcome — is built once
    and reused, so projecting a whole season (or every round of many seasons) costs one
    ``prepare`` per forecast instead of one per history event.
    """

    def __init__(
        self,
        tables: Tables,
        model: Forecaster,
        checkpoint: str,
        *,
        history_seasons: int = 8,
        target: str = "race",
    ) -> None:
        self.tables = tables
        self.model = model
        self.checkpoint = checkpoint
        self.history_seasons = history_seasons
        self.target = target
        self._history: dict[int, list[tuple[EventContext, Any, Any, datetime | None]]] = {}

    def _season_history(self, season: int) -> list[tuple[EventContext, Any, Any, datetime | None]]:
        """Prepared contexts of the ``history_seasons`` before ``season`` and of the season
        itself (filtered per forecast by when each outcome became public)."""
        if season not in self._history:
            contexts = build_contexts(
                self.tables,
                seasons=(season - self.history_seasons, season),
                checkpoints=(self.checkpoint,),
                target=self.target,
            )
            self._history[season] = [
                (c, self.model.prepare(c), o, known)
                for c, o, known in contexts
                if known is not None
            ]
        return self._history[season]

    def forecast(
        self,
        event_id: str,
        *,
        as_of: datetime | None = None,
        lineup: pl.DataFrame | None = None,
    ) -> tuple[Forecast | None, EventContext]:
        ctx = live_context(
            self.tables, event_id, self.checkpoint, as_of, lineup=lineup, target=self.target
        )
        history = [
            (c, sig, o)
            for c, sig, o, known in self._season_history(ctx.season)
            if known is not None and known <= ctx.as_of and c.event_id != event_id
        ]
        return self.model.forecast(ctx, self.model.prepare(ctx), history), ctx


def forecast_event(
    tables: Tables,
    model: Forecaster,
    event_id: str,
    checkpoint: str,
    *,
    as_of: datetime | None = None,
    history_seasons: int = 8,
    target: str = "race",
) -> tuple[Forecast | None, EventContext]:
    """Forecast one event (raced or not) exactly as the harness would at ``as_of``."""
    return EventForecaster(
        tables, model, checkpoint, history_seasons=history_seasons, target=target
    ).forecast(event_id, as_of=as_of)


def model_targets(model: object) -> frozenset[str]:
    """What a forecaster predicts (models without a ``targets`` attribute forecast races)."""
    return frozenset(getattr(model, "targets", frozenset({"race"})))


def better(metric: str, diff: float) -> bool:
    return diff > 0 if metric in HIGHER_IS_BETTER else diff < 0


def as_of_str(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")
