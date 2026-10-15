"""Rolling-origin evaluation (plan §10.1).

For each event and checkpoint, a model sees the context (data public at ``as_of``) plus the
*history* of earlier events whose results were public by then, and returns a forecast that is
scored against the outcome. Nothing a model receives was unknowable at the time.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import polars as pl

from f1p.core.logging import get_logger
from f1p.eval.bootstrap import Comparison, paired_bootstrap
from f1p.eval.context import CHECKPOINTS, Forecaster, Tables, build_contexts
from f1p.eval.forecast import Forecast
from f1p.eval.metrics import (
    HIGHER_IS_BETTER,
    calibration_pairs,
    expected_calibration_error,
    score,
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
    checkpoints: tuple[str, ...] = CHECKPOINTS,
    keep_forecasts: bool = False,
) -> EvalResult:
    """Evaluate ``models`` on every event in ``seasons`` (history starts at ``history_from``)."""
    lo, hi = seasons
    start = history_from if history_from is not None else lo - 8
    contexts = build_contexts(tables, seasons=(start, hi), checkpoints=checkpoints)
    rows: list[dict[str, Any]] = []
    calibration: dict[tuple[str, str, int], tuple[list[np.ndarray], list[np.ndarray]]] = {}
    forecasts: dict[tuple[str, str, str], Forecast] = {}
    for model in models:
        usable = [(c, o, t) for c, o, t in contexts if c.checkpoint in model.checkpoints]
        signals = {c.key: model.prepare(c) for c, _, _ in usable}
        for ctx, outcome, _ in usable:
            if not lo <= ctx.season <= hi:
                continue
            history = [
                (c, signals[c.key], o)
                for c, o, known in usable
                if c.checkpoint == ctx.checkpoint and known is not None and known <= ctx.as_of
            ]
            fc = model.forecast(ctx, signals[ctx.key], history)
            if fc is None:
                continue
            m = score(fc, outcome)
            rows.append(
                {
                    "model": model.name,
                    "checkpoint": ctx.checkpoint,
                    "event_id": ctx.event_id,
                    "season": ctx.season,
                    "round": ctx.round,
                    "field_added": ctx.meta.get("field_added", 0),
                    **m,
                }
            )
            for k in (3, 10):
                p, y = calibration_pairs(fc, outcome, k)
                bucket = calibration.setdefault((model.name, ctx.checkpoint, k), ([], []))
                bucket[0].append(p)
                bucket[1].append(y)
            if keep_forecasts:
                forecasts[(model.name, ctx.checkpoint, ctx.event_id)] = fc
        log.info("evaluated %s", model.name)
    per_event = pl.DataFrame(rows) if rows else pl.DataFrame()
    cal = {k: (np.concatenate(p), np.concatenate(y)) for k, (p, y) in calibration.items()}
    return EvalResult(per_event, cal, forecasts)


def better(metric: str, diff: float) -> bool:
    return diff > 0 if metric in HIGHER_IS_BETTER else diff < 0


def as_of_str(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")
