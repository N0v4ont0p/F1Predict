"""Fast rolling-origin evaluation: separate *prepare* and *score* stages, run in parallel, with
prepared signals cached on disk (plan M6 §C.2).

A backtest has two very different stages:

* **prepare** — every context's model signal (rating beliefs, lap-model pace, …). Expensive,
  and independent per context once the model's filter state exists. It depends only on the
  tables and the model's *prepare signature* (the settings that change signals).
* **score** — each event's forecast from its signal and the history before it, then the
  metrics. Cheap per event; depends on the combiner settings too.

Splitting them lets a tuning run that only changes combiner settings reuse one cached
``Prepared`` across thousands of trials. Both stages fan out over one pool of worker processes;
results are identical to the serial path (forecast seeds depend only on the event, and every
fit is keyed by season and checkpoint).

Workers are *spawned* (never forked), so a script that calls this with ``workers > 1`` needs
the usual ``if __name__ == "__main__":`` guard; the ``f1p`` CLI already has one.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import re
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, is_dataclass
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from f1p.core.logging import get_logger
from f1p.eval.context import EventContext, Forecaster, Tables, build_contexts, target_checkpoints
from f1p.eval.forecast import Forecast
from f1p.eval.metrics import Outcome, calibration_pairs, score

log = get_logger("eval.runner")

Key = tuple[str, str]  # (event_id, checkpoint)
Context = tuple[EventContext, Outcome, datetime | None]

CACHE_VERSION = 1


# -- fingerprints -----------------------------------------------------------------------------
_TABLE_FP: dict[int, str] = {}


def table_fingerprint(tables: Tables) -> str:
    """A content hash of every table (cached per table-set object)."""
    key = id(tables)
    if key not in _TABLE_FP:
        h = hashlib.sha256()
        for name in sorted(tables):
            df = tables[name]
            h.update(name.encode())
            h.update(str(df.shape).encode())
            h.update(str(df.schema).encode())
            if df.height:
                h.update(str(int(df.hash_rows(seed=0).sum())).encode())
        _TABLE_FP[key] = h.hexdigest()[:20]
    return _TABLE_FP[key]


_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")


def _settings_repr(model: Any) -> str:
    """Every public setting, without memory addresses (a function field reprs as
    ``<function name>``), so the text is the same in every process."""
    if is_dataclass(model):
        text = repr(model)  # the fields declared with repr=True
    else:
        text = repr({k: v for k, v in sorted(vars(model).items()) if not k.startswith("_")})
    return _ADDRESS.sub("", text)


def _digest(model: Any, text: str) -> str:
    return hashlib.sha256(f"{type(model).__name__}|{text}".encode()).hexdigest()[:20]


def prepare_signature(model: Any) -> str:
    """What makes a model's signals: ``model.prepare_signature()`` when defined (only the
    settings ``prepare`` reads), else every public setting (conservative: any change misses)."""
    sig = getattr(model, "prepare_signature", None)
    text = _ADDRESS.sub("", repr(sig())) if callable(sig) else _settings_repr(model)
    return _digest(model, text)


def model_signature(model: Any) -> str:
    """What makes a model's forecasts: every public setting. Identifies an item in the
    holdout ledger."""
    return _digest(model, _settings_repr(model))


# -- the prepared stage -----------------------------------------------------------------------
@dataclass
class Prepared:
    """Signals of one model for every context in ``span`` (history included)."""

    model: str
    target: str
    checkpoints: tuple[str, ...]
    span: tuple[int, int]
    signature: str
    tables: str
    signals: dict[Key, Any] = field(default_factory=dict)


_CONTEXTS: dict[tuple[int, tuple[int, int], tuple[str, ...], str], list[Context]] = {}


def _contexts(
    tables: Tables, span: tuple[int, int], checkpoints: tuple[str, ...], target: str
) -> list[Context]:
    """Every context of ``span`` (memoised per table set: both stages and every model of a
    run share them)."""
    key = (id(tables), span, checkpoints, target)
    if key not in _CONTEXTS:
        if len(_CONTEXTS) > 16:
            _CONTEXTS.clear()
        _CONTEXTS[key] = build_contexts(
            tables, seasons=span, checkpoints=checkpoints, target=target
        )
    return _CONTEXTS[key]


# worker globals (set once per process by the initialiser)
_W: dict[str, Any] = {}

# a worker is one core: one polars thread, one BLAS/OpenMP thread
_SINGLE_THREAD = (
    "POLARS_MAX_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _init_worker(
    tables: Tables, model: Any, span: tuple[int, int], cps: tuple[str, ...], target: str
) -> None:
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(1)
    except ImportError:  # pragma: no cover - optional
        pass
    _W["tables"] = tables
    _W["model"] = model
    _W["contexts"] = {c.key: (c, o, k) for c, o, k in _contexts(tables, span, cps, target)}


def _prepare_chunk(keys: list[Key]) -> dict[Key, Any]:
    model, ctxs = _W["model"], _W["contexts"]
    return {k: model.prepare(ctxs[k][0]) for k in keys}


class WorkerPool:
    """Spawned workers holding one model's tables, contexts and state, shared by the prepare
    and score stages. No process starts until the first task, so a fully cached, serial run
    costs nothing.

    Spawn, never fork: polars' thread pool (and Accelerate/Metal on macOS) deadlock in a forked
    child. Each worker receives the tables once, through the initialiser, and runs
    single-threaded so ``workers`` processes use ``workers`` cores."""

    def __init__(
        self,
        workers: int,
        tables: Tables,
        model: Any,
        span: tuple[int, int],
        cps: tuple[str, ...],
        target: str,
    ) -> None:
        self.workers = workers
        self._init = (tables, model, span, cps, target)
        self._executor: ProcessPoolExecutor | None = None

    def map(self, fn: Callable[[Any], Any], tasks: Sequence[Any]) -> list[Any]:
        saved = {k: os.environ.get(k) for k in _SINGLE_THREAD}
        os.environ.update(dict.fromkeys(_SINGLE_THREAD, "1"))  # read by children at spawn
        try:
            if self._executor is None:
                self._executor = ProcessPoolExecutor(
                    max_workers=self.workers,
                    mp_context=get_context("spawn"),
                    initializer=_init_worker,
                    initargs=self._init,
                )
            return list(self._executor.map(fn, tasks))
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None

    def __enter__(self) -> WorkerPool:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _chunks(items: Sequence[Any], n: int) -> list[list[Any]]:
    """``n`` contiguous, near-equal chunks (contiguity keeps per-window caches warm)."""
    n = max(1, min(n, len(items)))
    size, rem = divmod(len(items), n)
    out, i = [], 0
    for j in range(n):
        step = size + (1 if j < rem else 0)
        out.append(list(items[i : i + step]))
        i += step
    return [c for c in out if c]


def prepare(
    tables: Tables,
    model: Forecaster,
    *,
    span: tuple[int, int],
    checkpoints: tuple[str, ...] | None = None,
    target: str = "race",
    workers: int = 1,
    cache_dir: Path | None = None,
    pool: WorkerPool | None = None,
) -> Prepared:
    """Signals for every context in ``span`` at ``checkpoints`` the model forecasts.

    With ``workers`` > 1 the contexts are split into contiguous chunks over ``pool`` (or a
    pool of this call's own)."""
    cps = model_checkpoints(model, checkpoints, target)
    sig = prepare_signature(model)
    fp = table_fingerprint(tables)
    path = None
    if cache_dir is not None:
        parts = [f"v{CACHE_VERSION}", model.name, target, *map(str, span), *cps, sig, fp]
        name = "-".join(parts) + ".pkl"
        path = Path(cache_dir) / name
        if path.exists():
            with path.open("rb") as fh:
                cached: Prepared = pickle.load(fh)
            log.info("prepared %s from cache (%d signals)", model.name, len(cached.signals))
            return cached
    contexts = _contexts(tables, span, cps, target)
    keys = [c.key for c, _, _ in contexts]
    if workers <= 1 or len(keys) < 2 * workers:
        signals = {c.key: model.prepare(c) for c, _, _ in contexts}
    else:
        signals = {}
        own = pool is None
        pool = pool or WorkerPool(workers, tables, model, span, cps, target)
        try:
            for part in pool.map(_prepare_chunk, _chunks(keys, workers)):
                signals.update(part)
        finally:
            if own:
                pool.close()
    prepared = Prepared(model.name, target, cps, span, sig, fp, signals)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as fh:
            pickle.dump(prepared, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)
    return prepared


# -- the score stage --------------------------------------------------------------------------
@dataclass
class Scored:
    rows: list[dict[str, Any]]
    calibration: dict[tuple[str, int], tuple[list[np.ndarray], list[np.ndarray]]]
    forecasts: dict[Key, Forecast]


def _score_events(
    model: Any,
    contexts: dict[Key, Context],
    order: list[Key],
    signals: dict[Key, Any],
    targets: Iterable[Key],
    target: str,
    keep: bool,
) -> Scored:
    rows: list[dict[str, Any]] = []
    cal: dict[tuple[str, int], tuple[list[np.ndarray], list[np.ndarray]]] = {}
    fcs: dict[Key, Forecast] = {}
    for key in targets:
        ctx, outcome, _ = contexts[key]
        history = [
            (contexts[k][0], signals[k], contexts[k][1])
            for k in order
            if k[1] == ctx.checkpoint and contexts[k][2] is not None and contexts[k][2] <= ctx.as_of  # type: ignore[operator]
        ]
        fc = model.forecast(ctx, signals[key], history)
        if fc is None:
            continue
        rows.append(
            {
                "model": model.name,
                "target": target,
                "checkpoint": ctx.checkpoint,
                "event_id": ctx.event_id,
                "season": ctx.season,
                "round": ctx.round,
                "field_added": ctx.meta.get("field_added", 0),
                **score(fc, outcome),
            }
        )
        for k in (3, 10):
            p, y = calibration_pairs(fc, outcome, k)
            bucket = cal.setdefault((ctx.checkpoint, k), ([], []))
            bucket[0].append(p)
            bucket[1].append(y)
        if keep:
            fcs[key] = fc
    return Scored(rows, cal, fcs)


def _score_chunk(args: tuple[dict[Key, Any], list[Key], list[Key], str, bool]) -> Scored:
    signals, order, targets, target, keep = args
    return _score_events(_W["model"], _W["contexts"], order, signals, targets, target, keep)


def score_prepared(
    tables: Tables,
    model: Forecaster,
    prepared: Prepared,
    *,
    seasons: tuple[int, int],
    keep_forecasts: bool = False,
    workers: int = 1,
    pool: WorkerPool | None = None,
) -> Scored:
    """Forecast and score every event of ``seasons`` from ``prepared``. Work is split by
    (season, checkpoint), the key of every per-season fit, so each worker refits a group's
    combiner once and the results equal the serial path's."""
    contexts = _contexts(tables, prepared.span, prepared.checkpoints, prepared.target)
    by_key = {c.key: (c, o, k) for c, o, k in contexts}
    order = [c.key for c, _, _ in contexts]
    lo, hi = seasons
    targets = [k for k in order if lo <= by_key[k][0].season <= hi]
    groups: dict[tuple[int, str], list[Key]] = {}
    for k in targets:
        groups.setdefault((by_key[k][0].season, k[1]), []).append(k)
    if workers <= 1 or len(groups) < 2:
        return _score_events(
            model, by_key, order, prepared.signals, targets, prepared.target, keep_forecasts
        )
    tasks = []
    for (_, cp), group in groups.items():
        # only the signals the group reads: its own and its checkpoint's earlier events
        last = max(by_key[k][0].as_of for k in group)
        mine = set(group)
        need = [k for k in order if k[1] == cp and (k in mine or _known_by(by_key[k][2], last))]
        sub = {k: prepared.signals[k] for k in need}
        tasks.append((sub, need, group, prepared.target, keep_forecasts))
    out = Scored([], {}, {})
    own = pool is None
    pool = pool or WorkerPool(
        min(workers, len(tasks)),
        tables,
        model,
        prepared.span,
        prepared.checkpoints,
        prepared.target,
    )
    try:
        parts = pool.map(_score_chunk, tasks)
    finally:
        if own:
            pool.close()
    for part in parts:
        out.rows.extend(part.rows)
        for k, (p, y) in part.calibration.items():
            bucket = out.calibration.setdefault(k, ([], []))
            bucket[0].extend(p)
            bucket[1].extend(y)
        out.forecasts.update(part.forecasts)
    rank = {k: i for i, k in enumerate(order)}
    out.rows.sort(key=lambda r: rank[(r["event_id"], r["checkpoint"])])
    return out


def _known_by(known: datetime | None, as_of: datetime) -> bool:
    return known is not None and known <= as_of


def model_checkpoints(
    model: Forecaster, checkpoints: tuple[str, ...] | None, target: str
) -> tuple[str, ...]:
    """The requested checkpoints (default: the target's) that ``model`` forecasts."""
    return tuple(c for c in (checkpoints or target_checkpoints(target)) if c in model.checkpoints)


def per_event_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def cpu_workers(reserve: int = 0) -> int:
    """Worker count: the strong (performance) cores on Apple silicon, else all cores, minus
    ``reserve``."""
    try:
        from f1p.silicon.device import detect

        strong = detect().strong_cores
        if strong:
            return max(1, strong - reserve)
    except Exception:  # discovery is best effort
        pass
    return max(1, (os.cpu_count() or 2) - reserve)
