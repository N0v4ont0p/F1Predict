"""The search engine (plan M6 §C.1-§C.4).

* **Objective** — a configuration is scored season by season over *tune* (2014-2021 by
  default) on one per-event metric (lower is better: log-losses as they are, likelihoods
  negated); after each season the running mean is reported, so Hyperband/ASHA prunes a bad
  trial after 2014-2016 (the season is the budget unit). Prepared signals come from the
  runner's cache: combiner-only spaces share them across every trial.
* **Storage** — an Optuna journal file under ``artifacts/tuning/``: resumable, and safe for
  several worker processes (one per performance core), which are spawned, single-threaded.
* **Selection** — the best trials on *tune* are re-scored on *select*; the winner is the
  simplest configuration within one standard error (paired over events) of the best.
* **Null** — how much "best of N" improves on the hand-set configuration by chance: the
  per-event differences of every trial to the baseline get one shared random sign per
  event (exchangeable under the null of no real difference), and the maximum mean
  improvement over trials is recorded per flip.
"""

from __future__ import annotations

import json
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from f1p.eval import runner, splits
from f1p.eval.context import Tables
from f1p.eval.metrics import HIGHER_IS_BETTER
from f1p.tuning.spaces import SPACES, Space

DEFAULT_METRIC = "finisher_pl_loglik"


def loss_sign(metric: str) -> float:
    """+1 when lower is better, -1 when higher is better (so every objective is a loss)."""
    return -1.0 if metric in HIGHER_IS_BETTER else 1.0


# -- scoring one configuration ---------------------------------------------------------------------
@dataclass
class Scores:
    events: list[str]  # "event_id/checkpoint"
    losses: list[float]  # per event, the loss (sign applied)

    @property
    def mean(self) -> float:
        return float(np.mean(self.losses)) if self.losses else math.inf


def score_config(
    tables: Tables,
    model: Any,
    seasons: tuple[int, int],
    *,
    metric: str = DEFAULT_METRIC,
    target: str = "race",
    cache_dir: Path | None = None,
    report: Any | None = None,
) -> Scores:
    """Score ``model`` season by season over ``seasons``. ``report(step, running_mean)``
    is called after each season and may raise to stop (Optuna pruning)."""
    lo, hi = seasons
    span = (lo - 8, hi)
    cps = runner.model_checkpoints(model, None, target)
    prepared = runner.prepare(
        tables, model, span=span, checkpoints=cps, target=target, cache_dir=cache_dir
    )
    sign = loss_sign(metric)
    events: list[str] = []
    losses: list[float] = []
    for step, season in enumerate(range(lo, hi + 1)):
        scored = runner.score_prepared(tables, model, prepared, seasons=(season, season))
        for row in scored.rows:
            value = row.get(metric)
            if value is None or not math.isfinite(float(value)):
                continue
            events.append(f"{row['event_id']}/{row['checkpoint']}")
            losses.append(sign * float(value))
        if report is not None and losses:
            report(step, float(np.mean(losses)))
    return Scores(events, losses)


# -- studies ---------------------------------------------------------------------------------------
def study_path(root: Path, name: str) -> Path:
    return Path(root) / "tuning" / f"{name}.journal"


def _storage(path: Path) -> Any:
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend

    path.parent.mkdir(parents=True, exist_ok=True)
    return JournalStorage(JournalFileBackend(str(path)))


def _sampler(kind: str, seed: int) -> Any:
    import optuna

    if kind == "cmaes":
        return optuna.samplers.CmaEsSampler(seed=seed, warn_independent_sampling=False)
    if kind == "gp":
        return optuna.samplers.GPSampler(seed=seed)
    return optuna.samplers.TPESampler(seed=seed, multivariate=True)


def open_study(path: Path, name: str, space: Space, *, seed: int = 0) -> Any:
    """Create or resume the study; the hand-set configuration is always trial 0."""
    import optuna

    hb = optuna.pruners.HyperbandPruner(min_resource=3, max_resource=8, reduction_factor=3)
    study = optuna.create_study(
        study_name=name,
        storage=_storage(path),
        sampler=_sampler(space.sampler, seed),
        pruner=hb,
        direction="minimize",
        load_if_exists=True,
    )
    if not study.trials:
        study.enqueue_trial(space.baseline, user_attrs={"baseline": True})
    return study


@dataclass
class _WorkerArgs:
    path: str
    name: str
    space: str
    trials: int
    seasons: tuple[int, int]
    metric: str
    cache_dir: str | None
    seed: int
    table_names: tuple[str, ...]


_TABLES: dict[str, Tables] = {}


def load_tables(names: tuple[str, ...]) -> Tables:
    """The lake tables a worker evaluates on (read once per process)."""
    if "t" not in _TABLES:
        from f1p.core.config import get_settings
        from f1p.lake.pipeline import lake_for

        with lake_for(get_settings(), read_only=True) as lake:
            present = set(lake.tables())
            _TABLES["t"] = lake.read_all([n for n in names if n in present])
    return _TABLES["t"]


def make_objective(
    tables: Tables, space: Space, seasons: tuple[int, int], metric: str, cache_dir: Path | None
) -> Any:
    import optuna

    def objective(trial: Any) -> float:
        values = space.suggest(trial)
        model = space.build(values)

        baseline = bool(trial.user_attrs.get("baseline"))

        def report(step: int, value: float) -> None:
            trial.report(value, step)
            # the hand-set configuration is the reference (for the null and the gate):
            # it always runs to the end
            if not baseline and trial.should_prune():
                raise optuna.TrialPruned

        scores = score_config(
            tables,
            model,
            seasons,
            metric=metric,
            target=space.target,
            cache_dir=cache_dir,
            report=report,
        )
        trial.set_user_attr("events", scores.events)
        trial.set_user_attr("losses", [round(x, 6) for x in scores.losses])
        return scores.mean

    return objective


def _work(args: _WorkerArgs) -> int:
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(1)
    except ImportError:  # pragma: no cover
        pass
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    space = SPACES[args.space]
    study = open_study(Path(args.path), args.name, space, seed=args.seed)
    cache = Path(args.cache_dir) if args.cache_dir else None
    objective = make_objective(
        load_tables(args.table_names), space, args.seasons, args.metric, cache
    )
    study.optimize(objective, n_trials=args.trials, catch=(ValueError, RuntimeError))
    return args.trials


def optimize(
    path: Path,
    name: str,
    space_name: str,
    *,
    trials: int,
    workers: int = 1,
    seasons: tuple[int, int] = splits.TUNE,
    metric: str = DEFAULT_METRIC,
    cache_dir: Path | None = None,
    tables: Tables | None = None,
    table_names: tuple[str, ...] = (),
) -> Any:
    """Run ``trials`` trials over ``workers`` spawned processes (each with its own sampler
    seed, reading ``table_names`` from the lake) and return the study. The search never
    sees the holdout."""
    splits.search_seasons(seasons)
    space = SPACES[space_name]
    if workers <= 1:
        study = open_study(path, name, space)
        data = tables if tables is not None else load_tables(table_names)
        study.optimize(
            make_objective(data, space, seasons, metric, cache_dir),
            n_trials=trials,
            catch=(ValueError, RuntimeError),
        )
        return study
    open_study(path, name, space)  # create it (and enqueue the baseline) once
    per = math.ceil(trials / workers)
    cache = str(cache_dir) if cache_dir else None
    jobs = [
        _WorkerArgs(str(path), name, space_name, per, seasons, metric, cache, seed, table_names)
        for seed in range(workers)
    ]
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as pool:
        list(pool.map(_work, jobs))
    return open_study(path, name, space)


# -- selection on select, with the one-SE rule ---------------------------------------------------
@dataclass
class Candidate:
    number: int
    params: dict[str, Any]
    tune: float
    select: float
    se: float  # SE of the paired difference to the best on select
    complexity: float
    baseline: bool


@dataclass
class Selection:
    chosen: Candidate
    candidates: list[Candidate]
    trials: int
    complete: int
    pruned: int
    null: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def select(
    study: Any,
    space_name: str,
    tables: Tables,
    *,
    top: int = 5,
    seasons: tuple[int, int] = splits.SELECT,
    metric: str = DEFAULT_METRIC,
    cache_dir: Path | None = None,
    null_flips: int = 2000,
) -> Selection:
    """Re-score the ``top`` trials (and the hand-set baseline) on *select*; choose the
    simplest within one SE of the best."""
    import optuna

    space = SPACES[space_name]
    splits.search_seasons(seasons)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
    ranked = sorted(done, key=lambda t: t.value)
    pool = ranked[:top]
    base = next((t for t in done if t.user_attrs.get("baseline")), None)
    if base is not None and base not in pool:
        pool.append(base)
    scored = []
    for t in pool:
        s = score_config(
            tables,
            space.build(t.params),
            seasons,
            metric=metric,
            target=space.target,
            cache_dir=cache_dir,
        )
        scored.append((t, s))
    best_t, best_s = min(scored, key=lambda ts: ts[1].mean)
    best = dict(zip(best_s.events, best_s.losses, strict=True))
    cands = []
    for t, s in scored:
        mine = dict(zip(s.events, s.losses, strict=True))
        common = [e for e in mine if e in best]
        diff = np.array([mine[e] - best[e] for e in common])
        se = float(diff.std(ddof=1) / math.sqrt(len(diff))) if len(diff) > 1 else 0.0
        cands.append(
            Candidate(
                t.number,
                dict(t.params),
                float(t.value),
                s.mean,
                se,
                space.complexity(t.params),
                bool(t.user_attrs.get("baseline")),
            )
        )
    best_c = next(c for c in cands if c.number == best_t.number)
    eligible = [c for c in cands if c.select <= best_c.select + c.se]
    chosen = min(eligible, key=lambda c: (c.complexity, c.select))
    return Selection(
        chosen,
        sorted(cands, key=lambda c: c.select),
        len(study.trials),
        len(done),
        pruned,
        null_inflation(study, flips=null_flips),
    )


def null_inflation(study: Any, *, flips: int = 2000, seed: int = 0) -> dict[str, float]:
    """The best observed improvement over the baseline on *tune* against its distribution
    under the null (random shared signs per event on every trial's differences)."""
    import optuna

    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    base = next((t for t in done if t.user_attrs.get("baseline")), None)
    if base is None or len(done) < 2:
        return {}
    ref = dict(zip(base.user_attrs["events"], base.user_attrs["losses"], strict=True))
    events = sorted(ref)
    rows = []
    for t in done:
        if t is base:
            continue
        mine = dict(
            zip(t.user_attrs.get("events", []), t.user_attrs.get("losses", []), strict=False)
        )
        if all(e in mine for e in events):
            rows.append([ref[e] - mine[e] for e in events])  # + = this trial is better
    if not rows:
        return {}
    d = np.array(rows)  # (trials, events)
    observed = float(d.mean(axis=1).max())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(flips, d.shape[1]))
    null_max = (signs @ d.T / d.shape[1]).max(axis=1)
    return {
        "best_improvement": observed,
        "null_mean_max": float(null_max.mean()),
        "null_q95": float(np.quantile(null_max, 0.95)),
        "p": float((null_max >= observed).mean()),
        "trials": float(d.shape[0]),
    }


def save_selection(sel: Selection, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sel.to_dict(), indent=1, default=str))
    return path
