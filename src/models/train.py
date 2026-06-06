"""Training orchestration: presets, time-series CV tuning, stacked-ensemble assembly,
memory profiling, benchmarking and experiment logging.

The headline feature is **training presets** (``test``/``light``/``medium``/``deep``/``max``).
A preset controls which base learners go into the stacked ensemble, how many Optuna trials
each one gets, the number of expanding-window CV folds, and the size of the chronological
holdout. Even the fast ``test`` preset runs *real* cross-validated tuning on the full real
dataset — there is no toy/synthetic shortcut — so its predictions are genuinely accurate.

Why this design is accurate **and** robust through the 2026 reset:

* The ensemble blends bagged trees and gradient boosting, so no single inductive bias
  dominates.
* Hyper-parameters are chosen by **expanding-window time-series CV** (train on past seasons,
  validate on the next), which mirrors real deployment and avoids leakage.
* The feature layer already injects causal ELO + a regulation-reset down-weight, so the model
  leans on driver skill when prior-car history is stale (e.g. 2026).
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config, load_config
from ..evaluation.metrics import aggregate_metrics, race_metrics
from ..features import build_features, session_target, session_feature_cols
from ..utils.logging import get_logger
from ..utils.profiling import profile
from .experiment import log_experiment
from .predictor import RacePredictor
from .registry import FAMILIES, build_model

log = get_logger()
warnings.filterwarnings("ignore", category=UserWarning)

# Human-readable model names for progress labels, so the bar shows e.g.
# "tuning LightGBM" / "fitting Random Forest" instead of raw family ids.
_PRETTY_FAMILY = {
    "random_forest": "Random Forest", "rf": "Random Forest",
    "extra_trees": "Extra Trees", "et": "Extra Trees",
    "lightgbm": "LightGBM", "lgbm": "LightGBM",
    "histgb": "HistGB", "hgb": "HistGB", "hist": "HistGB",
    "gbr": "Gradient Boosting",
    "ensemble": "Ensemble",
}


def _pretty(family: str) -> str:
    return _PRETTY_FAMILY.get(family, family.replace("_", " ").title())


# --------------------------------------------------------------------------- splitting
def time_split(feat_df: pd.DataFrame, holdout_seasons: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split chronologically: hold out the last ``holdout_seasons`` season(s)."""
    seasons = sorted(feat_df["season"].unique())
    cutoff = seasons[-holdout_seasons] if len(seasons) > holdout_seasons else seasons[-1]
    train = feat_df[feat_df["season"] < cutoff]
    valid = feat_df[feat_df["season"] >= cutoff]
    if len(train) == 0:
        n = int(len(feat_df) * 0.8)
        train, valid = feat_df.iloc[:n], feat_df.iloc[n:]
    return train, valid


def _cv_folds(feat_df: pd.DataFrame, folds: int) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Expanding-window season folds: train on past seasons, validate on the next one."""
    seasons = sorted(int(s) for s in feat_df["season"].unique())
    if len(seasons) <= 2:
        tr, va = time_split(feat_df)
        return [(tr, va)]
    val_seasons = seasons[-folds:] if folds < len(seasons) else seasons[1:]
    out = []
    for vs in val_seasons:
        tr = feat_df[feat_df["season"] < vs]
        va = feat_df[feat_df["season"] == vs]
        if len(tr) and len(va):
            out.append((tr, va))
    return out or [time_split(feat_df)]


# --------------------------------------------------------------------------- evaluation
def _race_mae(model, valid: pd.DataFrame, cols: list[str],
              target_col: str = "target_position") -> float:
    pred = model.predict(valid[cols].to_numpy())
    return float(np.abs(pred - valid[target_col].to_numpy()).mean())


def _evaluate(predictor: RacePredictor, valid: pd.DataFrame,
              target_col: str = "target_position") -> dict[str, float]:
    per_race = []
    for _, race in valid.groupby(["season", "round"]):
        race = race.copy()
        race["pred_position"] = predictor.predict_positions(race)
        per_race.append(race_metrics(race, actual_col=target_col))
    return aggregate_metrics(per_race)


# --------------------------------------------------------------------------- tuning
def _search_space(trial, family: str) -> dict:
    if family in ("random_forest", "rf", "extra_trees", "et"):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 900, step=100),
            "max_depth": trial.suggest_int("max_depth", 12, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
        }
    if family in ("lightgbm", "lgbm"):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 400, 1400, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 23, 160),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
        }
    if family in ("histgb", "hgb", "hist"):
        return {
            "max_iter": trial.suggest_int("max_iter", 300, 1000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 23, 160),
            "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 5.0),
        }
    # gbr
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 700, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 5),
    }


def tune_member(family: str, cfg: Config, cols: list[str],
                folds: list[tuple[pd.DataFrame, pd.DataFrame]], n_trials: int,
                progress=None, target_col: str = "target_position") -> dict:
    """Optuna search for one base learner against expanding-window CV MAE.

    ``progress``, if given, is called after every trial with
    ``(done, total, best_mae)`` so callers can render fine-grained, moving
    feedback (and a meaningful ETA) instead of a bar frozen on one family.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial) -> float:
        params = _search_space(trial, family)
        maes = []
        for tr, va in folds:
            model = build_model(family, cfg, params)
            model.fit(tr[cols].to_numpy(), tr[target_col].to_numpy())
            maes.append(_race_mae(model, va, cols, target_col))
        return float(np.mean(maes))

    def _on_trial(study, trial) -> None:
        if progress:
            try:
                best = study.best_value
            except ValueError:  # no completed trial yet
                best = float("nan")
            progress(trial.number + 1, n_trials, best)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False,
                   callbacks=[_on_trial] if progress else None)
    log.info(f"  tuned[{family}] CV-MAE={study.best_value:.4f}")
    return study.best_params


# --------------------------------------------------------------------------- fit
def _fit_predictor(family: str, cfg: Config, cols: list[str], train: pd.DataFrame,
                   members: list[str] | None = None,
                   member_params: dict[str, dict] | None = None,
                   params: dict | None = None,
                   target_col: str = "target_position") -> RacePredictor:
    model = build_model(family, cfg, params=params, members=members, member_params=member_params)
    Xtr, ytr = train[cols].to_numpy(), train[target_col].to_numpy()
    model.fit(Xtr, ytr)
    resid = ytr - model.predict(Xtr)
    return RacePredictor(model=model, feature_cols=cols, family=family,
                         residual_std=float(np.std(resid)) or 3.0)


PRESETS = ("test", "light", "medium", "deep", "max")

# Weekend sessions f1predict can model, each with its own production pointer file. The race
# pointer keeps its legacy name so existing installs and tooling continue to work unchanged.
SESSIONS = ("race", "qualifying", "sprint")
_PRODUCTION_PTR = {
    "race": "production.txt",
    "qualifying": "production_qualifying.txt",
    "sprint": "production_sprint.txt",
}

# Human-facing metadata for each preset: estimated wall-clock on an M5 Air, the ensemble it
# assembles, and what it's best suited for. Used by the CLI to render the overview + picker.
PRESET_INFO: dict[str, dict[str, str]] = {
    "test": {
        "eta": "~5-10 min", "ensemble": "RF + LightGBM",
        "best_for": "A quick but genuinely accurate model to get going.",
    },
    "light": {
        "eta": "~15 min", "ensemble": "RF + ExtraTrees + LightGBM",
        "best_for": "A solid everyday model with a bit more tuning.",
    },
    "medium": {
        "eta": "~40 min", "ensemble": "RF + ExtraTrees + LightGBM + HistGB",
        "best_for": "Strong accuracy; a good weekend-prep default.",
    },
    "deep": {
        "eta": "~1.5 hr", "ensemble": "5-model ensemble",
        "best_for": "Serious accuracy with deep hyper-parameter search.",
    },
    "max": {
        "eta": "~3-5 hr", "ensemble": "5-model ensemble",
        "best_for": "Maximum accuracy; leave it running.",
    },
}


def preset_overview_rows(cfg: Config | None = None) -> list[dict]:
    """Return per-preset display rows (merging config detail with PRESET_INFO)."""
    cfg = cfg or load_config()
    rows = []
    for name in PRESETS:
        pc = cfg.get(f"presets.{name}", {})
        info = PRESET_INFO[name]
        rows.append({
            "preset": name,
            "eta": info["eta"],
            "ensemble": info["ensemble"],
            "trials": int(pc.get("n_trials", 0)),
            "cv_folds": int(pc.get("cv_folds", 0)),
            "holdout": int(pc.get("holdout_seasons", 1)),
            "best_for": info["best_for"],
        })
    return rows


def train(
    cfg: Config | None = None,
    family: str | None = None,
    preset: str | None = None,
    do_tune: bool = False,
    n_trials: int = 40,
    profile_memory: bool = False,
    experiment_name: str = "default",
    compare_all: bool = False,
    df: pd.DataFrame | None = None,
    progress=None,
    session: str = "race",
) -> dict[str, Any]:
    """Train (optionally via a preset), evaluate, persist the predictor and log the run.

    ``session`` selects what the model predicts: ``"race"`` (finishing order, the default),
    ``"qualifying"`` (grid order — trained on a leakage-free feature set that excludes the
    current weekend's grid/quali) or ``"sprint"`` (sprint order — trained only on sprint
    weekends). Each session is persisted as its own production artifact.
    """
    cfg = cfg or load_config()
    session = (session or "race").lower()
    target_col = session_target(session)  # validates session

    # --- resolve preset --------------------------------------------------------------
    members = None
    cv_folds = 1
    holdout = 1
    if preset:
        preset = preset.lower()
        if preset not in PRESETS:
            raise ValueError(f"unknown preset {preset!r}; choose from {PRESETS}")
        pc = cfg.get(f"presets.{preset}", {})
        members = list(pc.get("members", cfg.get("model.ensemble.members")))
        n_trials = int(pc.get("n_trials", n_trials))
        cv_folds = int(pc.get("cv_folds", 4))
        holdout = int(pc.get("holdout_seasons", 1))
        do_tune = bool(pc.get("tune", True))
        family = family or "ensemble"
        log.info(f"preset [bold]{preset}[/bold]: members={members} trials={n_trials} "
                 f"cv_folds={cv_folds} holdout={holdout}")
    family = family or cfg.get("model.family", "ensemble")

    def _report(frac, label):
        if progress:
            progress(frac, label)

    _report(0.0, "loading dataset")
    if df is None:
        from ..data_pipeline import load_master
        df = load_master(cfg)
    _report(0.02, "building features")
    feat_df, all_cols = build_features(df, cfg)
    # Session-specific feature set + target. Qualifying drops grid/quali leakage; sprint
    # trains only on sprint weekends (where a sprint target actually exists).
    cols = session_feature_cols(all_cols, session)
    if session == "sprint":
        feat_df = feat_df[feat_df["has_sprint"] == 1].copy()
        n_sprint = feat_df[["season", "round"]].drop_duplicates().shape[0]
        if len(feat_df) == 0:
            raise ValueError(
                "no sprint weekends in the dataset — cannot train a sprint model. "
                "Sprints exist from 2021 onward; build/update real data first."
            )
        if n_sprint < 6:
            log.warning(f"only {n_sprint} sprint weekend(s) available — sprint model may be weak.")
    train_df, valid_df = time_split(feat_df, holdout_seasons=holdout)

    # --- benchmark path (compare single families) ------------------------------------
    if compare_all:
        benchmark, best = [], None
        for i, fam in enumerate(FAMILIES):
            _report(i / len(FAMILIES), f"fitting {_pretty(fam)} ({i + 1}/{len(FAMILIES)})")
            with profile(f"train[{fam}]") as prof:
                predictor = _fit_predictor(fam, cfg, cols, train_df, target_col=target_col)
            metrics = _evaluate(predictor, valid_df, target_col=target_col)
            benchmark.append({"family": fam, **metrics,
                              "peak_rss_mb": prof.peak_rss_mb, "train_seconds": prof.seconds})
            log.info(f"[{fam}] MAE={metrics['mae']:.3f} winner_acc={metrics['winner_accuracy']:.3f} "
                     f"mem={prof.peak_rss_mb:.0f}MB")
            if best is None or metrics["mae"] < best["metrics"]["mae"]:
                best = {"predictor": predictor, "metrics": metrics, "family": fam}
            _report((i + 1) / len(FAMILIES), f"{_pretty(fam)} · MAE {metrics['mae']:.3f}")
        return _finish(cfg, best, benchmark, cols, feat_df, experiment_name,
                       do_tune=False, family=best["family"], session=session)

    # --- tune members (real cross-validated search) ----------------------------------
    # Wall-clock is dominated by tuning, so it owns the bulk of the bar (5%→85%). We
    # advance once per *Optuna trial* (not once per family) so the bar moves smoothly
    # and Rich can show a real ETA, while the label names the exact model + best CV-MAE.
    member_params: dict[str, dict] = {}
    tuned_params = None
    if do_tune:
        folds = _cv_folds(train_df, cv_folds)
        tune_targets = members if (family == "ensemble" and members) else [family]
        if family == "ensemble" and not members:
            tune_targets = list(cfg.get("model.ensemble.members", FAMILIES))

        TUNE_LO, TUNE_HI = 0.05, 0.85
        total_trials = max(len(tune_targets) * n_trials, 1)

        def _tune_frac(done_trials: float) -> float:
            return TUNE_LO + (TUNE_HI - TUNE_LO) * (done_trials / total_trials)

        n_targets = len(tune_targets)
        for j, fam in enumerate(tune_targets):
            pretty = _pretty(fam)
            _report(_tune_frac(j * n_trials),
                    f"tuning {pretty} ({j + 1}/{n_targets}) · starting…")

            def _trial_cb(done, total, best, _fam=pretty, _j=j):
                frac = _tune_frac(_j * n_trials + done)
                mae = f"{best:.3f}" if best == best else "—"  # NaN guard
                _report(frac, f"tuning {_fam} ({_j + 1}/{n_targets}) · "
                              f"trial {done}/{total} · best CV-MAE {mae}")

            best_p = tune_member(fam, cfg, cols, folds, n_trials, progress=_trial_cb,
                                  target_col=target_col)
            member_params[fam] = best_p
        tuned_params = member_params.get(family) if family != "ensemble" else None

    # --- fit final model -------------------------------------------------------------
    _report(0.86, f"fitting final {_pretty(family)} on all training seasons")
    with profile(f"train[{family}]") as prof:
        predictor = _fit_predictor(
            family, cfg, cols, train_df,
            members=members,
            member_params=member_params or None,
            params=tuned_params,
            target_col=target_col,
        )
    _report(0.95, "evaluating on hold-out season")
    metrics = _evaluate(predictor, valid_df, target_col=target_col)
    log.info(f"[{family}] MAE={metrics['mae']:.3f} winner_acc={metrics['winner_accuracy']:.3f} "
             f"podium_acc={metrics['podium_accuracy']:.3f} mem={prof.peak_rss_mb:.0f}MB "
             f"({prof.seconds:.0f}s)")

    benchmark = [{"family": family, **metrics,
                  "peak_rss_mb": prof.peak_rss_mb, "train_seconds": prof.seconds}]
    best = {"predictor": predictor, "metrics": metrics, "family": family}
    _report(1.0, "done")
    return _finish(cfg, best, benchmark, cols, feat_df, experiment_name,
                   do_tune=do_tune, family=family, preset=preset,
                   member_params=member_params, members=members, session=session)


def _finish(cfg, best, benchmark, cols, feat_df, experiment_name,
            do_tune, family, preset=None, member_params=None, members=None, session="race"):
    best["predictor"].metadata = {
        "experiment_name": experiment_name,
        "preset": preset,
        "session": session,
        "target": session_target(session),
        "members": members,
        "member_params": member_params or {},
        "metrics": best["metrics"],
        "seasons": sorted(int(s) for s in feat_df["season"].unique()),
    }
    store = cfg.path("paths.models_store")
    # Race keeps the legacy ``{family}_{exp}.joblib`` name (back-compat); other sessions get a
    # session suffix, and each session has its own production pointer.
    suffix = "" if session == "race" else f"_{session}"
    model_path = store / f"{best['family']}_{experiment_name}{suffix}.joblib"
    best["predictor"].save(model_path)
    (store / _PRODUCTION_PTR[session]).write_text(str(model_path.name))

    log_experiment({
        "experiment_name": experiment_name,
        "family": best["family"],
        "preset": preset,
        "session": session,
        "tuned": do_tune,
        "n_features": len(cols),
        "metrics": best["metrics"],
        "benchmark": benchmark,
        "model_path": str(model_path.name),
        "data_seasons": sorted(int(s) for s in feat_df["season"].unique()),
    }, cfg)

    return {
        "predictor": best["predictor"],
        "metrics": best["metrics"],
        "family": best["family"],
        "benchmark": benchmark,
        "model_path": model_path,
        "feature_cols": cols,
    }


_PRODUCTION_CACHE: dict[str, tuple[float, "RacePredictor"]] = {}


def load_production(cfg: Config | None = None, session: str = "race") -> RacePredictor:
    cfg = cfg or load_config()
    session = (session or "race").lower()
    if session not in _PRODUCTION_PTR:
        raise ValueError(f"unknown session {session!r}; choose from {SESSIONS}")
    store = cfg.path("paths.models_store")
    ptr = store / _PRODUCTION_PTR[session]
    if not ptr.exists():
        if session == "race":
            log.warning("no production model -> training a quick ensemble (test preset)")
            return train(cfg, preset="test")["predictor"]
        log.warning(f"no production {session} model -> training a quick one (test preset)")
        return train(cfg, preset="test", session=session)["predictor"]
    model_path = store / ptr.read_text().strip()
    key = f"{session}:{model_path}"
    try:
        mtime = model_path.stat().st_mtime
    except OSError:
        mtime = -1.0
    cached = _PRODUCTION_CACHE.get(key)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    pred = RacePredictor.load(model_path)
    _PRODUCTION_CACHE[key] = (mtime, pred)
    return pred


def clear_model_cache() -> None:
    """Drop the cached production predictor (used by the shell `/reload`)."""
    _PRODUCTION_CACHE.clear()


# --------------------------------------------------------------------------- model store
def production_file(cfg: Config | None = None, session: str = "race") -> str | None:
    """Filename (e.g. ``ensemble_default.joblib``) of the active production model for ``session``."""
    cfg = cfg or load_config()
    ptr = cfg.path("paths.models_store") / _PRODUCTION_PTR.get(session, "production.txt")
    return ptr.read_text().strip() if ptr.exists() else None


def list_models(cfg: Config | None = None) -> list[dict]:
    """Enumerate every trained model in the store with its card metadata.

    Returns dicts with: ``name`` (easy slug == file stem), ``file``, ``family``,
    ``preset``, ``metrics``, ``size_mb``, ``modified`` and ``is_production``.
    """
    import json
    from datetime import datetime

    cfg = cfg or load_config()
    store = cfg.path("paths.models_store")
    active = production_file(cfg)
    out: list[dict] = []
    if not store.exists():
        return out
    for jb in sorted(store.glob("*.joblib")):
        card_path = jb.with_suffix(".card.json")
        card = {}
        if card_path.exists():
            try:
                card = json.loads(card_path.read_text())
            except (json.JSONDecodeError, OSError):
                card = {}
        stat = jb.stat()
        out.append({
            "name": jb.stem,
            "file": jb.name,
            "family": card.get("family", "?"),
            "preset": card.get("preset") or "—",
            "metrics": card.get("metrics", {}),
            "size_mb": stat.st_size / 1e6,
            "modified": datetime.fromtimestamp(stat.st_mtime),
            "is_production": jb.name == active,
        })
    # newest first
    out.sort(key=lambda r: r["modified"], reverse=True)
    return out


def resolve_model(query: str, cfg: Config | None = None) -> dict:
    """Resolve an easy/partial model name to a single model record.

    Matching is case-insensitive and tolerant: it tries (in order) exact stem,
    exact filename, then unique prefix, then unique substring. Raises
    ``ValueError`` with a helpful message on no/ambiguous match.
    """
    cfg = cfg or load_config()
    models = list_models(cfg)
    if not models:
        raise ValueError("No trained models found. Run `f1predict train` first.")
    q = query.strip().lower()

    exact = [m for m in models if m["name"].lower() == q or m["file"].lower() == q]
    if exact:
        return exact[0]
    prefix = [m for m in models if m["name"].lower().startswith(q)]
    if len(prefix) == 1:
        return prefix[0]
    substr = [m for m in models if q in m["name"].lower()]
    if len(substr) == 1:
        return substr[0]

    candidates = prefix or substr
    if not candidates:
        names = ", ".join(m["name"] for m in models)
        raise ValueError(f"No model matches '{query}'. Available: {names}")
    names = ", ".join(m["name"] for m in candidates)
    raise ValueError(f"'{query}' is ambiguous — matches: {names}")


def set_production(query: str, cfg: Config | None = None) -> dict:
    """Point the production pointer at the model matching ``query``. Returns the record."""
    cfg = cfg or load_config()
    rec = resolve_model(query, cfg)
    store = cfg.path("paths.models_store")
    (store / "production.txt").write_text(rec["file"])
    rec["is_production"] = True
    return rec


def delete_model(query: str, cfg: Config | None = None) -> dict:
    """Delete a trained model (its ``.joblib`` + ``.card.json``) from the store.

    Resolves ``query`` with the same easy/partial matching as :func:`resolve_model`,
    removes the model artifacts, and keeps the production pointer consistent: if the
    deleted model was active, the pointer is repointed to the newest remaining model
    (or cleared when the store becomes empty).

    Returns a dict with the deleted ``record``, whether it ``was_production``, and the
    name of the ``new_production`` model (or ``None``).
    """
    cfg = cfg or load_config()
    rec = resolve_model(query, cfg)
    store = cfg.path("paths.models_store")
    was_production = rec["is_production"]

    jb = store / rec["file"]
    removed: list[str] = []
    for path in (jb, jb.with_suffix(".card.json")):
        if path.exists():
            path.unlink()
            removed.append(path.name)

    new_production: str | None = None
    ptr = store / "production.txt"
    if was_production:
        if ptr.exists():
            ptr.unlink()
        remaining = list_models(cfg)
        if remaining:
            ptr.write_text(remaining[0]["file"])
            new_production = remaining[0]["name"]

    return {
        "record": rec,
        "removed": removed,
        "was_production": was_production,
        "new_production": new_production,
    }


