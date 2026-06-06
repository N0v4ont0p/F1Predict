"""Model registry — build regressors and the stacked ensemble.

We frame race prediction as **regression on finishing position** (lower == better) and rank
drivers within each race by predicted position. Six families are available:

* ``random_forest`` / ``extra_trees`` — bagged trees, robust, the workhorses.
* ``lightgbm`` / ``histgb`` / ``gbr`` — gradient boosting, strong on interactions.
* ``ensemble`` — a :class:`~sklearn.ensemble.StackingRegressor` that blends the above with a
  Ridge meta-learner. Stacking consistently beats any single family and, combined with the
  ELO/grid priors baked into the features, is far more robust through the 2026 regulation
  reset than a single tree model.

All are CPU-friendly and memory-efficient for the M5 / 24 GB target.
"""
from __future__ import annotations

from typing import Any

from ..config import Config

FAMILIES = ["random_forest", "extra_trees", "lightgbm", "histgb", "gbr"]


def _single(family: str, cfg: Config, params: dict[str, Any] | None = None):
    family = family.lower()
    overrides = params or {}
    seed = cfg.get("project.seed", 42)

    if family in ("random_forest", "rf"):
        from sklearn.ensemble import RandomForestRegressor
        p = dict(cfg.get("model.random_forest", {})); p.update(overrides)
        p.setdefault("random_state", seed)
        return RandomForestRegressor(**p)

    if family in ("extra_trees", "et"):
        from sklearn.ensemble import ExtraTreesRegressor
        p = dict(cfg.get("model.extra_trees", {})); p.update(overrides)
        p.setdefault("random_state", seed)
        return ExtraTreesRegressor(**p)

    if family in ("lightgbm", "lgbm"):
        from lightgbm import LGBMRegressor
        p = dict(cfg.get("model.lightgbm", {})); p.update(overrides)
        p.setdefault("random_state", seed); p.setdefault("verbosity", -1)
        return LGBMRegressor(**p)

    if family in ("histgb", "hgb", "hist"):
        from sklearn.ensemble import HistGradientBoostingRegressor
        p = dict(cfg.get("model.histgb", {})); p.update(overrides)
        p.setdefault("random_state", seed)
        return HistGradientBoostingRegressor(**p)

    if family in ("gbr", "gradient_boosting"):
        from sklearn.ensemble import GradientBoostingRegressor
        p = dict(cfg.get("model.gbr", {})); p.update(overrides)
        p.setdefault("random_state", seed)
        return GradientBoostingRegressor(**p)

    raise ValueError(f"unknown model family: {family!r}")


def build_model(family: str, cfg: Config, params: dict[str, Any] | None = None,
                members: list[str] | None = None,
                member_params: dict[str, dict] | None = None):
    """Instantiate an unfitted regressor (single family or stacked ensemble).

    ``member_params`` optionally supplies tuned hyper-parameters per ensemble member.
    """
    family = (family or "ensemble").lower()
    if family != "ensemble":
        return _single(family, cfg, params)

    from sklearn.ensemble import StackingRegressor
    from sklearn.linear_model import Ridge

    members = members or list(cfg.get("model.ensemble.members", FAMILIES))
    member_params = member_params or {}
    estimators = [(m, _single(m, cfg, member_params.get(m))) for m in members]
    meta_name = cfg.get("model.ensemble.meta", "ridge")
    final = Ridge(alpha=1.0) if meta_name == "ridge" else _single(meta_name, cfg)
    return StackingRegressor(
        estimators=estimators,
        final_estimator=final,
        passthrough=False,
        n_jobs=1,  # base learners already parallelise; avoid oversubscription on M5.
        cv=3,
    )
