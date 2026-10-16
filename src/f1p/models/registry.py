"""Models known to ``f1p eval`` and the forecasting commands."""

from __future__ import annotations

from typing import Any


def registered_models() -> dict[str, Any]:
    from f1p.models.quali_v1 import QUALI_FEATURES, QualiV1
    from f1p.models.race_sim import RaceSim
    from f1p.models.race_sim_v2 import RaceSimV2, load_knobs
    from f1p.models.world_v1 import WorldV1

    models: list[Any] = [
        WorldV1(),
        # ablations: each switches one ingredient off (or on)
        WorldV1(name="world_v1_plain", smooth_stages=False),
        WorldV1(name="world_v1_noret", retirements=False),
        WorldV1(name="world_v1_dry", weather=False),
        WorldV1(
            name="world_v1_nolap",
            features={"C1": ("rating",), "C4": ("rating", "grid", "quali")},
        ),
        WorldV1(  # the previous pace signal: EW clean-lap deficit (team + driver)
            name="world_v1_ewpace",
            features={
                "C1": ("rating", "race_pace"),
                "C4": ("rating", "grid", "quali", "race_pace"),
            },
        ),
        WorldV1(
            name="world_v1_aff",
            features={
                "C1": ("rating", "lap_pace", "team_affinity", "driver_affinity"),
                "C4": ("rating", "grid", "quali", "lap_pace", "team_affinity", "driver_affinity"),
            },
        ),
        WorldV1(name="world_v1_wet_temp", rain_temperature=True),
        WorldV1(name="world_v1_q", quali_obs=True),
        WorldV1(
            name="world_v1_2r",
            features={
                "C1": ("rating", "rating_q", "lap_pace"),
                "C4": ("rating", "rating_q", "grid", "quali", "lap_pace"),
            },
        ),
        WorldV1(
            name="world_v1_lap2",
            features={
                "C1": ("rating", "race_pace", "lap_pace"),
                "C4": ("rating", "grid", "quali", "race_pace", "lap_pace"),
            },
        ),
        WorldV1(  # plan M6 §A.7 gate 3: the Bayesian lap model's pace (mean, and mean x sd)
            name="world_v1_lapv2",
            features={
                "C1": ("rating", "lap_pace_v2", "lap_pace_v2_unc"),
                "C4": ("rating", "grid", "quali", "lap_pace_v2", "lap_pace_v2_unc"),
            },
            lap_v2="posteriors/lap_v2/default",
        ),
        RaceSim(),
        RaceSimV2(knobs=load_knobs()),
        *_challengers(),
        # qualifying target
        QualiV1(),
        QualiV1(name="quali_v1_rating", features={"C1": ("rating_q",), "C2": ("rating_q",)}),
        QualiV1(
            name="quali_v1_noprac",
            features={"C1": QUALI_FEATURES["C1"], "C2": QUALI_FEATURES["C1"]},
        ),
    ]
    return {m.name: m for m in models}


def _challengers() -> list[Any]:
    """The gradient-boosted rankers, when the optional ``ml`` extra is installed and loads
    (LightGBM needs OpenMP; on macOS that is ``brew install libomp``)."""
    import importlib

    from f1p.models.challengers import RankerModel

    out: list[Any] = []
    for module, kind in (("lightgbm", "lgbm"), ("catboost", "catboost")):
        try:
            importlib.import_module(module)
        except Exception:
            continue
        out.append(RankerModel(kind))
    return out
