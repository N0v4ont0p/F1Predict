"""Feature engineering factory (v2 — sophisticated, real-data, race-anchored).

**Temporal safety is the cardinal rule**: every feature for race *N* uses only information
available *strictly before* race *N*. We sort chronologically and use ``shift(1)`` / expanding
windows that exclude the current row, and a causal ELO engine whose ratings are recorded
*before* each race is scored. This makes backtests honest and predictions trustworthy.

Feature families (all configurable / versioned):

* ``grid``        — start slot & derived (pole, front row, top-10, pit-lane start).
* ``quali``       — qualifying position, gap-to-pole, gap-to-teammate, quali vs grid penalty.
* ``elo``         — causal driver & constructor ELO (the dominant skill/car prior).
* ``driver_form`` — multi-window rolling finishing position / points / finish-rate + EWMA
                    momentum + career expanding average + experience.
* ``constructor`` — constructor rolling pace, reliability (DNF rate) and momentum.
* ``track``       — driver & constructor affinity at this circuit + static track-type cluster.
* ``teammate``    — qualifying & race deltas vs the sister car (car-adjusted skill).
* ``regulation``  — era one-hots + 2026 reset flags that down-weight stale car history.
* ``sprint``      — sprint-weekend signals.
* ``advanced``    — interaction terms & non-linear transforms.

All numeric outputs are downcast to the configured float dtype to respect the 24 GB budget.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config, load_config
from ..utils.logging import get_logger
from .elo import compute_elo
from .regulations import era_onehot
from .tracks import track_profile

log = get_logger()

# Columns that must never be used as model inputs (outcomes / identifiers).
LEAKAGE_COLUMNS = {
    "position", "points", "status", "finished", "laps", "target_position", "dnf",
    "driver_name", "constructor_name", "race_name", "country", "locality", "date",
    "driver_id", "constructor_id", "circuit_id", "driver_code",
    "sprint_position", "sprint_points",
    "target_quali", "target_sprint",
}

_ID_KEEP = [
    "season", "round", "date", "race_name", "circuit_id", "country",
    "driver_id", "driver_code", "driver_name", "constructor_id", "constructor_name",
    "grid", "quali_position", "position", "points", "finished", "dnf",
    "has_sprint", "target_position", "target_quali", "target_sprint",
]

# --------------------------------------------------------------------------- sessions
# f1predict predicts three weekend sessions, each with its own regression target. The race
# is the dense, universal signal (always present), so its target doubles as the basis for
# every driver/constructor *form* feature; qualifying and sprint add their own targets.
SESSION_TARGETS = {
    "race": "target_position",
    "qualifying": "target_quali",
    "sprint": "target_sprint",
}

# Features that encode the *current* weekend's grid or qualifying outcome. They are perfectly
# legitimate inputs when predicting the race or the sprint (grid/quali are known beforehand),
# but for the **qualifying** model they ARE the thing being predicted — using them would leak
# the answer. ``session_feature_cols`` strips these for the qualifying target only.
QUALI_LEAKAGE_FEATURES = {
    "f_grid", "f_pit_start", "f_pole", "f_front_row", "f_top10_start",
    "f_quali_pos", "f_quali_gap_to_pole", "f_quali_gap_pct", "f_grid_penalty",
    "f_teammate_grid_delta", "f_teammate_quali_delta",
    "f_grid_x_drvform", "f_elo_x_grid", "f_quali_x_overtake", "f_grid_sq", "f_grid_log",
    "wx_rain_x_grid",
}


def session_target(session: str) -> str:
    """Return the target column for a weekend session (``race``/``qualifying``/``sprint``)."""
    try:
        return SESSION_TARGETS[session]
    except KeyError:
        raise ValueError(
            f"unknown session {session!r}; choose from {tuple(SESSION_TARGETS)}"
        )


def session_feature_cols(all_cols: list[str], session: str) -> list[str]:
    """Filter feature columns to those that are leakage-free for ``session``.

    Qualifying drops grid/quali-derived features (they encode the qualifying outcome);
    race and sprint keep the full set (grid & quali are known before those sessions run).
    """
    if session == "qualifying":
        return [c for c in all_cols if c not in QUALI_LEAKAGE_FEATURES]
    return list(all_cols)


def _target_position(df: pd.DataFrame) -> pd.Series:
    """Regression target: finishing position; DNFs ranked just beyond the classified field."""
    starters = df.groupby(["season", "round"])["driver_id"].transform("count")
    pos = df["position"].astype("float32").copy()
    dnf = df["position"] <= 0
    pos = pos.where(~dnf, starters.astype("float32") + 1)
    return pos.astype("float32")


def _target_quali(df: pd.DataFrame) -> pd.Series:
    """Qualifying target: classification position; unknown (0) ranked at the back of the field."""
    starters = df.groupby(["season", "round"])["driver_id"].transform("count")
    pos = df["quali_position"].astype("float32").copy()
    unknown = df["quali_position"] <= 0
    pos = pos.where(~unknown, starters.astype("float32") + 1)
    return pos.astype("float32")


def _target_sprint(df: pd.DataFrame) -> pd.Series:
    """Sprint target: sprint finishing position; non-finishers/absent ranked beyond the field.

    Only meaningful on sprint weekends (``has_sprint == 1``); training filters to those rows.
    """
    starters = df.groupby(["season", "round"])["driver_id"].transform("count")
    pos = df["sprint_position"].astype("float32").copy()
    unknown = df["sprint_position"] <= 0
    pos = pos.where(~unknown, starters.astype("float32") + 1)
    return pos.astype("float32")


def _roll(s: pd.Series, window: int) -> pd.Series:
    return s.shift(1).rolling(window, min_periods=1).mean()


def _ewm(s: pd.Series, span: int) -> pd.Series:
    return s.shift(1).ewm(span=span, min_periods=1).mean()


def build_features(
    df: pd.DataFrame,
    cfg: Config | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Return ``(features_df, feature_columns)`` with the target column attached."""
    cfg = cfg or load_config()
    windows = list(cfg.get("features.form_windows", [3, 5, 10]))
    fd = cfg.get("features.float_dtype", "float32")

    df = df.copy()
    # Chronological order is essential for every temporal-safe feature.
    df["_order"] = (
        df["season"].astype(int) * 100 + df["round"].astype(int)
    )
    df = df.sort_values(["_order", "position"]).reset_index(drop=True)
    df["target_position"] = _target_position(df)
    # Session-specific regression targets (used when training the qualifying / sprint models).
    df["target_quali"] = _target_quali(df)
    df["target_sprint"] = _target_sprint(df)

    # === causal ELO (driver + constructor) ============================================
    df = compute_elo(df)
    feats: list[str] = [
        "elo_driver", "elo_constructor", "elo_combined",
        "elo_driver_field_gap", "elo_constructor_field_gap",
    ]

    # === grid / start ==================================================================
    df["f_grid"] = df["grid"].replace(0, np.nan).astype("float32")
    df["f_grid"] = df["f_grid"].fillna(df["f_grid"].median())
    df["f_pit_start"] = (df["grid"] == 0).astype(fd)
    df["f_pole"] = (df["grid"] == 1).astype(fd)
    df["f_front_row"] = (df["grid"] <= 2).astype(fd)
    df["f_top10_start"] = (df["grid"] <= 10).astype(fd)
    feats += ["f_grid", "f_pit_start", "f_pole", "f_front_row", "f_top10_start"]

    # === qualifying ====================================================================
    df["f_quali_pos"] = df["quali_position"].replace(0, np.nan).astype("float32")
    df["f_quali_pos"] = df["f_quali_pos"].fillna(df["f_grid"])
    pole_ms = (
        df["quali_best_ms"].replace(0, np.nan)
        .groupby([df["season"], df["round"]]).transform("min")
    )
    gap = (df["quali_best_ms"].replace(0, np.nan) - pole_ms).astype("float32")
    df["f_quali_gap_to_pole"] = gap.fillna(gap.median() if gap.notna().any() else 0.0)
    # Pct gap is more comparable across eras/tracks than raw ms.
    df["f_quali_gap_pct"] = (df["f_quali_gap_to_pole"] / pole_ms).astype("float32").fillna(0.0)
    # Grid penalty: started behind where you qualified.
    df["f_grid_penalty"] = (df["f_grid"] - df["f_quali_pos"]).clip(lower=0).astype(fd)
    feats += ["f_quali_pos", "f_quali_gap_to_pole", "f_quali_gap_pct", "f_grid_penalty"]

    # === driver form (multi-window rolling + EWMA momentum) ============================
    gd = df.groupby("driver_id", group_keys=False)
    for w in windows:
        df[f"f_drv_pos_{w}"] = gd["target_position"].apply(lambda s: _roll(s, w)).astype(fd)
        df[f"f_drv_pts_{w}"] = gd["points"].apply(lambda s: _roll(s, w)).astype(fd)
        df[f"f_drv_finrate_{w}"] = gd["finished"].apply(lambda s: _roll(s, w)).astype(fd)
        feats += [f"f_drv_pos_{w}", f"f_drv_pts_{w}", f"f_drv_finrate_{w}"]
    # Rolling qualifying form (PAST quali positions only — leak-free; the core signal for the
    # qualifying model, which can't see the current weekend's grid/quali).
    df["_qpos_known"] = df["quali_position"].replace(0, np.nan).astype("float32")
    gdq = df.groupby("driver_id", group_keys=False)
    _quali_mid = float(df["target_quali"].median())
    for w in windows:
        df[f"f_quali_form_{w}"] = (
            gdq["_qpos_known"].apply(lambda s: _roll(s, w)).fillna(_quali_mid).astype(fd)
        )
        feats.append(f"f_quali_form_{w}")
    df["f_drv_momentum"] = gd["target_position"].apply(lambda s: _ewm(s, 4)).astype(fd)
    df["f_drv_pts_momentum"] = gd["points"].apply(lambda s: _ewm(s, 4)).astype(fd)
    # Trend: short window minus long window (improving form => negative).
    df["f_drv_form_trend"] = (df[f"f_drv_pos_{windows[0]}"] - df[f"f_drv_pos_{windows[-1]}"]).astype(fd)
    df["f_drv_career_pos"] = gd["target_position"].apply(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    ).astype(fd)
    df["f_drv_best_career_pos"] = gd["target_position"].apply(
        lambda s: s.shift(1).expanding(min_periods=1).min()
    ).astype(fd)
    df["f_drv_races"] = gd.cumcount().astype(fd)
    df["f_rookie"] = (df["f_drv_races"] < 5).astype(fd)
    df["f_experienced"] = (df["f_drv_races"] > 80).astype(fd)
    feats += [
        "f_drv_momentum", "f_drv_pts_momentum", "f_drv_form_trend",
        "f_drv_career_pos", "f_drv_best_career_pos", "f_drv_races",
        "f_rookie", "f_experienced",
    ]
    # Season-to-date form (resets each year).
    gds = df.groupby(["driver_id", "season"], group_keys=False)
    df["f_drv_season_pos"] = gds["target_position"].apply(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    ).astype(fd)
    df["f_drv_season_pts"] = gds["points"].apply(
        lambda s: s.shift(1).expanding(min_periods=1).sum()
    ).astype(fd)
    feats += ["f_drv_season_pos", "f_drv_season_pts"]

    # === constructor form & reliability ================================================
    gc = df.groupby("constructor_id", group_keys=False)
    for w in windows:
        df[f"f_con_pos_{w}"] = gc["target_position"].apply(lambda s: _roll(s, w)).astype(fd)
        df[f"f_con_finrate_{w}"] = gc["finished"].apply(lambda s: _roll(s, w)).astype(fd)
        df[f"f_con_dnf_{w}"] = gc["dnf"].apply(lambda s: _roll(s, w)).astype(fd)
        feats += [f"f_con_pos_{w}", f"f_con_finrate_{w}", f"f_con_dnf_{w}"]
    df["f_con_momentum"] = gc["target_position"].apply(lambda s: _ewm(s, 5)).astype(fd)
    gcs = df.groupby(["constructor_id", "season"], group_keys=False)
    df["f_con_season_pos"] = gcs["target_position"].apply(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    ).astype(fd)
    feats += ["f_con_momentum", "f_con_season_pos"]

    # === track affinity & type =========================================================
    gdc = df.groupby(["driver_id", "circuit_id"], group_keys=False)
    df["f_drv_track_pos"] = gdc["target_position"].apply(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    ).astype(fd)
    df["f_drv_track_starts"] = gdc.cumcount().astype(fd)
    gcc = df.groupby(["constructor_id", "circuit_id"], group_keys=False)
    df["f_con_track_pos"] = gcc["target_position"].apply(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    ).astype(fd)
    feats += ["f_drv_track_pos", "f_drv_track_starts", "f_con_track_pos"]
    # Static track-type cluster + characteristics.
    prof = df["circuit_id"].map(lambda c: track_profile(str(c))).apply(pd.Series)
    for col in prof.columns:
        df[col] = prof[col].astype(fd)
    feats += list(prof.columns)

    # === teammate deltas (car-adjusted skill) ==========================================
    grp = df.groupby(["season", "round", "constructor_id"])
    team_grid = grp["f_grid"].transform("mean")
    team_quali = grp["f_quali_pos"].transform("mean")
    df["f_teammate_grid_delta"] = (df["f_grid"] - team_grid).astype(fd)
    df["f_teammate_quali_delta"] = (df["f_quali_pos"] - team_quali).astype(fd)
    # Rolling qualifying head-to-head vs teammate (skill signal independent of car).
    df["_beat_tm_quali"] = (df["f_quali_pos"] < team_quali).astype("float32")
    df["f_tm_quali_winrate"] = gd["_beat_tm_quali"].apply(lambda s: _roll(s, 10)).astype(fd)
    feats += ["f_teammate_grid_delta", "f_teammate_quali_delta", "f_tm_quali_winrate"]

    # === regulation era awareness ======================================================
    era_df = df["season"].map(lambda s: era_onehot(int(s))).apply(pd.Series)
    for col in era_df.columns:
        df[col] = era_df[col].astype(fd)
    feats += list(era_df.columns)
    # Down-weight stale constructor history during a reset (esp. 2026).
    df["f_con_pos_reg_adj"] = (
        df[f"f_con_pos_{windows[-1]}"] * df["reg_constructor_trust"]
        + df["target_position"].median() * (1 - df["reg_constructor_trust"])
    ).astype(fd)
    feats.append("f_con_pos_reg_adj")

    # === sprint signals ================================================================
    df["f_has_sprint"] = df["has_sprint"].astype(fd)
    df["f_sprint_pos"] = gd["sprint_position"].apply(lambda s: _roll(s, 5)).astype(fd)
    feats += ["f_has_sprint", "f_sprint_pos"]

    # === season progress ===============================================================
    df["f_season_progress"] = (
        df["round"] / df.groupby("season")["round"].transform("max")
    ).astype(fd)
    feats.append("f_season_progress")

    # === advanced interactions / non-linear ============================================
    df["f_grid_x_drvform"] = (df["f_grid"] * df[f"f_drv_pos_{windows[0]}"].fillna(df["f_grid"])).astype(fd)
    df["f_elo_x_grid"] = (df["elo_driver_field_gap"] * df["f_grid"]).astype(fd)
    df["f_elo_x_power"] = (df["elo_constructor_field_gap"] * df["trk_power_sensitivity"]).astype(fd)
    df["f_quali_x_overtake"] = (df["f_quali_pos"] * df["trk_overtake_difficulty"]).astype(fd)
    df["f_grid_sq"] = (df["f_grid"] ** 2).astype(fd)
    df["f_grid_log"] = np.log1p(df["f_grid"]).astype(fd)
    feats += [
        "f_grid_x_drvform", "f_elo_x_grid", "f_elo_x_power",
        "f_quali_x_overtake", "f_grid_sq", "f_grid_log",
    ]

    # === weather & environment =========================================================
    # Weather reshapes a race: tyre working range (air temp), grip & cooling (humidity),
    # and above all rain. If the master was enriched via the Open-Meteo pipeline the
    # measured columns are present; otherwise we fall back to each circuit's researched
    # race-month climatology so these features are always populated (incl. future races).
    from .tracks import track_climate

    if "wx_temp_c" not in df.columns:
        clim = df["circuit_id"].map(lambda c: track_climate(str(c))).apply(pd.Series)
        for col in ("wx_temp_c", "wx_humidity", "wx_rain_prob", "wx_wind_kmh"):
            df[col] = clim[col]
    else:
        # Backfill any rows the enrichment missed with climatology (keeps it leak-free:
        # climatology is a static venue prior, not race-outcome information).
        clim = df["circuit_id"].map(lambda c: track_climate(str(c))).apply(pd.Series)
        for col in ("wx_temp_c", "wx_humidity", "wx_rain_prob", "wx_wind_kmh"):
            df[col] = df[col].fillna(clim[col]) if col in df else clim[col]

    df["wx_temp_norm"] = ((df["wx_temp_c"] - 10.0) / 25.0).clip(0, 1).astype(fd)
    df["wx_humidity"] = df["wx_humidity"].astype(fd)
    df["wx_rain_prob"] = df["wx_rain_prob"].astype(fd)
    df["wx_wind_norm"] = (df["wx_wind_kmh"] / 40.0).clip(0, 1).astype(fd)
    df["wx_is_hot"] = (df["wx_temp_c"] > 28).astype(fd)
    df["wx_is_wet"] = (df["wx_rain_prob"] > 0.4).astype(fd)
    # Thermal tyre stress: hot, abrasive, high-degradation tracks punish tyre management.
    df["f_tyre_stress"] = (
        df["trk_degradation"] * (0.5 + 0.5 * df["wx_temp_norm"]) * (0.6 + 0.4 * df["trk_abrasion"])
    ).astype(fd)
    # Wet weather is a great equaliser — it rewards driver skill (ELO) over car/grid.
    df["wx_rain_x_skill"] = (df["wx_rain_prob"] * df["elo_driver_field_gap"]).astype(fd)
    df["wx_rain_x_grid"] = (df["wx_rain_prob"] * df["f_grid"]).astype(fd)
    feats += [
        "wx_temp_norm", "wx_humidity", "wx_rain_prob", "wx_wind_norm",
        "wx_is_hot", "wx_is_wet", "f_tyre_stress", "wx_rain_x_skill", "wx_rain_x_grid",
    ]

    # === cold-start fills ==============================================================
    midfield = float(df["target_position"].median())
    for col in feats:
        if df[col].isna().any():
            fill = midfield if ("pos" in col or "track" in col) else 0.0
            df[col] = df[col].fillna(fill).astype(fd)

    keep = [c for c in _ID_KEEP if c in df.columns] + feats
    out = df[keep].copy()
    log.info(
        f"features built: [cyan]{len(feats)}[/cyan] cols x {len(out)} rows "
        f"(v={cfg.get('features.version')})"
    )
    return out, feats
