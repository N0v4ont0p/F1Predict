"""Auto-generated documentation: data dictionary + feature glossary + model card.

Keeps human docs in sync with code. ``f1predict``'s ``reporting`` package exposes these so
the dashboard and CLI can render living documentation.
"""
from __future__ import annotations

from ..data_pipeline.schema import RAW_COLUMNS

# Human-readable descriptions for engineered features (prefix ``f_`` / ``elo_`` / ``reg_`` / ``trk_``).
FEATURE_GLOSSARY: dict[str, str] = {
    "elo_driver": "Causal driver ELO rating (pre-race; skill prior that persists across reg resets).",
    "elo_constructor": "Causal constructor ELO rating (pre-race; regressed to mean at each reg reset).",
    "elo_combined": "0.6·driver + 0.4·constructor ELO (overall expected competitiveness).",
    "elo_driver_field_gap": "Driver ELO minus the race field's mean ELO.",
    "elo_constructor_field_gap": "Constructor ELO minus the race field's mean ELO.",
    "f_grid": "Starting grid position (0/unknown imputed to field median).",
    "f_pole": "1 if starting from pole position.",
    "f_front_row": "1 if starting on the front row (grid <= 2).",
    "f_top10_start": "1 if starting inside the top 10.",
    "f_pit_start": "1 if starting from the pit lane (grid == 0).",
    "f_quali_pos": "Qualifying classification (imputed from grid if unknown).",
    "f_quali_gap_to_pole": "Best qualifying lap gap to pole, in milliseconds.",
    "f_quali_gap_pct": "Qualifying gap to pole as a fraction of pole time (era-comparable).",
    "f_grid_penalty": "Grid slots lost to penalties (grid minus qualifying position).",
    "f_drv_pos_{w}": "Driver mean finishing position over the previous {w} races.",
    "f_drv_pts_{w}": "Driver mean points over the previous {w} races.",
    "f_drv_finrate_{w}": "Driver finish (classified) rate over the previous {w} races.",
    "f_drv_momentum": "EWMA of recent finishing position (recency-weighted form).",
    "f_drv_form_trend": "Short-window minus long-window form (negative == improving).",
    "f_drv_career_pos": "Driver career mean finishing position (expanding, excl. current).",
    "f_drv_best_career_pos": "Driver best career finishing position so far.",
    "f_drv_races": "Number of races contested before this one (experience).",
    "f_drv_season_pos": "Season-to-date mean finishing position.",
    "f_drv_season_pts": "Season-to-date points tally (before this race).",
    "f_rookie": "1 if fewer than 5 career races (cold-start flag).",
    "f_experienced": "1 if more than 80 career races.",
    "f_con_pos_{w}": "Constructor mean finishing position over previous {w} race-entries.",
    "f_con_finrate_{w}": "Constructor reliability (finish rate) over previous {w} entries.",
    "f_con_dnf_{w}": "Constructor DNF rate over previous {w} entries.",
    "f_con_momentum": "EWMA of constructor finishing position (recent pace).",
    "f_con_pos_reg_adj": "Constructor form down-weighted by regulation-reset trust (2026-aware).",
    "f_drv_track_pos": "Driver historical mean position at this circuit.",
    "f_con_track_pos": "Constructor historical mean position at this circuit.",
    "f_drv_track_starts": "Driver's number of prior starts at this circuit.",
    "trk_is_street": "1 if the venue is a street circuit.",
    "trk_is_high_speed": "1 if the venue is a high-speed/power circuit.",
    "trk_is_technical": "1 if the venue is a technical/downforce circuit.",
    "trk_overtake_difficulty": "How hard overtaking is (0 easy .. 1 very hard).",
    "trk_power_sensitivity": "How power-unit-sensitive the circuit is (0..1).",
    "f_teammate_grid_delta": "Driver grid minus team mean grid (car-adjusted qualifying skill).",
    "f_teammate_quali_delta": "Driver qualifying minus teammate (intra-team skill signal).",
    "f_tm_quali_winrate": "Rolling rate of out-qualifying the teammate.",
    "f_quali_form_{w}": "Driver mean *qualifying* position over the previous {w} weekends "
                        "(past-only; the core signal for the qualifying model).",
    "f_has_sprint": "1 if this weekend runs a sprint race.",
    "f_sprint_pos": "Driver mean sprint finishing position over the previous 5 sprints.",
    "reg_seasons_since_reset": "Seasons since the last major rule reset (0 == reset year).",
    "reg_is_reset_year": "1 in a regulation-reset year (2014/2017/2022/2026).",
    "reg_constructor_trust": "How much prior-car history is trusted this season (low in a reset).",
    "f_season_progress": "Round number normalised by season length (0..1).",
    "f_grid_x_drvform": "Interaction: grid position x recent driver form.",
    "f_elo_x_grid": "Interaction: driver ELO gap x grid position.",
    "f_elo_x_power": "Interaction: constructor ELO gap x circuit power-sensitivity.",
    "f_grid_sq": "Grid position squared (non-linear).",
    "f_grid_log": "log(1 + grid) (non-linear).",
}


def data_dictionary_markdown() -> str:
    lines = ["# Data Dictionary — master dataset\n",
             "One row == one driver's entry in one Grand Prix (qualifying & sprint merged in).\n",
             "| Column | Type | Description |", "| --- | --- | --- |"]
    desc = {
        "season": "Championship year.", "round": "Round number within the season.",
        "date": "Race date (ISO).", "race_name": "Grand Prix name.",
        "circuit_id": "Circuit identifier.", "country": "Host country.",
        "locality": "Host city/locality.", "driver_code": "3-letter driver code.",
        "quali_position": "Qualifying classification (0 == unknown).",
        "quali_best_ms": "Best qualifying lap time in ms (0 == unknown).",
        "dnf": "1 if mechanical/accident DNF.",
        "sprint_position": "Sprint finishing position (0 == no sprint).",
        "sprint_points": "Points scored in the sprint.",
        "has_sprint": "1 if the weekend had a sprint.",
        "driver_id": "Driver identifier.", "driver_name": "Driver full name.",
        "constructor_id": "Constructor identifier.", "constructor_name": "Constructor name.",
        "grid": "Starting position (0 == pit lane/unknown).",
        "position": "Classified finishing position (0 == DNF/unclassified).",
        "status": "Result status (Finished / +1 Lap / Accident / Engine / ...).",
        "points": "Championship points scored.", "laps": "Laps completed.",
        "finished": "1 if classified finish else 0.",
    }
    for col, dtype in RAW_COLUMNS.items():
        lines.append(f"| `{col}` | {dtype} | {desc.get(col, '')} |")
    return "\n".join(lines)


def feature_glossary_markdown(form_windows=(3, 5, 10)) -> str:
    lines = ["# Feature Glossary\n",
             "All features are temporal-safe (computed using only prior races).\n",
             "| Feature | Description |", "| --- | --- |"]
    for key, text in FEATURE_GLOSSARY.items():
        if "{w}" in key:
            for w in form_windows:
                lines.append(f"| `{key.format(w=w)}` | {text.format(w=w)} |")
        else:
            lines.append(f"| `{key}` | {text} |")
    return "\n".join(lines)
