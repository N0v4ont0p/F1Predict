"""Canonical schema for the f1predict master dataset (v2 — real-data enriched).

One row == one driver's entry in one Grand Prix weekend (the **race** is the anchor, with
qualifying and sprint merged in as columns). This denormalised shape is ideal for the
feature factory and keeps everything in a single Parquet file.
"""
from __future__ import annotations

RAW_COLUMNS: dict[str, str] = {
    # --- identifiers ---------------------------------------------------------------
    "season": "int32",
    "round": "int32",
    "date": "str",
    "race_name": "str",
    "circuit_id": "str",
    "country": "str",
    "locality": "str",
    "driver_id": "str",
    "driver_code": "str",
    "driver_name": "str",
    "constructor_id": "str",
    "constructor_name": "str",
    # --- qualifying ----------------------------------------------------------------
    "quali_position": "int32",   # 0 == no/unknown qualifying
    "quali_best_ms": "float32",  # best Q time in milliseconds (0 == unknown)
    # --- race ----------------------------------------------------------------------
    "grid": "int32",             # actual race start slot (after penalties; 0 == pit lane)
    "position": "int32",         # classified finishing position (0 == DNF/unclassified)
    "status": "str",
    "points": "float32",
    "laps": "int32",
    "finished": "int32",         # 1 if classified finish
    "dnf": "int32",              # 1 if mechanical/accident DNF
    # --- sprint (2021+) ------------------------------------------------------------
    "sprint_position": "int32",  # 0 if no sprint that weekend
    "sprint_points": "float32",
    "has_sprint": "int32",
}

KEY_COLUMNS = ["season", "round", "driver_id"]


def empty_like() -> dict[str, list]:
    return {col: [] for col in RAW_COLUMNS}
