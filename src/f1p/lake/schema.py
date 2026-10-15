"""The curated entity model (plan Appendix A) as explicit, validated table specs.

Every curated table is declared here once: column names, dtypes, primary key and required
columns. :func:`conform` casts a frame to the spec and :func:`validate` enforces it, so a
schema drift in a source fails the pipeline instead of silently reaching a model.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

UTC_TS = pl.Datetime("us", "UTC")


@dataclass(frozen=True, slots=True)
class TableSpec:
    name: str
    columns: dict[str, pl.DataType]
    primary_key: tuple[str, ...]
    required: tuple[str, ...] = ()
    description: str = ""

    @property
    def not_null(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.primary_key, *self.required)))


class SchemaError(ValueError):
    """A curated table does not match its spec."""


def conform(df: pl.DataFrame, spec: TableSpec) -> pl.DataFrame:
    """Select and cast ``df`` to ``spec`` (missing columns are an error, extras are dropped)."""
    missing = [c for c in spec.columns if c not in df.columns]
    if missing:
        raise SchemaError(f"{spec.name}: missing columns {missing}")
    return df.select(pl.col(c).cast(t, strict=True) for c, t in spec.columns.items())


def validate(df: pl.DataFrame, spec: TableSpec) -> None:
    """Raise :class:`SchemaError` unless ``df`` matches ``spec`` exactly."""
    expected = list(spec.columns.items())
    actual = list(df.schema.items())
    if actual != expected:
        raise SchemaError(f"{spec.name}: schema {actual} != expected {expected}")
    for col in spec.not_null:
        nulls = df[col].null_count()
        if nulls:
            raise SchemaError(f"{spec.name}: {nulls} null(s) in required column {col!r}")
    if spec.primary_key:
        dupes = df.group_by(spec.primary_key).len().filter(pl.col("len") > 1)
        if dupes.height:
            sample = dupes.head(5).to_dicts()
            raise SchemaError(f"{spec.name}: duplicate primary keys, e.g. {sample}")


_S, _I, _F, _B, _D = pl.String(), pl.Int32(), pl.Float64(), pl.Boolean(), pl.Date()

SEASONS = TableSpec(
    "seasons",
    {"season": _I, "rounds": _I, "era_start": _I, "is_reset": _B, "seasons_since_reset": _I},
    ("season",),
    ("rounds", "era_start", "is_reset", "seasons_since_reset"),
)

CIRCUITS = TableSpec(
    "circuits",
    {
        "circuit_id": _S,
        "name": _S,
        "full_name": _S,
        "circuit_type": _S,
        "direction": _S,
        "place_name": _S,
        "country_id": _S,
        "latitude": _F,
        "longitude": _F,
        "length_km": _F,
        "turns": _I,
    },
    ("circuit_id",),
    ("name", "latitude", "longitude"),
)

CIRCUIT_LAYOUTS = TableSpec(
    "circuit_layouts",
    {"layout_id": _S, "circuit_id": _S, "length_km": _F, "turns": _I},
    ("layout_id",),
    ("circuit_id",),
)

DRIVERS = TableSpec(
    "drivers",
    {
        "driver_id": _S,
        "name": _S,
        "first_name": _S,
        "last_name": _S,
        "abbreviation": _S,
        "permanent_number": _S,
        "gender": _S,
        "date_of_birth": _D,
        "nationality_country_id": _S,
    },
    ("driver_id",),
    ("name",),
)

CONSTRUCTORS = TableSpec(
    "constructors",
    {"constructor_id": _S, "name": _S, "full_name": _S, "country_id": _S},
    ("constructor_id",),
    ("name",),
)

ENGINE_MANUFACTURERS = TableSpec(
    "engine_manufacturers",
    {"engine_manufacturer_id": _S, "name": _S, "country_id": _S},
    ("engine_manufacturer_id",),
    ("name",),
)

CONSTRUCTOR_SEASONS = TableSpec(
    "constructor_seasons",
    {"season": _I, "constructor_id": _S, "lineage_id": _S},
    ("season", "constructor_id"),
    ("lineage_id",),
    "Team lineage per season (e.g. jordan 1991 → aston_martin lineage).",
)

EVENTS = TableSpec(
    "events",
    {
        "event_id": _S,
        "season": _I,
        "round": _I,
        "grand_prix_id": _S,
        "name": _S,
        "circuit_id": _S,
        "layout_id": _S,
        "circuit_type": _S,
        "race_date": _D,
        "race_start_utc": UTC_TS,
        "laps": _I,
        "scheduled_laps": _I,
        "distance_km": _F,
        "scheduled_distance_km": _F,
        "distance_fraction": _F,
        "has_sprint": _B,
        "qualifying_format": _S,
    },
    ("event_id",),
    ("season", "round", "circuit_id", "race_date", "has_sprint"),
)

SESSIONS = TableSpec(
    "sessions",
    {"event_id": _S, "session": _S, "session_date": _D, "start_utc": UTC_TS},
    ("event_id", "session"),
)

ENTRIES = TableSpec(
    "entries",
    {
        "season": _I,
        "round": _I,
        "driver_id": _S,
        "constructor_id": _S,
        "engine_manufacturer_id": _S,
        "entrant_id": _S,
        "test_driver": _B,
    },
    (),
    ("season", "driver_id", "constructor_id", "test_driver"),
    "Season entry lists; race drivers are exploded per round, test drivers have round=null.",
)

RESULTS = TableSpec(
    "results",
    {
        "event_id": _S,
        "season": _I,
        "round": _I,
        "session": _S,
        "display_order": _I,
        "driver_id": _S,
        "constructor_id": _S,
        "engine_manufacturer_id": _S,
        "car_number": _S,
        "position": _I,
        "position_text": _S,
        "classified": _B,
        "status": _S,
        "reason_retired": _S,
        "laps": _I,
        "time_ms": pl.Int64(),
        "gap_laps": _I,
        "points": _F,
        "grid": _I,
        "qualifying_position": _I,
        "fastest_lap": _B,
        "shared_car": _B,
        "observed_at": UTC_TS,
    },
    ("event_id", "session", "display_order"),
    ("season", "round", "driver_id", "constructor_id", "classified", "status", "observed_at"),
    "Race and sprint classifications, one row per car entry.",
)

QUALIFYING = TableSpec(
    "qualifying",
    {
        "event_id": _S,
        "session": _S,
        "display_order": _I,
        "driver_id": _S,
        "constructor_id": _S,
        "position": _I,
        "position_text": _S,
        "best_ms": pl.Int64(),
        "q1_ms": pl.Int64(),
        "q2_ms": pl.Int64(),
        "q3_ms": pl.Int64(),
        "laps": _I,
        "observed_at": UTC_TS,
    },
    ("event_id", "session", "display_order"),
    ("driver_id", "constructor_id", "observed_at"),
)

GRID = TableSpec(
    "grid",
    {
        "event_id": _S,
        "session": _S,
        "display_order": _I,
        "driver_id": _S,
        "constructor_id": _S,
        "grid_position": _I,
        "position_text": _S,
        "qualification_position": _I,
        "grid_penalty": _S,
        "grid_penalty_positions": _I,
        "observed_at": UTC_TS,
    },
    ("event_id", "session", "display_order"),
    ("driver_id", "constructor_id", "observed_at"),
)

PRACTICE = TableSpec(
    "practice",
    {
        "event_id": _S,
        "session": _S,
        "display_order": _I,
        "driver_id": _S,
        "constructor_id": _S,
        "position": _I,
        "best_ms": pl.Int64(),
        "gap_ms": pl.Int64(),
        "laps": _I,
        "observed_at": UTC_TS,
    },
    ("event_id", "session", "display_order"),
    ("driver_id", "constructor_id", "observed_at"),
)

PIT_STOPS = TableSpec(
    "pit_stops",
    {
        "event_id": _S,
        "driver_id": _S,
        "constructor_id": _S,
        "stop": _I,
        "lap": _I,
        "duration_ms": pl.Int64(),
        "observed_at": UTC_TS,
    },
    ("event_id", "driver_id", "stop"),
    ("constructor_id", "observed_at"),
)

FASTEST_LAPS = TableSpec(
    "fastest_laps",
    {
        "event_id": _S,
        "display_order": _I,
        "driver_id": _S,
        "constructor_id": _S,
        "position": _I,
        "lap": _I,
        "time_ms": pl.Int64(),
        "observed_at": UTC_TS,
    },
    ("event_id", "display_order"),
    ("driver_id", "observed_at"),
)

DRIVER_STANDINGS = TableSpec(
    "driver_standings",
    {
        "event_id": _S,
        "season": _I,
        "round": _I,
        "display_order": _I,
        "driver_id": _S,
        "position": _I,
        "position_text": _S,
        "points": _F,
        "observed_at": UTC_TS,
    },
    ("event_id", "display_order"),
    ("season", "round", "driver_id", "points", "observed_at"),
)

CONSTRUCTOR_STANDINGS = TableSpec(
    "constructor_standings",
    {
        "event_id": _S,
        "season": _I,
        "round": _I,
        "display_order": _I,
        "constructor_id": _S,
        "engine_manufacturer_id": _S,
        "position": _I,
        "position_text": _S,
        "points": _F,
        "observed_at": UTC_TS,
    },
    ("event_id", "display_order"),
    ("season", "round", "constructor_id", "points", "observed_at"),
)

CURATED: tuple[TableSpec, ...] = (
    SEASONS,
    CIRCUITS,
    CIRCUIT_LAYOUTS,
    DRIVERS,
    CONSTRUCTORS,
    ENGINE_MANUFACTURERS,
    CONSTRUCTOR_SEASONS,
    EVENTS,
    SESSIONS,
    ENTRIES,
    RESULTS,
    QUALIFYING,
    GRID,
    PRACTICE,
    PIT_STOPS,
    FASTEST_LAPS,
    DRIVER_STANDINGS,
    CONSTRUCTOR_STANDINGS,
)
WEATHER_FORECAST = TableSpec(
    "weather_forecast",
    {
        "event_id": _S,
        "kind": _S,
        "wet_prob": _F,
        "precip_mm": _F,
        "temp_c": _F,
        "observed_at": UTC_TS,
    },
    ("event_id", "kind"),
    ("wet_prob", "observed_at"),
    "What could be known about race weather in advance: 'climatology' (prior years only) "
    "and 'forecast' (archived or live model forecast for the race window).",
)

WEATHER_OBSERVED = TableSpec(
    "weather_observed",
    {
        "event_id": _S,
        "window_start_utc": UTC_TS,
        "precip_mm": _F,
        "max_hourly_mm": _F,
        "temp_c": _F,
        "wet": _B,
        "observed_at": UTC_TS,
    },
    ("event_id",),
    ("window_start_utc", "precip_mm", "wet", "observed_at"),
    "Reanalysis (ERA5) weather over the race window — known only once the race is run.",
)

WEATHER: tuple[TableSpec, ...] = (WEATHER_FORECAST, WEATHER_OBSERVED)

RACE_PACE = TableSpec(
    "race_pace",
    {
        "event_id": _S,
        "driver_id": _S,
        "car_number": _S,
        "clean_laps": _I,
        "median_lap_ms": pl.Int64(),
        "pace_deficit_pct": _F,
        "observed_at": UTC_TS,
    },
    ("event_id", "driver_id"),
    ("clean_laps", "median_lap_ms", "pace_deficit_pct", "observed_at"),
    "Clean-lap race pace per car (live timing, 2018+): median lap and % deficit to the fastest.",
)

TIMING: tuple[TableSpec, ...] = (RACE_PACE,)
SPECS: dict[str, TableSpec] = {s.name: s for s in (*CURATED, *WEATHER, *TIMING)}
