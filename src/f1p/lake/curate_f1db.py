"""F1DB SQLite → curated tables (plan §6.2, Appendix A).

Conventions applied here, once, for every downstream consumer:

* ids are F1DB ids with ``-`` → ``_`` (``max-verstappen`` → ``max_verstappen``);
* ``event_id`` is the :class:`~f1p.core.ids.EventKey` string (``2026-R17``);
* sessions use :class:`~f1p.core.ids.SessionType` values (``race``, ``sprint``, ``quali``,
  ``sq``, ``fp1``…);
* result status uses :class:`~f1p.core.status.StatusClass` (fixes legacy defect D1);
* every fact carries ``observed_at`` from :mod:`f1p.lake.timing`.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import polars as pl

from f1p.core import regulations
from f1p.core.ids import SessionType
from f1p.core.status import StatusClass, classify_status
from f1p.lake import schema as S
from f1p.lake import timing

_ID_COLUMNS = (
    "driver_id",
    "constructor_id",
    "engine_manufacturer_id",
    "circuit_id",
    "layout_id",
    "grand_prix_id",
    "country_id",
    "nationality_country_id",
    "entrant_id",
    "lineage_id",
)


def _ids(df: pl.DataFrame) -> pl.DataFrame:
    """Normalise every id column from kebab-case to snake_case."""
    cols = [c for c in _ID_COLUMNS if c in df.columns]
    return df.with_columns(pl.col(cols).str.replace_all("-", "_"))


class F1dbReader:
    """Thin read-only accessor over an F1DB SQLite file."""

    def __init__(self, path: Path):
        self.path = path
        self._con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> F1dbReader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def query(self, sql: str) -> pl.DataFrame:
        cur = self._con.execute(sql)
        names = [d[0] for d in cur.description]
        return pl.DataFrame(cur.fetchall(), schema=names, orient="row", infer_schema_length=None)


# --------------------------------------------------------------------------- status
_POSITION_TEXT_STATUS = {
    "NC": StatusClass.NOT_CLASSIFIED,
    "DSQ": StatusClass.DSQ,
    "EX": StatusClass.DSQ,
    "DNS": StatusClass.DNS,
    "DNP": StatusClass.DNS,
    "DNQ": StatusClass.DNQ,
    "DNPQ": StatusClass.DNPQ,
}


def result_status(
    position: int | None, position_text: str | None, reason: str | None, gap_laps: int | None
) -> StatusClass:
    """Status for one F1DB race/sprint result row."""
    if position is not None:
        if reason:
            cls = classify_status(reason)
            if cls.retired:
                return cls  # classified despite retiring late (≥ 90% distance)
        return StatusClass.LAPPED if gap_laps and gap_laps > 0 else StatusClass.FINISHED
    text = (position_text or "").upper()
    if text in _POSITION_TEXT_STATUS:
        return _POSITION_TEXT_STATUS[text]
    cls = classify_status(reason or "Retired")
    return cls if cls.retired else StatusClass.DNF_OTHER


# --------------------------------------------------------------------------- events
_EVENT_SQL = """
SELECT r.id AS race_id, r.year AS season, r.round, r.grand_prix_id, r.official_name AS name,
       r.circuit_id, r.circuit_layout_id AS layout_id, r.circuit_type,
       r.date AS race_date, r.time AS race_time,
       r.laps, r.scheduled_laps, r.distance AS distance_km,
       r.scheduled_distance AS scheduled_distance_km, r.qualifying_format,
       r.free_practice_1_date AS fp1_date, r.free_practice_1_time AS fp1_time,
       r.free_practice_2_date AS fp2_date, r.free_practice_2_time AS fp2_time,
       r.free_practice_3_date AS fp3_date, r.free_practice_3_time AS fp3_time,
       r.sprint_qualifying_date AS sq_date, r.sprint_qualifying_time AS sq_time,
       r.sprint_race_date AS sprint_date, r.sprint_race_time AS sprint_time,
       r.qualifying_date AS quali_date, r.qualifying_time AS quali_time,
       (SELECT COUNT(*) FROM sprint_race_result s WHERE s.race_id = r.id) AS sprint_rows,
       (SELECT rr.time_millis FROM race_result rr
         WHERE rr.race_id = r.id AND rr.position_number = 1 LIMIT 1) AS winner_ms,
       (SELECT s.time_millis FROM sprint_race_result s
         WHERE s.race_id = r.id AND s.position_number = 1 LIMIT 1) AS sprint_winner_ms
FROM race r
"""

_DATE_COLS = (
    "race_date",
    "fp1_date",
    "fp2_date",
    "fp3_date",
    "sq_date",
    "sprint_date",
    "quali_date",
)


def _load_events(db: F1dbReader) -> pl.DataFrame:
    df = db.query(_EVENT_SQL)
    df = df.with_columns(pl.col(c).str.to_date("%Y-%m-%d") for c in _DATE_COLS)
    return _ids(df).with_columns(
        pl.format("{}-R{}", pl.col("season"), pl.col("round").cast(pl.String).str.zfill(2)).alias(
            "event_id"
        )
    )


def _start(row: dict[str, object], prefix: str) -> datetime | None:
    day = row.get(f"{prefix}_date")
    clock = row.get(f"{prefix}_time")
    return timing.utc_datetime(
        day if isinstance(day, date) else None, clock if isinstance(clock, str) else None
    )


def _timing_frame(events: pl.DataFrame) -> pl.DataFrame:
    """Per-event observed_at for every kind of fact, computed with :mod:`f1p.lake.timing`."""
    out = []
    for row in events.iter_rows(named=True):
        race_date: date = row["race_date"]
        race_start = _start(row, "race")
        out.append(
            {
                "race_id": row["race_id"],
                "race_start_utc": race_start,
                "obs_race": timing.race_results_observed(race_date, race_start, row["winner_ms"]),
                "obs_sprint": timing.sprint_results_observed(
                    race_date, _start(row, "sprint"), row["sprint_winner_ms"]
                ),
                "obs_quali": timing.qualifying_observed(race_date, _start(row, "quali")),
                "obs_sq": timing.qualifying_observed(race_date, _start(row, "sq"), sprint=True),
                "obs_grid": timing.grid_observed(race_date, race_start),
                "obs_sprint_grid": timing.grid_observed(race_date, _start(row, "sprint")),
                "obs_fp1": timing.practice_observed(race_date, _start(row, "fp1")),
                "obs_fp2": timing.practice_observed(race_date, _start(row, "fp2")),
                "obs_fp3": timing.practice_observed(race_date, _start(row, "fp3")),
            }
        )
    return pl.DataFrame(out, schema_overrides={k: S.UTC_TS for k in out[0] if k != "race_id"})


def build_events(raw: pl.DataFrame, tim: pl.DataFrame) -> pl.DataFrame:
    df = raw.join(tim.select("race_id", "race_start_utc"), on="race_id")
    frac_dist = pl.col("distance_km") / pl.col("scheduled_distance_km")
    frac_laps = pl.col("laps") / pl.col("scheduled_laps")
    df = df.with_columns(
        pl.coalesce(frac_dist, frac_laps, pl.lit(1.0)).clip(0.0, 1.0).alias("distance_fraction"),
        (pl.col("sprint_rows") > 0).alias("has_sprint"),
    )
    return S.conform(df, S.EVENTS).sort("season", "round")


def build_sessions(raw: pl.DataFrame) -> pl.DataFrame:
    parts = []
    for prefix, session in (
        ("fp1", SessionType.FP1),
        ("fp2", SessionType.FP2),
        ("fp3", SessionType.FP3),
        ("sq", SessionType.SPRINT_QUALIFYING),
        ("sprint", SessionType.SPRINT),
        ("quali", SessionType.QUALIFYING),
        ("race", SessionType.RACE),
    ):
        day, clock = pl.col(f"{prefix}_date"), pl.col(f"{prefix}_time")
        parts.append(
            raw.filter(day.is_not_null()).select(
                "event_id",
                pl.lit(session.value).alias("session"),
                day.alias("session_date"),
                pl.when(clock.is_not_null())
                .then(
                    pl.concat_str(
                        day.cast(pl.String), pl.lit("T"), clock, pl.lit(":00Z")
                    ).str.to_datetime("%Y-%m-%dT%H:%M:%SZ", time_zone="UTC", time_unit="us")
                )
                .alias("start_utc"),
            )
        )
    return S.conform(pl.concat(parts), S.SESSIONS).sort("event_id", "session")


# --------------------------------------------------------------------------- results
_RESULT_COLS = """
    rr.race_id, rr.position_display_order AS display_order, rr.position_number AS position,
    rr.position_text, rr.driver_number AS car_number, rr.driver_id, rr.constructor_id,
    rr.engine_manufacturer_id, rr.laps, rr.time_millis AS time_ms, rr.gap_laps,
    rr.reason_retired, rr.points, rr.grid_position_number AS grid,
    rr.qualification_position_number AS qualifying_position
"""


def build_results(
    db: F1dbReader, events: pl.DataFrame, tim: pl.DataFrame, notes: list[str]
) -> pl.DataFrame:
    race = db.query(
        f"SELECT {_RESULT_COLS}, rr.fastest_lap, rr.shared_car FROM race_result rr"
    ).with_columns(pl.lit(SessionType.RACE.value).alias("session"))
    sprint = db.query(
        f"SELECT {_RESULT_COLS}, 0 AS fastest_lap, 0 AS shared_car FROM sprint_race_result rr"
    ).with_columns(pl.lit(SessionType.SPRINT.value).alias("session"))
    df = _ids(pl.concat([race, sprint], how="diagonal_relaxed"))
    # A retirement with zero laps cannot be classified, yet the source occasionally gives one a
    # finishing position (cars eliminated at an aborted start, e.g. 1987 Austria). Such rows
    # are de-classified; Jolpica/Ergast lists the same entries as retirements.
    bogus = (
        pl.col("position").is_not_null()
        & (pl.col("laps") == 0)
        & pl.col("reason_retired").is_not_null()
    )
    n_bogus = df.filter(bogus).height
    if n_bogus:
        notes.append(
            f"results: de-classified {n_bogus} zero-lap retirement(s) listed with a position"
        )
        df = df.with_columns(
            pl.when(bogus).then(None).otherwise(pl.col("position")).alias("position"),
            pl.when(bogus)
            .then(pl.lit("DNF"))
            .otherwise(pl.col("position_text"))
            .alias("position_text"),
        )
    df = df.with_columns(
        pl.struct("position", "position_text", "reason_retired", "gap_laps")
        .map_elements(
            lambda r: (
                result_status(
                    r["position"], r["position_text"], r["reason_retired"], r["gap_laps"]
                ).value
            ),
            return_dtype=pl.String,
        )
        .alias("status"),
        pl.col("position").is_not_null().alias("classified"),
        pl.col("points").fill_null(0.0),
        pl.col("fastest_lap").cast(pl.Boolean).fill_null(False),
        pl.col("shared_car").cast(pl.Boolean).fill_null(False),
    )
    df = df.join(_race_keys(events, tim), on="race_id", how="inner")
    df = df.with_columns(
        pl.when(pl.col("session") == SessionType.SPRINT.value)
        .then(pl.col("obs_sprint"))
        .otherwise(pl.col("obs_race"))
        .alias("observed_at")
    )
    return S.conform(df, S.RESULTS).sort("event_id", "session", "display_order")


def _race_keys(events: pl.DataFrame, tim: pl.DataFrame) -> pl.DataFrame:
    return events.select("race_id", "event_id", "season", "round").join(tim, on="race_id")


def _session_table(
    db: F1dbReader,
    sources: list[tuple[str, SessionType, str]],
    keys: pl.DataFrame,
    select_sql: str,
    spec: S.TableSpec,
    rename: Callable[[pl.DataFrame], pl.DataFrame] = lambda d: d,
) -> pl.DataFrame:
    parts = []
    for table, session, obs_col in sources:
        part = db.query(f"SELECT {select_sql} FROM {table} t").with_columns(
            pl.lit(session.value).alias("session")
        )
        part = _ids(rename(part)).join(keys, on="race_id", how="inner")
        parts.append(part.with_columns(pl.col(obs_col).alias("observed_at")))
    return S.conform(pl.concat(parts, how="diagonal_relaxed"), spec).sort(
        "event_id", "session", "display_order"
    )


def build_qualifying(db: F1dbReader, keys: pl.DataFrame) -> pl.DataFrame:
    sql = """t.race_id, t.position_display_order AS display_order, t.position_number AS position,
             t.position_text, t.driver_id, t.constructor_id, t.time_millis AS best_ms,
             t.q1_millis AS q1_ms, t.q2_millis AS q2_ms, t.q3_millis AS q3_ms, t.laps"""
    return _session_table(
        db,
        [
            ("qualifying_result", SessionType.QUALIFYING, "obs_quali"),
            ("sprint_qualifying_result", SessionType.SPRINT_QUALIFYING, "obs_sq"),
        ],
        keys,
        sql,
        S.QUALIFYING,
    )


def build_grid(db: F1dbReader, keys: pl.DataFrame) -> pl.DataFrame:
    sql = """t.race_id, t.position_display_order AS display_order,
             t.position_number AS grid_position, t.position_text, t.driver_id, t.constructor_id,
             t.qualification_position_number AS qualification_position,
             t.grid_penalty, t.grid_penalty_positions"""
    return _session_table(
        db,
        [
            ("starting_grid_position", SessionType.RACE, "obs_grid"),
            ("sprint_starting_grid_position", SessionType.SPRINT, "obs_sprint_grid"),
        ],
        keys,
        sql,
        S.GRID,
    )


def build_practice(db: F1dbReader, keys: pl.DataFrame) -> pl.DataFrame:
    sql = """t.race_id, t.position_display_order AS display_order, t.position_number AS position,
             t.driver_id, t.constructor_id, t.time_millis AS best_ms, t.gap_millis AS gap_ms,
             t.laps"""
    return _session_table(
        db,
        [
            ("free_practice_1_result", SessionType.FP1, "obs_fp1"),
            ("free_practice_2_result", SessionType.FP2, "obs_fp2"),
            ("free_practice_3_result", SessionType.FP3, "obs_fp3"),
        ],
        keys,
        sql,
        S.PRACTICE,
    )


def build_pit_stops(db: F1dbReader, keys: pl.DataFrame, notes: list[str]) -> pl.DataFrame:
    df = _ids(
        db.query(
            "SELECT race_id, driver_id, constructor_id, stop, lap, time_millis AS duration_ms "
            "FROM pit_stop"
        )
    ).join(keys, on="race_id", how="inner")
    # The source occasionally lists an identical stop twice (e.g. 2026-R13). Exact duplicates
    # carry no information, so they are dropped — and reported, never silently.
    exact = ["event_id", "driver_id", "stop", "lap", "duration_ms"]
    dupes = df.filter(df.select(exact).is_duplicated())
    if dupes.height:
        by_event = dupes.group_by("event_id").len().sort("event_id")
        detail = ", ".join(f"{r['event_id']} x{r['len'] // 2}" for r in by_event.to_dicts())
        notes.append(f"pit_stops: dropped exact duplicate rows from source ({detail})")
        df = df.unique(subset=exact, keep="first", maintain_order=True)
    df = df.with_columns(pl.col("obs_race").alias("observed_at"))
    return S.conform(df, S.PIT_STOPS).sort("event_id", "driver_id", "stop")


def build_fastest_laps(db: F1dbReader, keys: pl.DataFrame) -> pl.DataFrame:
    df = _ids(
        db.query(
            "SELECT race_id, position_display_order AS display_order, driver_id, constructor_id, "
            "position_number AS position, lap, time_millis AS time_ms FROM fastest_lap"
        )
    ).join(keys, on="race_id", how="inner")
    df = df.with_columns(pl.col("obs_race").alias("observed_at"))
    return S.conform(df, S.FASTEST_LAPS).sort("event_id", "display_order")


def build_standings(db: F1dbReader, keys: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    drv = _ids(
        db.query(
            "SELECT race_id, position_display_order AS display_order, driver_id, "
            "position_number AS position, position_text, points FROM race_driver_standing"
        )
    ).join(keys, on="race_id", how="inner")
    con = _ids(
        db.query(
            "SELECT race_id, position_display_order AS display_order, constructor_id, "
            "engine_manufacturer_id, position_number AS position, position_text, points "
            "FROM race_constructor_standing"
        )
    ).join(keys, on="race_id", how="inner")
    drv = drv.with_columns(pl.col("obs_race").alias("observed_at"), pl.col("points").fill_null(0.0))
    con = con.with_columns(pl.col("obs_race").alias("observed_at"), pl.col("points").fill_null(0.0))
    return (
        S.conform(drv, S.DRIVER_STANDINGS).sort("event_id", "display_order"),
        S.conform(con, S.CONSTRUCTOR_STANDINGS).sort("event_id", "display_order"),
    )


# --------------------------------------------------------------------------- entities
def build_entities(db: F1dbReader) -> dict[str, pl.DataFrame]:
    circuits = _ids(
        db.query(
            "SELECT id AS circuit_id, name, full_name, type AS circuit_type, direction, "
            "place_name, country_id, latitude, longitude, length AS length_km, turns FROM circuit"
        )
    )
    layouts = _ids(
        db.query(
            "SELECT id AS layout_id, circuit_id, length AS length_km, turns FROM circuit_layout"
        )
    )
    drivers = _ids(
        db.query(
            "SELECT id AS driver_id, name, first_name, last_name, abbreviation, permanent_number, "
            "gender, date_of_birth, nationality_country_id FROM driver"
        ).with_columns(pl.col("date_of_birth").str.to_date("%Y-%m-%d"))
    )
    constructors = _ids(
        db.query("SELECT id AS constructor_id, name, full_name, country_id FROM constructor")
    )
    engines = _ids(
        db.query("SELECT id AS engine_manufacturer_id, name, country_id FROM engine_manufacturer")
    )
    return {
        "circuits": S.conform(circuits, S.CIRCUITS).sort("circuit_id"),
        "circuit_layouts": S.conform(layouts, S.CIRCUIT_LAYOUTS).sort("layout_id"),
        "drivers": S.conform(drivers, S.DRIVERS).sort("driver_id"),
        "constructors": S.conform(constructors, S.CONSTRUCTORS).sort("constructor_id"),
        "engine_manufacturers": S.conform(engines, S.ENGINE_MANUFACTURERS).sort(
            "engine_manufacturer_id"
        ),
    }


def build_entries(db: F1dbReader) -> pl.DataFrame:
    df = _ids(
        db.query(
            "SELECT year AS season, rounds, driver_id, constructor_id, engine_manufacturer_id, "
            "entrant_id, test_driver FROM season_entrant_driver"
        )
    ).with_columns(pl.col("test_driver").cast(pl.Boolean).fill_null(False))
    df = df.with_columns(
        pl.when(pl.col("rounds").is_null() | (pl.col("rounds") == ""))
        .then(pl.lit([None], dtype=pl.List(pl.String)))
        .otherwise(pl.col("rounds").str.split(";"))
        .alias("round")
    ).explode("round")
    df = df.with_columns(pl.col("round").cast(pl.Int32))
    return S.conform(df, S.ENTRIES).sort("season", "round", "constructor_id", "driver_id")


def build_constructor_seasons(db: F1dbReader) -> pl.DataFrame:
    """Lineage per (season, constructor) from F1DB's constructor chronology.

    Every member of a chain stores the full chain, so the lineage of a constructor in a season
    is the *latest* name of any chain containing it in that season; constructors in no chain
    are their own lineage.
    """
    chron = db.query(
        "SELECT constructor_id AS owner, position_display_order AS pos, "
        "other_constructor_id AS member, year_from, year_to FROM constructor_chronology"
    )
    finals = chron.sort("pos").group_by("owner").agg(pl.col("member").last().alias("final"))
    chron = chron.join(finals, on="owner")
    used = db.query(
        "SELECT DISTINCT r.year AS season, rr.constructor_id FROM race_result rr "
        "JOIN race r ON r.id = rr.race_id"
    )
    matched = (
        used.join(chron, left_on="constructor_id", right_on="member", how="left")
        .filter(
            pl.col("owner").is_null()
            | (
                (pl.col("season") >= pl.col("year_from"))
                & (pl.col("year_to").is_null() | (pl.col("season") <= pl.col("year_to")))
            )
        )
        .group_by("season", "constructor_id")
        .agg(pl.col("final").drop_nulls().sort().first().alias("lineage_id"))
    )
    df = used.join(matched, on=["season", "constructor_id"], how="left").with_columns(
        pl.coalesce("lineage_id", "constructor_id").alias("lineage_id")
    )
    return S.conform(_ids(df), S.CONSTRUCTOR_SEASONS).sort("season", "constructor_id")


def build_seasons(events: pl.DataFrame) -> pl.DataFrame:
    df = events.group_by("season").agg(pl.col("round").max().alias("rounds"))
    df = df.with_columns(
        pl.col("season")
        .map_elements(regulations.era_start, return_dtype=pl.Int32)
        .alias("era_start"),
        pl.col("season")
        .map_elements(regulations.is_reset_season, return_dtype=pl.Boolean)
        .alias("is_reset"),
    ).with_columns((pl.col("season") - pl.col("era_start")).alias("seasons_since_reset"))
    return S.conform(df, S.SEASONS).sort("season")


# --------------------------------------------------------------------------- entry point
@dataclass(frozen=True, slots=True)
class Curated:
    tables: dict[str, pl.DataFrame]
    notes: tuple[str, ...]


def curate(path: Path) -> Curated:
    """Build every curated table from one F1DB SQLite file, validated against its spec.

    ``notes`` records every correction applied to the source (e.g. dropped duplicates).
    """
    notes: list[str] = []
    with F1dbReader(path) as db:
        raw_events = _load_events(db)
        tim = _timing_frame(raw_events)
        events = build_events(raw_events, tim)
        keys = _race_keys(raw_events, tim)
        drv_st, con_st = build_standings(db, keys)
        tables: dict[str, pl.DataFrame] = {
            **build_entities(db),
            "constructor_seasons": build_constructor_seasons(db),
            "events": events,
            "sessions": build_sessions(raw_events),
            "entries": build_entries(db),
            "results": build_results(db, raw_events, tim, notes),
            "qualifying": build_qualifying(db, keys),
            "grid": build_grid(db, keys),
            "practice": build_practice(db, keys),
            "pit_stops": build_pit_stops(db, keys, notes),
            "fastest_laps": build_fastest_laps(db, keys),
            "driver_standings": drv_st,
            "constructor_standings": con_st,
        }
        tables["seasons"] = build_seasons(events)
    for name, df in tables.items():
        S.validate(df, S.SPECS[name])
    return Curated(tables=tables, notes=tuple(notes))
