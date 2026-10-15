"""fetch → curate → gate → commit (plan §6, §10.7).

The pipeline is idempotent: if the newest F1DB release is already the one the lake was built
from, nothing is rewritten. Gates run before the commit, so a failing source never reaches
the lake (``force=True`` commits anyway and records the failures in the ingest log).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from f1p.core.config import Settings
from f1p.core.logging import get_logger
from f1p.lake.curate_f1db import curate
from f1p.lake.quality import GateResult, QualityError, run_gates
from f1p.lake.store import Lake, Snapshot
from f1p.sources import f1db
from f1p.sources.http import HttpClient

log = get_logger("lake.pipeline")


@dataclass(frozen=True, slots=True)
class UpdateReport:
    version: str
    coverage: str
    changed: bool
    snapshot: Snapshot | None
    gates: list[GateResult] = field(default_factory=list)
    notes: tuple[str, ...] = ()
    row_counts: dict[str, int] = field(default_factory=dict)


def raw_dir(settings: Settings) -> Path:
    return settings.paths.data / "raw"


def lake_for(settings: Settings, *, read_only: bool = False) -> Lake:
    return Lake(
        settings.paths.lake,
        memory_limit_gb=min(3.0, settings.memory.working_gb),
        read_only=read_only,
    )


def _built_from(lake: Lake) -> str | None:
    if not lake.exists:
        return None
    log_df = lake.ingest_log()
    if log_df.is_empty():
        return None
    rows = log_df.filter(pl.col("source") == "f1db")
    return rows["version"][-1] if rows.height else None


def update(
    settings: Settings,
    *,
    tag: str | None = None,
    local_file: Path | None = None,
    force: bool = False,
    http: HttpClient | None = None,
) -> UpdateReport:
    """Bring the lake up to date with F1DB: the latest release, a pinned ``tag``, or a local
    ``f1db-sqlite.zip`` / ``f1db.db`` (``local_file``, for offline use)."""
    raw = raw_dir(settings)
    if local_file is not None:
        release = f1db.install_local(local_file, raw, tag=tag)
    else:
        own = http is None
        client = http or HttpClient(cache_dir=raw / "http")
        try:
            release = f1db.fetch(client, raw, tag=tag)
        finally:
            if own:
                client.close()

    with lake_for(settings) as lake:
        if not force and _built_from(lake) == release.version:
            log.info("lake already built from F1DB %s — nothing to do", release.version)
            return UpdateReport(release.version, release.coverage, False, lake.latest_snapshot())

        curated = curate(release.path)
        gates = run_gates(curated.tables)
        failed = [g for g in gates if not g.passed]
        if failed and not force:
            raise QualityError(gates)
        snap = lake.commit(
            curated.tables,
            {
                "source": "f1db",
                "version": release.version,
                "coverage": release.coverage,
                "sha256": release.sha256,
                "source_url": release.source_url,
                "notes": list(curated.notes),
                "gates": {g.name: g.summary for g in gates},
                "forced": bool(failed),
                "attribution": f1db.ATTRIBUTION,
            },
        )
        return UpdateReport(
            version=release.version,
            coverage=release.coverage,
            changed=True,
            snapshot=snap,
            gates=gates,
            notes=curated.notes,
            row_counts={k: v.height for k, v in curated.tables.items()},
        )


def update_weather(
    settings: Settings,
    *,
    seasons: tuple[int, int],
    http: HttpClient | None = None,
) -> dict[str, int]:
    """Fetch race weather (Open-Meteo) for ``seasons`` and commit it as one snapshot.

    Fetching can take minutes, so it runs against a read-only view of the lake; the write
    lock is only taken for the commit.
    """
    from f1p.lake import weather
    from f1p.sources import openmeteo

    with lake_for(settings, read_only=True) as lake:
        if not lake.exists:
            raise FileNotFoundError("the lake is empty — run `f1p data update` first")
        tables = lake.read_all(["events", "circuits"])
    own = http is None
    client = http or HttpClient(cache_dir=raw_dir(settings) / "http")
    try:
        frames = weather.build(client, tables, seasons=seasons)
    finally:
        if own:
            client.close()
    with lake_for(settings) as lake:
        lake.commit(
            frames,
            {
                "source": "open-meteo",
                "version": f"{seasons[0]}-{seasons[1]}",
                "rows": {k: v.height for k, v in frames.items()},
                "attribution": openmeteo.ATTRIBUTION,
            },
        )
    return {k: v.height for k, v in frames.items()}


def _merge_seasons(
    lake: Lake, name: str, new: pl.DataFrame, events: pl.DataFrame, seasons: tuple[int, int]
) -> pl.DataFrame:
    """``new`` plus the stored rows of ``name`` for events outside ``seasons`` — so rebuilding
    one season never drops the others."""
    if name not in lake.tables():
        return new
    try:
        old = lake.read(name)
    except Exception:  # an older schema: rebuild from scratch
        log.warning("%s has an older schema; replacing it", name)
        return new
    if old.is_empty() or set(old.columns) != set(new.columns):
        return new
    in_range = events.filter(pl.col("season").is_between(*seasons))["event_id"]
    keep = old.filter(~pl.col("event_id").is_in(in_range.implode()))
    return pl.concat([keep.select(new.columns), new], how="vertical_relaxed").sort(
        "event_id", *[c for c in ("driver_id", "lap") if c in new.columns]
    )


def update_race_pace(
    settings: Settings,
    *,
    seasons: tuple[int, int],
    http: HttpClient | None = None,
) -> int:
    """Build ``race_pace`` from the live-timing archive and commit it as one snapshot."""
    from f1p.lake import race_pace

    with lake_for(settings, read_only=True) as lake:
        if not lake.exists:
            raise FileNotFoundError("the lake is empty — run `f1p data update` first")
        tables = lake.read_all(["events", "results"])
    own = http is None
    client = http or HttpClient(cache_dir=raw_dir(settings) / "http")
    try:
        df = race_pace.build(client, raw_dir(settings), tables, seasons=seasons)
    finally:
        if own:
            client.close()
    with lake_for(settings) as lake:
        df = _merge_seasons(lake, "race_pace", df, tables["events"], seasons)
        lake.commit(
            {"race_pace": df},
            {
                "source": "livetiming+jolpica",
                "version": f"{seasons[0]}-{seasons[1]}",
                "rows": df.height,
                "events": df["event_id"].n_unique(),
            },
        )
    return df.height


def update_laps(
    settings: Settings,
    *,
    seasons: tuple[int, int],
    http: HttpClient | None = None,
) -> int:
    """Build ``laps`` (every race lap with tyres and traffic) and commit it as one snapshot."""
    from f1p.lake import laps

    with lake_for(settings, read_only=True) as lake:
        if not lake.exists:
            raise FileNotFoundError("the lake is empty — run `f1p data update` first")
        tables = lake.read_all(["events", "results"])
    own = http is None
    client = http or HttpClient(cache_dir=raw_dir(settings) / "http")
    try:
        df = laps.build(client, raw_dir(settings), tables, seasons=seasons)
    finally:
        if own:
            client.close()
    with lake_for(settings) as lake:
        df = _merge_seasons(lake, "laps", df, tables["events"], seasons)
        lake.commit(
            {"laps": df},
            {
                "source": "livetiming+jolpica",
                "version": f"{seasons[0]}-{seasons[1]}",
                "rows": df.height,
                "events": df["event_id"].n_unique(),
            },
        )
    return df.height


def update_circuit_geometry(
    settings: Settings, *, season: int, http: HttpClient | None = None
) -> int:
    """Build ``circuit_geometry`` for ``season``'s calendar and commit it as one snapshot."""
    from f1p.lake import circuit_geometry

    with lake_for(settings, read_only=True) as lake:
        if not lake.exists:
            raise FileNotFoundError("the lake is empty — run `f1p data update` first")
        tables = lake.read_all(["events", "results"])
    own = http is None
    client = http or HttpClient(cache_dir=raw_dir(settings) / "http")
    try:
        df = circuit_geometry.build(client, tables, season=season)
    finally:
        if own:
            client.close()
    with lake_for(settings) as lake:
        lake.commit(
            {"circuit_geometry": df},
            {
                "source": "openf1",
                "version": str(season),
                "rows": df.height,
                "circuits": df["circuit_id"].n_unique(),
            },
        )
    return df["circuit_id"].n_unique()


def check(settings: Settings) -> list[GateResult]:
    """Re-run every gate against the lake's current snapshot."""
    from f1p.lake.schema import SPECS

    with lake_for(settings, read_only=True) as lake:
        if not lake.exists:
            raise FileNotFoundError("the lake is empty — run `f1p data update` first")
        tables = lake.read_all([n for n in SPECS if n in lake.tables()])
    return run_gates(tables)
