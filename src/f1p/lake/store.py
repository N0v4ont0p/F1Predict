"""DuckLake-backed store for curated tables (plan §3 A10, §6.2).

Each commit of the curated tables is one DuckLake **snapshot**, so every backtest can pin the
exact data version it ran against (``snapshot=…``) — "data-version time" — while
``observed_at`` + :func:`f1p.core.clock.known_at` handle "knowability time".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import polars as pl

from f1p.core.logging import get_logger

log = get_logger("lake.store")

CATALOG = "lake"
INGEST_LOG = "ingest_log"


@dataclass(frozen=True, slots=True)
class Snapshot:
    snapshot_id: int
    committed_at: datetime


class Lake:
    """A DuckLake catalogue at ``root`` (metadata in ``catalog.ducklake``, data in ``files/``)."""

    def __init__(self, root: Path, *, memory_limit_gb: float = 3.0, threads: int | None = None):
        self.root = root
        self.memory_limit_gb = memory_limit_gb
        self.threads = threads
        self._con: duckdb.DuckDBPyConnection | None = None

    # ------------------------------------------------------------------ connection
    @property
    def metadata_path(self) -> Path:
        return self.root / "catalog.ducklake"

    @property
    def data_path(self) -> Path:
        return self.root / "files"

    @property
    def exists(self) -> bool:
        return self.metadata_path.exists()

    def connect(self) -> duckdb.DuckDBPyConnection:
        if self._con is None:
            self.root.mkdir(parents=True, exist_ok=True)
            self.data_path.mkdir(parents=True, exist_ok=True)
            con = duckdb.connect(config={"memory_limit": f"{self.memory_limit_gb}GB"})
            if self.threads:
                con.execute(f"SET threads = {int(self.threads)}")
            con.execute("INSTALL ducklake; LOAD ducklake;")
            con.execute(
                f"ATTACH 'ducklake:{self.metadata_path}' AS {CATALOG} "
                f"(DATA_PATH '{self.data_path}/')"
            )
            self._con = con
        return self._con

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None

    def __enter__(self) -> Lake:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ write
    def commit(self, tables: dict[str, pl.DataFrame], log_entry: dict[str, Any]) -> Snapshot:
        """Replace ``tables`` and append ``log_entry`` to the ingest log — atomically.

        Everything lands in a single DuckLake snapshot or not at all.
        """
        con = self.connect()
        con.execute("BEGIN TRANSACTION")
        try:
            for name, df in tables.items():
                view = f"_incoming_{name}"
                con.register(view, df.to_arrow())
                con.execute(f'CREATE OR REPLACE TABLE {CATALOG}."{name}" AS SELECT * FROM {view}')
                con.unregister(view)
            entry = pl.DataFrame(
                {
                    "committed_at": [datetime.now(UTC)],
                    "source": [str(log_entry.get("source", ""))],
                    "version": [str(log_entry.get("version", ""))],
                    "details": [json.dumps(log_entry, default=str, sort_keys=True)],
                },
                schema={
                    "committed_at": pl.Datetime("us", "UTC"),
                    "source": pl.String,
                    "version": pl.String,
                    "details": pl.String,
                },
            )
            con.register("_incoming_log", entry.to_arrow())
            con.execute(
                f"CREATE TABLE IF NOT EXISTS {CATALOG}.{INGEST_LOG} AS "
                "SELECT * FROM _incoming_log LIMIT 0"
            )
            con.execute(f"INSERT INTO {CATALOG}.{INGEST_LOG} SELECT * FROM _incoming_log")
            con.unregister("_incoming_log")
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        snap = self.latest_snapshot()
        assert snap is not None
        log.info("committed %d table(s) as snapshot %d", len(tables), snap.snapshot_id)
        return snap

    # ------------------------------------------------------------------ read
    def tables(self) -> list[str]:
        con = self.connect()
        rows = con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_catalog = ? "
            "ORDER BY table_name",
            [CATALOG],
        ).fetchall()
        return [r[0] for r in rows]

    def read(self, name: str, *, snapshot: int | None = None) -> pl.DataFrame:
        """Read a table at the latest snapshot or at ``snapshot`` (time travel).

        Curated tables come back exactly as declared in :mod:`f1p.lake.schema` (DuckDB labels
        UTC as ``Etc/UTC``; that is normalised here and the spec re-validated).
        """
        from f1p.lake.schema import SPECS, validate

        con = self.connect()
        at = f" AT (VERSION => {int(snapshot)})" if snapshot is not None else ""
        df = con.execute(f'SELECT * FROM {CATALOG}."{name}"{at}').pl()
        df = df.with_columns(
            pl.col(c).dt.convert_time_zone("UTC")
            for c, t in df.schema.items()
            if isinstance(t, pl.Datetime) and t.time_zone is not None
        )
        if name in SPECS:
            validate(df, SPECS[name])
        return df

    def read_all(
        self, names: list[str] | None = None, *, snapshot: int | None = None
    ) -> dict[str, pl.DataFrame]:
        return {n: self.read(n, snapshot=snapshot) for n in (names or self.tables())}

    def snapshots(self) -> pl.DataFrame:
        con = self.connect()
        return con.execute(
            f"SELECT snapshot_id, snapshot_time FROM ducklake_snapshots('{CATALOG}') ORDER BY 1"
        ).pl()

    def latest_snapshot(self) -> Snapshot | None:
        snaps = self.snapshots()
        if snaps.is_empty():
            return None
        row = snaps.row(-1, named=True)
        return Snapshot(int(row["snapshot_id"]), row["snapshot_time"])

    def ingest_log(self) -> pl.DataFrame:
        if INGEST_LOG not in self.tables():
            return pl.DataFrame()
        return self.read(INGEST_LOG).sort("committed_at")
