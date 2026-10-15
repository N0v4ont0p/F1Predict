"""DuckLake store + pipeline. Needs the DuckLake extension (downloaded once by DuckDB)."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from f1p.cli.main import app
from f1p.core.config import Settings
from f1p.lake import pipeline
from f1p.lake.curate_f1db import Curated
from f1p.lake.schema import CURATED
from f1p.lake.store import Lake


def test_commit_read_and_time_travel(tmp_path: Path, curated: Curated) -> None:
    with Lake(tmp_path / "lake") as lake:
        first = lake.commit(
            {"events": curated.tables["events"]}, {"source": "test", "version": "a"}
        )
        events = lake.read("events")
        assert events.equals(curated.tables["events"])
        trimmed = curated.tables["events"].filter(pl.col("season") == 2026)
        second = lake.commit({"events": trimmed}, {"source": "test", "version": "b"})
        assert second.snapshot_id > first.snapshot_id
        assert lake.read("events").height == trimmed.height
        assert lake.read("events", snapshot=first.snapshot_id).height == events.height
        log = lake.ingest_log()
        assert log["version"].to_list() == ["a", "b"]


def test_failed_commit_rolls_back(tmp_path: Path, curated: Curated) -> None:
    with Lake(tmp_path / "lake") as lake:
        lake.commit({"events": curated.tables["events"]}, {"source": "test", "version": "a"})
        before = lake.latest_snapshot()

        class Boom:
            def to_arrow(self) -> object:
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            lake.commit({"events": curated.tables["events"], "bad": Boom()}, {"source": "x"})  # type: ignore[dict-item]
        assert lake.latest_snapshot() == before
        assert lake.read("events").height == curated.tables["events"].height


def test_read_only_readers_share_the_catalogue(tmp_path: Path, curated: Curated) -> None:
    root = tmp_path / "lake"
    with Lake(root) as lake:
        lake.commit({"events": curated.tables["events"]}, {"source": "test", "version": "a"})
    with Lake(root, read_only=True) as a, Lake(root, read_only=True) as b:
        assert a.read("events").height == b.read("events").height
        with pytest.raises(PermissionError):
            a.commit({"events": curated.tables["events"]}, {"source": "test"})


@pytest.fixture
def fixture_zip(tmp_path: Path, f1db_fixture: Path) -> Path:
    import zipfile

    archive = tmp_path / "f1db-sqlite.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(f1db_fixture, "f1db.db")
    return archive


def test_pipeline_update_is_gated_and_idempotent(
    isolated_settings: Settings, fixture_zip: Path
) -> None:
    report = pipeline.update(isolated_settings, local_file=fixture_zip)
    assert report.changed
    assert all(g.passed for g in report.gates)
    assert report.snapshot is not None
    assert report.row_counts["results"] > 1000
    again = pipeline.update(isolated_settings, local_file=fixture_zip)
    assert not again.changed
    assert again.version == report.version
    gates = pipeline.check(isolated_settings)
    assert all(g.passed for g in gates)
    with pipeline.lake_for(isolated_settings) as lake:
        assert {s.name for s in CURATED} <= set(lake.tables())


def test_data_cli(isolated_settings: Settings, fixture_zip: Path) -> None:
    runner = CliRunner()
    res = runner.invoke(app, ["data", "update", "--file", str(fixture_zip)])
    assert res.exit_code == 0, res.stdout
    assert "Committed snapshot" in res.stdout
    res = runner.invoke(app, ["data", "status"])
    assert res.exit_code == 0, res.stdout
    assert "results" in res.stdout
    res = runner.invoke(app, ["data", "check"])
    assert res.exit_code == 0, res.stdout
    res = runner.invoke(
        app, ["data", "show", "results", "--event", "2026-R17", "--session", "race", "--limit", "3"]
    )
    assert res.exit_code == 0, res.stdout
    assert "max_verstappen" in res.stdout
    res = runner.invoke(app, ["data", "show", "nope"])
    assert res.exit_code == 1


def test_status_without_lake(isolated_settings: Settings) -> None:
    res = CliRunner().invoke(app, ["data", "status"])
    assert res.exit_code == 1
    assert "f1p data update" in res.stdout
