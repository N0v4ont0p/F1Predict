"""DuckLake store + pipeline. Needs the DuckLake extension (downloaded once by DuckDB)."""

from __future__ import annotations

from datetime import UTC, datetime
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


def test_rebuilding_one_season_keeps_the_others(isolated_settings: Settings) -> None:
    import polars as pl

    from f1p.lake import pipeline
    from f1p.lake.schema import RACE_PACE, conform

    def pace(events: list[str]) -> pl.DataFrame:
        return conform(
            pl.DataFrame(
                {
                    "event_id": events,
                    "driver_id": ["x"] * len(events),
                    "car_number": ["1"] * len(events),
                    "clean_laps": [10] * len(events),
                    "median_lap_ms": [90_000] * len(events),
                    "pace_deficit_pct": [0.0] * len(events),
                    "source": ["livetiming"] * len(events),
                    "observed_at": [datetime(2025, 1, 1, tzinfo=UTC)] * len(events),
                }
            ),
            RACE_PACE,
        )

    events = pl.DataFrame(
        {"event_id": ["2025-R01", "2026-R01", "2026-R02"], "season": [2025, 2026, 2026]}
    )
    with pipeline.lake_for(isolated_settings) as lake:
        lake.commit({"race_pace": pace(["2025-R01", "2026-R01"])}, {"source": "test"})
        merged = pipeline._merge_seasons(
            lake, "race_pace", pace(["2026-R01", "2026-R02"]), events, (2026, 2026)
        )
    assert sorted(merged["event_id"].to_list()) == ["2025-R01", "2026-R01", "2026-R02"]
