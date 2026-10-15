from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from f1p.cli.main import app
from f1p.core.config import Settings
from f1p.lake import pipeline


@pytest.fixture
def lake(isolated_settings: Settings, f1db_fixture: Path, tmp_path: Path) -> Settings:
    archive = tmp_path / "f1db-sqlite.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(f1db_fixture, "f1db.db")
    pipeline.update(isolated_settings, local_file=archive)
    return isolated_settings


def test_forecast_json_for_a_raced_event(lake: Settings) -> None:
    res = CliRunner().invoke(app, ["model", "forecast", "2026-R17", "--checkpoint", "C4", "--json"])
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["event_id"] == "2026-R17"
    rows = payload["drivers"]
    assert len(rows) >= 20
    assert sum(r["p_win"] for r in rows) == pytest.approx(1.0, abs=1e-6)
    assert sum(r["p_podium"] for r in rows) == pytest.approx(3.0, abs=1e-3)
    assert rows[0]["p_win"] >= rows[-1]["p_win"]
    assert all(0.0 < r["p_finish"] <= 1.0 for r in rows)


def test_forecast_next_event_uses_latest_lineup(lake: Settings) -> None:
    res = CliRunner().invoke(app, ["model", "forecast", "--json"])
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["event_id"] == "2026-R18"
    assert len(payload["drivers"]) == 22


def test_forecast_table_and_ratings(lake: Settings) -> None:
    runner = CliRunner()
    res = runner.invoke(app, ["model", "forecast", "2026-R17"])
    assert res.exit_code == 0, res.stdout
    assert "E[pos]" in res.stdout
    res = runner.invoke(app, ["model", "ratings", "--top", "5"])
    assert res.exit_code == 0, res.stdout
    assert "team beliefs" in res.stdout


def test_unknown_model(lake: Settings) -> None:
    res = CliRunner().invoke(app, ["model", "forecast", "2026-R17", "--model", "nope"])
    assert res.exit_code == 1
