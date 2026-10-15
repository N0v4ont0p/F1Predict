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


def test_championship_as_of_a_round(lake: Settings) -> None:
    runner = CliRunner()
    res = runner.invoke(
        app, ["model", "championship", "--after-round", "12", "--sims", "500", "--json"]
    )
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["after_round"] == 12
    assert len(payload["remaining"]) == 11
    drivers = payload["drivers"]
    assert [d["most_likely"] for d in drivers] == list(range(1, len(drivers) + 1))
    assert sum(d["p_title"] for d in drivers) == pytest.approx(1.0, abs=1e-6)
    res = runner.invoke(
        app,
        [
            "model",
            "championship",
            "--as-of",
            "2026-06-01",
            "--sims",
            "300",
            "--kind",
            "constructor",
        ],
    )
    assert res.exit_code == 0, res.stdout
    assert "constructors" in res.stdout
    res = runner.invoke(app, ["model", "championship", "--after-round", "30"])
    assert res.exit_code == 1
    res = runner.invoke(
        app, ["model", "championship", "--after-round", "3", "--as-of", "2026-01-01"]
    )
    assert res.exit_code == 1


def test_forecast_qualifying(lake: Settings) -> None:
    runner = CliRunner()
    res = runner.invoke(
        app,
        ["model", "forecast", "2026-R17", "--target", "quali", "--checkpoint", "C2", "--json"],
    )
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["target"] == "quali"
    assert payload["model"] == "quali_v1"
    rows = payload["drivers"]
    assert sum(r["p_pole"] for r in rows) == pytest.approx(1.0, abs=1e-6)
    assert sum(r["p_q3"] for r in rows) == pytest.approx(10.0, abs=1e-3)
    assert sum(r["p_out_q1"] for r in rows) == pytest.approx(len(rows) - 16, abs=1e-3)
    res = runner.invoke(app, ["model", "forecast", "--target", "quali"])
    assert res.exit_code == 0, res.stdout
    assert "front row" in res.stdout
    # qualifying is not forecast after qualifying
    res = runner.invoke(app, ["model", "forecast", "--target", "quali", "--checkpoint", "C4"])
    assert res.exit_code == 1
