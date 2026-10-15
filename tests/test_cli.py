from __future__ import annotations

from typer.testing import CliRunner

from f1p import __version__
from f1p.cli.main import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_doctor() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "memory budget" in result.stdout


def test_points_for_a_season() -> None:
    result = runner.invoke(app, ["points", "2021"])
    assert result.exit_code == 0
    assert "25" in result.stdout
    assert "fastest lap" in result.stdout


def test_points_for_a_shortened_race() -> None:
    result = runner.invoke(app, ["points", "2023", "--distance", "0.3"])
    assert result.exit_code == 0
    assert "13" in result.stdout


def test_points_unsupported_season_fails_cleanly() -> None:
    result = runner.invoke(app, ["points", "1980"])
    assert result.exit_code == 1
    assert "1991" in result.stdout
