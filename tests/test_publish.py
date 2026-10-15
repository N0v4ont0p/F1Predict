from __future__ import annotations

import json
from pathlib import Path

import pytest

from f1p import publish
from f1p.lake.curate_f1db import Curated


@pytest.fixture(scope="module")
def marts(curated: Curated, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("marts")
    publish.publish(curated.tables, out, season=2026, snapshot=1, leaderboard_from=None)
    return out


def _load(path: Path) -> dict:  # type: ignore[type-arg]
    return json.loads(path.read_text())


def test_index_lists_the_whole_calendar(marts: Path) -> None:
    idx = _load(marts / "index.json")
    assert idx["schema_version"] == publish.SCHEMA_VERSION
    assert idx["lake_snapshot"] == 1
    statuses = [e["status"] for e in idx["events"]]
    assert statuses.count("raced") == 17
    assert statuses.count("next") == 1
    assert idx["next_event"] == "2026-R18"
    raced = idx["events"][16]
    assert raced["winner"]["driver_id"] == "max_verstappen"
    assert 0 < raced["p_winner_C4"] <= 1
    assert raced["name"] == "Singapore"


def test_raced_event_has_forecasts_result_and_scores(marts: Path) -> None:
    ev = _load(marts / "events" / "2026-R17.json")
    assert set(ev["checkpoints"]) == {"C1", "C4"}
    cars = ev["checkpoints"]["C4"]["cars"]
    assert sum(c["p_win"] for c in cars) == pytest.approx(1.0, abs=1e-3)
    assert sum(c["p_podium"] for c in cars) == pytest.approx(3.0, abs=1e-2)
    assert all(len(c["position_probs"]) == len(cars) for c in cars)
    assert cars[0]["p_win"] >= cars[-1]["p_win"]
    assert {"code", "team", "lineage", "p_finish"} <= set(cars[0])
    assert ev["result"][0]["position"] == 1
    assert {"world_v1", "quali"} <= set(ev["scores"]["C4"])


def test_next_event_only_has_checkpoints_whose_information_exists(marts: Path) -> None:
    ev = _load(marts / "events" / "2026-R18.json")
    assert ev["status"] == "next"
    assert list(ev["checkpoints"]) == ["C1"]  # no qualifying yet, so no C4
    assert len(ev["checkpoints"]["C1"]["cars"]) == 22
    assert ev["result"] is None


def test_ratings(marts: Path) -> None:
    r = _load(marts / "ratings.json")
    assert r["teams"][0]["mu"] >= r["teams"][-1]["mu"]
    assert r["drivers"]
    assert any(d["series"] for d in r["drivers"])


def test_short_name() -> None:
    assert publish.short_name("united_states", "x") == "United States"
    assert publish.short_name("sao_paulo", "x") == "São Paulo"
    assert publish.short_name(None, "Fallback") == "Fallback"
