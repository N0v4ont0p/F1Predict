from __future__ import annotations

import json
from pathlib import Path

import pytest

from f1p import publish
from f1p.lake.curate_f1db import Curated


@pytest.fixture(scope="module")
def marts(curated: Curated, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("marts")
    publish.publish(
        curated.tables,
        out,
        season=2026,
        snapshot=1,
        leaderboard_from=None,
        championship_history=False,
    )
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
    assert {"code", "team", "lineage", "p_finish", "why"} <= set(cars[0])
    assert set(cars[0]["why"]) == {"rating", "grid", "quali", "lap_pace"}
    # the parts are centred on the field: they sum to ~0 across cars
    assert abs(sum(c["why"]["rating"] for c in cars)) < 0.05
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


def test_championship_is_the_full_projected_table(marts: Path) -> None:
    c = _load(marts / "championship.json")
    assert c["after_round"] == 17
    assert len(c["remaining"]) == 6
    drivers = c["drivers"]
    assert [d["most_likely"] for d in drivers] == list(range(1, len(drivers) + 1))
    assert sum(d["p_title"] for d in drivers) == pytest.approx(1.0, abs=1e-3)
    for d in drivers:
        assert {"name", "team", "lineage", "quantiles", "p_top3", "p_position"} <= set(d)
        assert len(d["p_position"]) == len(drivers)
        assert d["p_top3"] == pytest.approx(sum(d["p_position"][:3]), abs=2e-3)
    assert {"team", "lineage"} <= set(c["constructors"][0])


def test_championship_history_has_a_table_per_round(curated: Curated) -> None:
    h = publish.build_championship_history(curated.tables, 2026, n_sims=400)
    assert h is not None
    snaps = h["snapshots"]
    assert [s["after_round"] for s in snaps] == list(range(18))
    assert snaps[0]["remaining"] == 23
    assert snaps[-1]["remaining"] == 6
    assert all(x["current"] == 0 for x in snaps[0]["drivers"])
    for s in snaps:
        assert sum(x["p_title"] for x in s["drivers"]) == pytest.approx(1.0, abs=1e-3)
        assert {x["driver_id"] for x in s["drivers"]} <= set(h["drivers"])
    assert len(h["rounds"]) == 23
    # points only ever accumulate from one round's snapshot to the next
    first = {x["driver_id"]: x["current"] for x in snaps[5]["drivers"]}
    later = {x["driver_id"]: x["current"] for x in snaps[6]["drivers"]}
    assert all(later.get(d, 0) >= v for d, v in first.items())


def test_events_carry_a_qualifying_block(marts: Path) -> None:
    raced = _load(marts / "events" / "2026-R17.json")["quali"]
    assert set(raced["checkpoints"]) == {"C1", "C2"}
    cars = raced["checkpoints"]["C2"]["cars"]
    assert sum(c["p_win"] for c in cars) == pytest.approx(1.0, abs=1e-3)  # P(pole)
    assert sum(c["p_q3"] for c in cars) == pytest.approx(10.0, abs=1e-2)
    assert {"p_front_row", "p_out_q2", "p_out_q1"} <= set(cars[0])
    assert raced["result"][0]["position"] == 1
    assert {"quali_v1", "last_quali", "practice"} <= set(raced["scores"]["C2"])
    nxt = _load(marts / "events" / "2026-R18.json")["quali"]
    assert list(nxt["checkpoints"]) == ["C1"]  # no practice yet, so no C2
    assert nxt["result"] is None
