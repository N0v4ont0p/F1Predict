from __future__ import annotations

import json
from pathlib import Path

from f1p import ledger


def _marts(root: Path, *, result: object = None) -> Path:
    (root / "events").mkdir(parents=True)
    (root / "index.json").write_text(json.dumps({"next_event": "2026-R18"}))
    cars = [
        {"driver_id": "a", "code": "AAA", "p_win": 0.6, "position_probs": [0.6, 0.4]},
        {"driver_id": "b", "code": "BBB", "p_win": 0.4, "position_probs": [0.4, 0.6]},
    ]
    event = {
        "event_id": "2026-R18",
        "season": 2026,
        "round": 18,
        "name": "Austin",
        "checkpoints": {"C1": {"model": "world_v1", "as_of": "2026-10-22T00:00:00", "cars": cars}},
        "quali": {
            "checkpoints": {"C1": {"model": "quali_v1", "as_of": "x", "cars": cars}},
            "result": None,
        },
        "result": result,
    }
    (root / "events" / "2026-R18.json").write_text(json.dumps(event))
    return root


def test_ledger_is_append_only_and_hashed(tmp_path: Path) -> None:
    marts = _marts(tmp_path / "marts")
    out = tmp_path / "ledger"
    first = ledger.record(marts, out)
    assert [(e.target, e.checkpoint, e.new) for e in first] == [
        ("race", "C1", True),
        ("quali", "C1", True),
    ]
    text = (out / "LEDGER.md").read_text()
    assert text.count("2026-R18") == 2
    assert first[0].sha256 in text
    # recording again changes nothing, even if the marts were republished differently
    ev = json.loads((marts / "events" / "2026-R18.json").read_text())
    ev["checkpoints"]["C1"]["cars"][0]["p_win"] = 0.9
    (marts / "events" / "2026-R18.json").write_text(json.dumps(ev))
    second = ledger.record(marts, out)
    assert not any(e.new for e in second)
    assert second[0].sha256 == first[0].sha256
    assert (out / "LEDGER.md").read_text() == text


def test_nothing_is_recorded_after_the_race(tmp_path: Path) -> None:
    marts = _marts(tmp_path / "marts", result=[{"position": 1}])
    assert ledger.record(marts, tmp_path / "ledger") == []
