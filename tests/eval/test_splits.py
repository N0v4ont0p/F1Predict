from __future__ import annotations

from pathlib import Path

import pytest

from f1p.eval import splits


def test_splits_partition_the_modern_era() -> None:
    assert splits.split_of(2013) is None
    assert [splits.split_of(s) for s in (2014, 2021, 2022, 2023, 2024, 2026)] == [
        "tune",
        "tune",
        "select",
        "select",
        "holdout",
        "holdout",
    ]
    seasons = [s for lo, hi in splits.SPLITS.values() for s in range(lo, hi + 1)]
    assert seasons == list(range(2014, 2027))


def test_holdout_guard() -> None:
    assert not splits.touches_holdout((2014, 2023))
    assert splits.touches_holdout((2023, 2024))
    assert splits.touches_holdout((2026, 2030))
    assert splits.search_seasons((2014, 2023)) == (2014, 2023)
    with pytest.raises(ValueError, match="holdout"):
        splits.search_seasons((2020, 2024))


def test_ledger_records_once_per_item(tmp_path: Path) -> None:
    ledger = splits.HoldoutLedger(tmp_path / "ledger" / "holdout.jsonl")
    assert ledger.entries() == []
    ledger.record("m", "sig1", target="race", seasons=(2024, 2026), kind="adhoc")
    assert ledger.check([("m", "sig1")], "race") == []  # a dev run is not the scoring
    e = ledger.record(
        "m", "sig1", target="race", seasons=(2024, 2026), kind="holdout", summary={"x": 1.0}
    )
    assert ledger.scored("m", "sig1", "race") == e
    assert ledger.check([("m", "sig1"), ("m", "sig2")], "race") == ["m"]
    assert ledger.check([("m", "sig1")], "quali") == []
    assert [x.kind for x in ledger.entries()] == ["adhoc", "holdout"]
    assert ledger.entries()[1].seasons == (2024, 2026)
