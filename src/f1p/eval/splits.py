"""The fixed three-way time split (plan M6 §0) and the holdout ledger.

* **tune** (2014-2021): every search, every fitted setting.
* **select** (2022-2023): choosing between candidates and architectures.
* **holdout** (2024-2026): scored **once per item**, at the end; the result is recorded even
  when it is bad.

The ledger is an append-only JSON-lines file. Each holdout scoring adds one entry per model,
keyed by the model's name and full-settings signature (``runner.model_signature``), so a
second scoring of the same item is refused unless it is explicitly marked as a rescore (and
the rescore is recorded too).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

Split = Literal["tune", "select", "holdout"]

TUNE: tuple[int, int] = (2014, 2021)
SELECT: tuple[int, int] = (2022, 2023)
HOLDOUT: tuple[int, int] = (2024, 2026)
SPLITS: dict[Split, tuple[int, int]] = {"tune": TUNE, "select": SELECT, "holdout": HOLDOUT}


def split_of(season: int) -> Split | None:
    """The split a season belongs to (``None`` before 2014)."""
    for name, (lo, hi) in SPLITS.items():
        if lo <= season <= hi:
            return name
    return None


def touches_holdout(seasons: tuple[int, int]) -> bool:
    lo, hi = seasons
    return lo <= HOLDOUT[1] and hi >= HOLDOUT[0]


def search_seasons(seasons: tuple[int, int]) -> tuple[int, int]:
    """Guard for searches and model selection: the range must lie inside tune + select."""
    if touches_holdout(seasons):
        raise ValueError(
            f"seasons {seasons} overlap the holdout {HOLDOUT}; searches use tune/select only"
        )
    return seasons


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    item: str  # model name
    signature: str  # what makes the model's forecasts (prepare signature + settings)
    target: str
    seasons: tuple[int, int]
    when: str
    kind: str  # "holdout" (the one scoring), "rescore" or "adhoc" (a dev run that saw it)
    summary: dict[str, Any]


class HoldoutLedger:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def entries(self) -> list[LedgerEntry]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                d["seasons"] = tuple(d["seasons"])
                out.append(LedgerEntry(**d))
        return out

    def scored(self, item: str, signature: str, target: str) -> LedgerEntry | None:
        """The item's holdout scoring, if it has had one."""
        for e in self.entries():
            if (e.item, e.signature, e.target, e.kind) == (item, signature, target, "holdout"):
                return e
        return None

    def check(self, items: Iterable[tuple[str, str]], target: str) -> list[str]:
        """Names of the (item, signature) pairs already scored on the holdout."""
        return [i for i, s in items if self.scored(i, s, target) is not None]

    def record(
        self,
        item: str,
        signature: str,
        *,
        target: str,
        seasons: tuple[int, int],
        kind: str,
        summary: dict[str, Any] | None = None,
    ) -> LedgerEntry:
        entry = LedgerEntry(
            item,
            signature,
            target,
            seasons,
            datetime.now(UTC).isoformat(timespec="seconds"),
            kind,
            summary or {},
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(json.dumps(asdict(entry), default=str) + "\n")
        return entry
