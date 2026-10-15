"""The forecast ledger: an append-only, hashed record of every forecast published before the
session it forecasts (plan, cross-cutting).

For the next event in the published marts, each forecast (race at C1/C4, qualifying at C1/C2)
that exists and has not been recorded yet is written once to
``docs/ledger/<season>/<event>-<target>-<checkpoint>.json`` in a canonical form (sorted keys,
fixed rounding), and a line with its SHA-256 is appended to ``docs/ledger/LEDGER.md``. Entries
are never rewritten, so once committed the forward record cannot be edited after the fact
without the hash (and git history) showing it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HEADER = (
    "# Forecast ledger\n\n"
    "Forecasts recorded before the session they forecast. Each file is canonical JSON; its "
    "SHA-256 is listed here. Entries are append-only.\n\n"
    "| Recorded (UTC) | Event | Target | Checkpoint | Data as of | Favourite | P | SHA-256 |\n"
    "|---|---|---|---|---|---|---|---|\n"
)


@dataclass(frozen=True, slots=True)
class Entry:
    event_id: str
    target: str
    checkpoint: str
    path: Path
    sha256: str
    new: bool


def canonical(event: dict[str, Any], target: str, cp: str, fc: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_id": event["event_id"],
        "name": event.get("name"),
        "season": event.get("season"),
        "round": event.get("round"),
        "target": target,
        "checkpoint": cp,
        "as_of": fc.get("as_of"),
        "model": fc.get("model"),
        "cars": [
            {
                "driver_id": c["driver_id"],
                "p_win": round(float(c["p_win"]), 4),
                "position_probs": [round(float(p), 4) for p in c.get("position_probs", [])],
            }
            for c in sorted(fc["cars"], key=lambda c: c["driver_id"])
        ],
    }


def digest(payload: dict[str, Any]) -> tuple[str, bytes]:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(body).hexdigest(), body


def record(marts: Path, ledger: Path) -> list[Entry]:
    """Record the next event's forecasts from ``marts`` into ``ledger``; returns every entry
    considered (``new`` says whether it was written now)."""
    index = json.loads((marts / "index.json").read_text())
    event_id = index.get("next_event")
    if not event_id:
        return []
    event = json.loads((marts / "events" / f"{event_id}.json").read_text())
    blocks = [("race", event.get("checkpoints", {}))]
    if event.get("quali"):
        if event["quali"].get("result"):
            blocks.append(("quali", {}))  # qualifying already happened: nothing to record
        else:
            blocks.append(("quali", event["quali"].get("checkpoints", {})))
    if event.get("result"):
        return []
    out: list[Entry] = []
    log = ledger / "LEDGER.md"
    ledger.mkdir(parents=True, exist_ok=True)
    if not log.exists():
        log.write_text(HEADER)
    for target, forecasts in blocks:
        for cp, fc in sorted(forecasts.items()):
            payload = canonical(event, target, cp, fc)
            sha, body = digest(payload)
            path = ledger / str(event.get("season")) / f"{event_id}-{target}-{cp}.json"
            if path.exists():
                out.append(
                    Entry(
                        event_id, target, cp, path, digest(json.loads(path.read_text()))[0], False
                    )
                )
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body + b"\n")
            fav = max(fc["cars"], key=lambda c: c["p_win"])
            with log.open("a") as fh:
                fh.write(
                    f"| {datetime.now(UTC):%Y-%m-%d %H:%M} | {event_id} | {target} | {cp} "
                    f"| {payload['as_of']} | {fav.get('code', fav['driver_id'])} "
                    f"| {float(fav['p_win']):.3f} | `{sha}` |\n"
                )
            out.append(Entry(event_id, target, cp, path, sha, True))
    return out
