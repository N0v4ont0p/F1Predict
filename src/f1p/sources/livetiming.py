"""F1 live-timing static archive (plan §6.1): the timing feeds behind every session, 2018 →.

The archive is served as ``https://livetiming.formula1.com/static/<path><Feed>.jsonStream``:
one line per update, ``HH:MM:SS.mmm`` session time followed by a JSON patch. Streams are
large (≈7 MB of TimingData per race) and immutable once a session ends, so they are stored
gzipped under ``raw/livetiming/`` instead of in the HTTP cache.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from f1p.sources.http import HttpClient

BASE = "https://livetiming.formula1.com/static/"
FIRST_SEASON = 2018


@dataclass(frozen=True, slots=True)
class SessionRef:
    season: int
    meeting: str
    name: str  # "Race", "Sprint", "Qualifying", …
    kind: str  # "Race", "Qualifying", "Practice"
    path: str  # e.g. "2024/2024-03-02_Bahrain_Grand_Prix/2024-03-02_Race/"

    @property
    def day(self) -> date:
        return date.fromisoformat(self.path.rstrip("/").split("/")[-1][:10])


def _decode(body: bytes) -> str:
    return body.decode("utf-8-sig")


def sessions(http: HttpClient, season: int) -> list[SessionRef]:
    """Every session the archive lists for ``season``."""
    raw = http.get(f"{BASE}{season}/Index.json").body
    idx = json.loads(_decode(raw))
    out = []
    for m in idx.get("Meetings", []):
        for s in m.get("Sessions", []):
            if s.get("Path"):
                out.append(
                    SessionRef(
                        season, m.get("Name", ""), s.get("Name", ""), s.get("Type", ""), s["Path"]
                    )
                )
    return out


def stream_path(raw_dir: Path, ref: SessionRef, feed: str) -> Path:
    return raw_dir / "livetiming" / ref.path / f"{feed}.jsonStream.gz"


def fetch_stream(http: HttpClient, raw_dir: Path, ref: SessionRef, feed: str) -> Path:
    """Download one feed once and keep it gzipped (immutable archive)."""
    dest = stream_path(raw_dir, ref, feed)
    if dest.exists():
        return dest
    resp = http.get(f"{BASE}{ref.path}{feed}.jsonStream", use_cache=False, store=False)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    with gzip.open(tmp, "wb", compresslevel=6) as fh:
        fh.write(resp.body)
    tmp.replace(dest)
    return dest


def read_stream(path: Path) -> Iterator[tuple[float, dict[str, Any]]]:
    """``(session_seconds, patch)`` for every line of a stored stream."""
    with gzip.open(path, "rt", encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if len(line) < 13 or line[2] != ":":
                continue
            h, m, s = line[:12].split(":")
            try:
                patch = json.loads(line[12:])
            except json.JSONDecodeError:
                continue
            if isinstance(patch, dict):
                yield int(h) * 3600 + int(m) * 60 + float(s), patch
