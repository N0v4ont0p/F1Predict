"""F1DB — the all-time Formula 1 database (CC BY 4.0), our historical backbone (plan §6.1).

One SQLite download per release covers 1950 → the latest round, including practice,
qualifying, grids, sprints, pit stops, fastest laps and standings. That is far cheaper and
more complete than thousands of API calls.

Each download is kept under ``raw/f1db/<sha256-prefix>/`` with a manifest recording its
source, hash and coverage. A release is identified by content (not by a mutable "latest"
pointer), so re-running against the same data is a no-op.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from f1p.core.logging import get_logger
from f1p.sources.http import HttpClient

log = get_logger("sources.f1db")

ATTRIBUTION = "Data from F1DB (https://github.com/f1db/f1db), licensed CC BY 4.0."
LATEST_URL = "https://github.com/f1db/f1db/releases/latest/download/f1db-sqlite.zip"
TAGGED_URL = "https://github.com/f1db/f1db/releases/download/{tag}/f1db-sqlite.zip"
DB_NAME = "f1db.db"
MANIFEST = "manifest.json"


@dataclass(frozen=True, slots=True)
class F1dbRelease:
    """A materialised F1DB SQLite file and what it covers."""

    path: Path
    sha256: str
    source_url: str
    tag: str | None
    fetched_at: str
    latest_season: int
    latest_round: int

    @property
    def coverage(self) -> str:
        return f"{self.latest_season}-R{self.latest_round:02d}"

    @property
    def version(self) -> str:
        """Stable identifier: the tag if pinned, else coverage + content-hash prefix."""
        return self.tag or f"{self.coverage}+{self.sha256[:12]}"


def _coverage(db: Path) -> tuple[int, int]:
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
        row = con.execute(
            "SELECT r.year, r.round FROM race r "
            "WHERE EXISTS (SELECT 1 FROM race_result rr WHERE rr.race_id = r.id) "
            "ORDER BY r.year DESC, r.round DESC LIMIT 1"
        ).fetchone()
    if row is None:
        raise ValueError(f"{db} contains no race results")
    return int(row[0]), int(row[1])


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_release(directory: Path) -> F1dbRelease:
    meta = json.loads((directory / MANIFEST).read_text())
    meta["path"] = directory / DB_NAME
    return F1dbRelease(**meta)


def installed_releases(raw_dir: Path) -> list[F1dbRelease]:
    """All releases already on disk, oldest coverage first."""
    root = raw_dir / "f1db"
    if not root.exists():
        return []
    found = [load_release(d) for d in root.iterdir() if (d / MANIFEST).exists()]
    return sorted(found, key=lambda r: (r.latest_season, r.latest_round, r.fetched_at))


def fetch(http: HttpClient, raw_dir: Path, tag: str | None = None) -> F1dbRelease:
    """Download an F1DB release (latest by default) and return it.

    If an identical release (same SHA-256 of the SQLite file) is already installed, the new
    download is discarded and the existing one returned.
    """
    url = TAGGED_URL.format(tag=tag) if tag else LATEST_URL
    staging = _fresh_staging(raw_dir)
    archive = http.download(url, staging / "f1db-sqlite.zip")
    _extract(archive, staging)
    return _install(staging, raw_dir, source_url=url, tag=tag)


def install_local(path: Path, raw_dir: Path, tag: str | None = None) -> F1dbRelease:
    """Install a local ``f1db-sqlite.zip`` or ``f1db.db`` (offline use, tests)."""
    staging = _fresh_staging(raw_dir)
    if path.suffix == ".zip":
        _extract(path, staging)
    else:
        shutil.copyfile(path, staging / DB_NAME)
    return _install(staging, raw_dir, source_url=path.resolve().as_uri(), tag=tag)


def _fresh_staging(raw_dir: Path) -> Path:
    staging = raw_dir / "f1db" / "_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    return staging


def _extract(archive: Path, staging: Path) -> None:
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        if DB_NAME not in names:
            raise ValueError(f"unexpected F1DB archive contents: {names}")
        zf.extract(DB_NAME, staging)


def _install(staging: Path, raw_dir: Path, *, source_url: str, tag: str | None) -> F1dbRelease:
    db = staging / DB_NAME
    digest = _sha256(db)
    target = raw_dir / "f1db" / digest[:16]
    if (target / MANIFEST).exists():
        shutil.rmtree(staging)
        release = load_release(target)
        log.info("F1DB %s already installed", release.version)
        return release
    season, rnd = _coverage(db)
    target.mkdir(parents=True, exist_ok=True)
    shutil.move(str(db), target / DB_NAME)
    shutil.rmtree(staging)
    release = F1dbRelease(
        path=target / DB_NAME,
        sha256=digest,
        source_url=source_url,
        tag=tag,
        fetched_at=datetime.now(UTC).isoformat(),
        latest_season=season,
        latest_round=rnd,
    )
    meta = asdict(release)
    meta.pop("path")
    (target / MANIFEST).write_text(json.dumps(meta, indent=2))
    log.info("installed F1DB %s (covers %s)", release.version, release.coverage)
    return release
