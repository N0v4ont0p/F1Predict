from __future__ import annotations

import gzip
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from f1p.core.config import Settings, get_settings
from f1p.lake.curate_f1db import Curated, curate

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def f1db_fixture(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A real F1DB subset (2021 + 2026 seasons plus edge-case events), decompressed."""
    dest = tmp_path_factory.mktemp("f1db") / "f1db.db"
    with gzip.open(FIXTURES / "f1db_subset.db.gz", "rb") as fin, dest.open("wb") as fout:
        shutil.copyfileobj(fin, fout)
    return dest


@pytest.fixture(scope="session")
def curated(f1db_fixture: Path) -> Curated:
    return curate(f1db_fixture)


@pytest.fixture
def isolated_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Settings]:
    """Settings rooted in a temporary directory (and the cached instance reset)."""
    monkeypatch.setenv("F1P_PATHS__ROOT", str(tmp_path))
    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()
