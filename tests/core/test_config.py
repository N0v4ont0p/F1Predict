from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from f1p.core.config import MemoryBudget, Settings, get_settings


def test_defaults_target_a_24gb_m6() -> None:
    s = Settings()
    assert s.memory.total_gb == 24
    assert s.memory.working_gb + s.memory.reserved_gb <= s.memory.total_gb


def test_budget_must_fit() -> None:
    with pytest.raises(ValidationError, match="exceeds total"):
        MemoryBudget(total_gb=24, reserved_gb=10, working_gb=16)


def test_env_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("F1P_SEED", "7")
    monkeypatch.setenv("F1P_MEMORY__WORKING_GB", "12")
    monkeypatch.setenv("F1P_PATHS__ROOT", str(tmp_path))
    s = Settings()
    assert s.seed == 7
    assert s.memory.working_gb == 12
    assert s.paths.data == tmp_path / "data"
    assert s.paths.lake == tmp_path / "data" / "lake"


def test_absolute_paths_are_kept(tmp_path: Path) -> None:
    s = Settings.model_validate({"paths": {"data_dir": tmp_path}})
    assert s.paths.data == tmp_path


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    assert get_settings() is get_settings()
