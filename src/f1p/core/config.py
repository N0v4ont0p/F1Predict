"""Typed settings, overridable with ``F1P_*`` environment variables.

Nested fields use ``__``: ``F1P_MEMORY__WORKING_GB=12`` or ``F1P_PATHS__DATA_DIR=/Volumes/f1``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Self

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Paths(BaseModel):
    root: Path = Field(default_factory=Path.cwd)
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")

    def resolve(self, path: Path) -> Path:
        return path if path.is_absolute() else self.root / path

    @property
    def data(self) -> Path:
        return self.resolve(self.data_dir)

    @property
    def lake(self) -> Path:
        return self.data / "lake"

    @property
    def artifacts(self) -> Path:
        return self.resolve(self.artifacts_dir)


class MemoryBudget(BaseModel):
    """Unified-memory budget (plan §11a.5). Defaults target a 24 GB Apple M6."""

    total_gb: float = Field(default=24.0, gt=0)
    reserved_gb: float = Field(default=7.0, ge=0, description="never allocated: macOS + apps")
    working_gb: float = Field(default=16.0, gt=0, description="pipeline ceiling")

    @model_validator(mode="after")
    def _fits(self) -> Self:
        if self.reserved_gb + self.working_gb > self.total_gb + 1e-9:
            raise ValueError(
                f"reserved ({self.reserved_gb} GB) + working ({self.working_gb} GB) "
                f"exceeds total ({self.total_gb} GB)"
            )
        return self


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="F1P_", env_nested_delimiter="__", env_file=".env", extra="ignore"
    )

    seed: int = 42
    paths: Paths = Field(default_factory=Paths)
    memory: MemoryBudget = Field(default_factory=MemoryBudget)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings (cached). Call ``get_settings.cache_clear()`` after env changes."""
    return Settings()
