"""Configuration loading with YAML + dotted-key overrides and ``defaults`` merging."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

# Project root = two levels up from this file (src/config/config.py -> project root).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "base.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base`` (override wins)."""
    out = copy.deepcopy(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


class Config:
    """Thin wrapper over a nested dict with dotted-key access and path resolution."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, dotted: str, value: Any) -> None:
        node = self._data
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def path(self, dotted: str) -> Path:
        """Return a config value as a resolved absolute Path."""
        return _resolve(self.get(dotted))

    @property
    def data(self) -> dict[str, Any]:
        return self._data

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)


def load_config(
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Config:
    """Load a config file, recursively applying any ``defaults:`` parent config.

    ``overrides`` is a mapping of dotted keys -> values applied last, e.g.
    ``{"model.family": "lightgbm"}``.
    """
    path = _resolve(config_path) if config_path else DEFAULT_CONFIG
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}

    parent_ref = raw.pop("defaults", None)
    if parent_ref:
        parent = load_config(parent_ref).as_dict()
        raw = _deep_merge(parent, raw)

    cfg = Config(raw)
    for dotted, value in (overrides or {}).items():
        cfg.set(dotted, value)
    return cfg
