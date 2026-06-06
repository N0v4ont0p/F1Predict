"""Lightweight experiment tracker (JSONL + manifest).

Deliberately avoids heavy deps like MLflow. Each ``train`` run appends one JSON line with
params, metrics, git commit, data hash, peak memory and training time. The dashboard and
``model leaderboard`` CLI command read this back.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from ..config import Config, load_config


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "nogit"
    except Exception:
        return "nogit"


def log_experiment(record: dict[str, Any], cfg: Config | None = None) -> Path:
    """Append an experiment record and return the experiments file path."""
    cfg = cfg or load_config()
    path = cfg.path("paths.experiments")
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": _git_commit(),
        **record,
    }
    with open(path, "a") as fh:
        fh.write(json.dumps(record) + "\n")
    return path


def load_experiments(cfg: Config | None = None) -> list[dict]:
    cfg = cfg or load_config()
    path = cfg.path("paths.experiments")
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out
