"""Shared pytest fixtures: a small, fast synthetic dataset and a tmp-configured Config."""
from __future__ import annotations

import pytest

from f1predict.config import load_config
from f1predict.data_pipeline import synthetic


@pytest.fixture(scope="session")
def small_df():
    return synthetic.generate([2018, 2019, 2020, 2021], rounds_per_season=8, seed=1)


@pytest.fixture()
def cfg(tmp_path):
    c = load_config()
    c.set("paths.master_dataset", str(tmp_path / "master.parquet"))
    c.set("paths.models_store", str(tmp_path / "models_store"))
    c.set("paths.experiments", str(tmp_path / "experiments.jsonl"))
    c.set("model.random_forest.n_estimators", 60)
    return c
