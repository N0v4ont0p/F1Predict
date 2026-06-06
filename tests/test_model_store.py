"""Tests for the trained-model store: listing, fuzzy resolution and switching."""
from __future__ import annotations

import json

import pytest

from f1predict.models import (
    list_models, resolve_model, set_production, production_file, delete_model,
)


def _make_model(store, name, family="ensemble", preset="test", metrics=None):
    store.mkdir(parents=True, exist_ok=True)
    (store / f"{name}.joblib").write_bytes(b"x" * 10)
    card = {"family": family, "preset": preset, "metrics": metrics or {}}
    (store / f"{name}.card.json").write_text(json.dumps(card))


def test_list_models_and_production(cfg):
    store = cfg.path("paths.models_store")
    _make_model(store, "ensemble_default", metrics={"winner_accuracy": 0.5, "mae": 3.6})
    _make_model(store, "random_forest_smoke", family="random_forest")
    (store / "production.txt").write_text("ensemble_default.joblib")

    models = list_models(cfg)
    assert {m["name"] for m in models} == {"ensemble_default", "random_forest_smoke"}
    active = [m for m in models if m["is_production"]]
    assert len(active) == 1 and active[0]["name"] == "ensemble_default"


def test_resolve_exact_prefix_and_substring(cfg):
    store = cfg.path("paths.models_store")
    _make_model(store, "ensemble_default")
    _make_model(store, "random_forest_smoke", family="random_forest")

    assert resolve_model("ensemble_default", cfg)["name"] == "ensemble_default"
    assert resolve_model("ensemble", cfg)["name"] == "ensemble_default"      # unique prefix
    assert resolve_model("smoke", cfg)["name"] == "random_forest_smoke"      # unique substring
    assert resolve_model("RANDOM_FOREST_SMOKE", cfg)["name"] == "random_forest_smoke"


def test_resolve_ambiguous_and_missing(cfg):
    store = cfg.path("paths.models_store")
    _make_model(store, "random_forest_smoke", family="random_forest")
    _make_model(store, "random_forest_default", family="random_forest")

    with pytest.raises(ValueError, match="ambiguous"):
        resolve_model("random", cfg)
    with pytest.raises(ValueError, match="No model matches"):
        resolve_model("lightgbm", cfg)


def test_set_production_switches_pointer(cfg):
    store = cfg.path("paths.models_store")
    _make_model(store, "ensemble_default")
    _make_model(store, "ensemble_overhaul_test")
    (store / "production.txt").write_text("ensemble_default.joblib")

    rec = set_production("overhaul", cfg)
    assert rec["name"] == "ensemble_overhaul_test"
    assert production_file(cfg) == "ensemble_overhaul_test.joblib"


def test_delete_model_removes_artifacts(cfg):
    store = cfg.path("paths.models_store")
    _make_model(store, "ensemble_default")
    _make_model(store, "random_forest_smoke", family="random_forest")
    (store / "production.txt").write_text("ensemble_default.joblib")

    res = delete_model("smoke", cfg)
    assert res["record"]["name"] == "random_forest_smoke"
    assert not res["was_production"]
    assert not (store / "random_forest_smoke.joblib").exists()
    assert not (store / "random_forest_smoke.card.json").exists()
    assert {m["name"] for m in list_models(cfg)} == {"ensemble_default"}
    # untouched production pointer
    assert production_file(cfg) == "ensemble_default.joblib"


def test_delete_production_promotes_newest(cfg):
    import os, time
    store = cfg.path("paths.models_store")
    _make_model(store, "old_model")
    time.sleep(0.01)
    _make_model(store, "new_model")
    # make new_model strictly newer so list_models (newest-first) ranks it #1
    now = time.time()
    os.utime(store / "old_model.joblib", (now - 100, now - 100))
    os.utime(store / "new_model.joblib", (now, now))
    (store / "production.txt").write_text("old_model.joblib")

    res = delete_model("old_model", cfg)
    assert res["was_production"]
    assert res["new_production"] == "new_model"
    assert production_file(cfg) == "new_model.joblib"


def test_delete_last_model_clears_pointer(cfg):
    store = cfg.path("paths.models_store")
    _make_model(store, "only_model")
    (store / "production.txt").write_text("only_model.joblib")

    res = delete_model("only", cfg)
    assert res["was_production"]
    assert res["new_production"] is None
    assert production_file(cfg) is None
    assert list_models(cfg) == []


def test_delete_missing_raises(cfg):
    store = cfg.path("paths.models_store")
    _make_model(store, "ensemble_default")
    with pytest.raises(ValueError, match="No model matches"):
        delete_model("lightgbm", cfg)
