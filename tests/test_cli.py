"""CLI smoke tests via Typer's CliRunner."""
from __future__ import annotations

import json

from typer.testing import CliRunner

from f1predict.cli import app

runner = CliRunner()


def _fake_model(store, name, family="ensemble"):
    store.mkdir(parents=True, exist_ok=True)
    (store / f"{name}.joblib").write_bytes(b"x" * 10)
    (store / f"{name}.card.json").write_text(
        json.dumps({"family": family, "preset": "test", "metrics": {}}))


def _cfg_file(tmp_path, store):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"paths:\n  models_store: {store}\n")
    return cfg


def test_help():
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    assert "f1predict" in res.stdout


def test_version():
    res = runner.invoke(app, ["version"])
    assert res.exit_code == 0


def test_subcommand_help():
    for cmd in ["train", "predict", "backtest", "simulate", "whatif",
                "regulations", "explain", "weather", "shell", "data", "model"]:
        res = runner.invoke(app, [cmd, "--help"])
        assert res.exit_code == 0, f"{cmd} help failed: {res.stdout}"


def test_no_args_shows_help_when_not_a_tty():
    # CliRunner stdin/stdout are not TTYs, so the root callback must print help
    # instead of dropping into the interactive shell (which would hang).
    res = runner.invoke(app, [])
    assert res.exit_code == 0
    assert "Usage" in res.stdout or "f1predict" in res.stdout


def test_train_without_preset_refuses_on_non_tty(tmp_path):
    # Non-TTY (pipe/CI/tests) must NOT silently fall back to the default preset and
    # kick off a multi-hour train — it must refuse and tell the user to pass one.
    master = tmp_path / "master.parquet"
    import pandas as pd
    pd.DataFrame({"season": [2023]}).to_parquet(master)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"paths:\n  master_dataset: {master}\n")
    res = runner.invoke(app, ["train", "--config", str(cfg)])
    assert res.exit_code == 1
    assert "preset" in res.stdout.lower()


def test_model_delete_help():
    res = runner.invoke(app, ["model", "delete", "--help"])
    assert res.exit_code == 0
    assert "delete" in res.stdout.lower()


def test_model_delete_confirms_and_removes(tmp_path):
    store = tmp_path / "models_store"
    _fake_model(store, "ensemble_default")
    _fake_model(store, "rf_smoke", family="random_forest")
    (store / "production.txt").write_text("ensemble_default.joblib")
    cfg = _cfg_file(tmp_path, store)

    # decline first → nothing removed
    res = runner.invoke(app, ["model", "delete", "rf_smoke", "--config", str(cfg)], input="n\n")
    assert res.exit_code == 0
    assert (store / "rf_smoke.joblib").exists()

    # confirm → removed
    res = runner.invoke(app, ["model", "delete", "rf_smoke", "--config", str(cfg)], input="y\n")
    assert res.exit_code == 0
    assert not (store / "rf_smoke.joblib").exists()
    assert not (store / "rf_smoke.card.json").exists()


def test_model_delete_yes_skips_prompt_and_promotes(tmp_path):
    store = tmp_path / "models_store"
    _fake_model(store, "ensemble_default")
    _fake_model(store, "rf_smoke", family="random_forest")
    (store / "production.txt").write_text("ensemble_default.joblib")
    cfg = _cfg_file(tmp_path, store)

    res = runner.invoke(
        app, ["model", "delete", "ensemble_default", "-y", "--config", str(cfg)])
    assert res.exit_code == 0
    assert not (store / "ensemble_default.joblib").exists()
    # active model auto-promoted to the survivor
    assert (store / "production.txt").read_text().strip() == "rf_smoke.joblib"


def test_model_delete_unknown_errors(tmp_path):
    store = tmp_path / "models_store"
    _fake_model(store, "ensemble_default")
    cfg = _cfg_file(tmp_path, store)
    res = runner.invoke(app, ["model", "delete", "lightgbm", "-y", "--config", str(cfg)])
    assert res.exit_code == 1


def test_model_delete_clear_wipes_everything(tmp_path):
    store = tmp_path / "models_store"
    _fake_model(store, "ensemble_default")
    _fake_model(store, "rf_smoke", family="random_forest")
    _fake_model(store, "extra_trees_x", family="extra_trees")
    (store / "production.txt").write_text("ensemble_default.joblib")
    cfg = _cfg_file(tmp_path, store)

    res = runner.invoke(app, ["model", "delete", "clear", "-y", "--config", str(cfg)])
    assert res.exit_code == 0
    assert "EVERY trained model" in res.stdout
    assert list(store.glob("*.joblib")) == []
    assert list(store.glob("*.card.json")) == []
    assert not (store / "production.txt").exists()


def test_model_delete_all_can_be_aborted(tmp_path):
    store = tmp_path / "models_store"
    _fake_model(store, "a_model")
    _fake_model(store, "b_model")
    (store / "production.txt").write_text("a_model.joblib")
    cfg = _cfg_file(tmp_path, store)

    res = runner.invoke(app, ["model", "delete", "all", "--config", str(cfg)], input="n\n")
    assert res.exit_code == 0
    assert len(list(store.glob("*.joblib"))) == 2  # nothing deleted


def test_model_delete_clear_empty_store(tmp_path):
    store = tmp_path / "models_store"
    store.mkdir(parents=True, exist_ok=True)
    cfg = _cfg_file(tmp_path, store)
    res = runner.invoke(app, ["model", "delete", "clear", "-y", "--config", str(cfg)])
    assert res.exit_code == 0
    assert "No trained models" in res.stdout
