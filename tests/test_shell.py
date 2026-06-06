"""Tests for the interactive shell: command tree, context-aware completion, dispatch."""
from __future__ import annotations

import json

from prompt_toolkit.document import Document

from f1predict.config import load_config
from f1predict.cli.shell import _command_tree, _build_completer, _dispatch


def _completions(text, cfg=None):
    comp = _build_completer(cfg or load_config())
    doc = Document(text, len(text))
    return [c.text for c in comp.get_completions(doc, None)]


def _cfg_with_models(tmp_path):
    store = tmp_path / "models_store"
    store.mkdir(parents=True, exist_ok=True)
    for name in ("ensemble_default", "rf_smoke"):
        (store / f"{name}.joblib").write_bytes(b"x")
        (store / f"{name}.card.json").write_text(
            json.dumps({"family": "ensemble", "preset": "test", "metrics": {}}))
    (store / "production.txt").write_text("ensemble_default.joblib")
    cfg = load_config()
    cfg.set("paths.models_store", str(store))
    return cfg


def test_command_tree_has_core_commands():
    tree = _command_tree()
    for name in ("train", "predict", "simulate", "model", "data", "weather"):
        assert name in tree
    # hidden helpers are excluded
    assert "_select" not in tree.get("model", {}).get("subs", {})
    # groups expose sub-commands
    assert "delete" in tree["model"]["subs"]
    assert "build" in tree["data"]["subs"]


def test_complete_top_level_commands_and_meta():
    out = _completions("")
    assert "predict" in out and "train" in out
    assert "/help" in out and "/exit" in out


def test_complete_command_prefix():
    out = _completions("pre")
    assert "predict" in out
    assert all(c.startswith("pre") for c in out)


def test_complete_race_reference():
    out = _completions("predict su")
    assert any(c.startswith("su") for c in out)
    assert "suzuka" in out


def test_complete_options_and_preset_values():
    assert "--preset" in _completions("train --pre")
    presets = _completions("train --preset ")
    assert "test" in presets and "max" in presets


def test_complete_model_subcommands_and_names(tmp_path):
    cfg = _cfg_with_models(tmp_path)
    out = _completions("model ", cfg)
    assert "delete" in out                      # real sub-command
    assert "ensemble_default" in out            # model name (for `model <name>`)


def test_dispatch_runs_without_exiting():
    # version + a bad flag must both return control to the caller (no SystemExit escaping)
    _dispatch(["version"])
    _dispatch(["predict", "--nope"])            # UsageError handled internally
    _dispatch(["definitely-not-a-command"])     # unknown command handled internally
