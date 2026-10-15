# Legacy snapshot (v0)

The pre-rebuild implementation is preserved in git history.

| Item | Value |
|---|---|
| Last legacy commit | `95d8892` on `main` (the parent of the rebuild) |
| Recommended tag | `v0-legacy`. Create it with `git tag -a v0-legacy 95d8892 -m "Legacy f1predict" && git push origin v0-legacy` (the rebuild session could not push tags) |
| Restore | `git checkout 95d8892` or `git worktree add ../f1predict-legacy 95d8892` |

## What it was

- Python package `f1predict` (Typer CLI + Streamlit dashboard), ~7.4k lines.
- 5,215 race-result rows (2014 → 2026 Round 5) from Jolpica, cached as Parquet in git.
- About 90 hand-built features; tree ensemble (RF / ExtraTrees / LightGBM / HistGB / GBR) with a
  Ridge stacker, regressing finishing position.
- Probabilities from Gaussian noise sized by in-sample residual spread; a rank-shuffle Monte Carlo.

## Recorded metrics (`experiments/experiments.jsonl`, `deep` preset)

| Model | Races | MAE | Spearman | Winner | Podium |
|---|---|---|---|---|---|
| Race | 29 | 3.70 | 0.659 | 55.2% | 66.7% |
| Qualifying | 29 | 3.37 | 0.700 | 27.6% | 63.2% |
| Sprint | 9 | 3.96 | 0.599 | 55.6% | 70.4% |

These were measured under defects D1–D4 (plan §1.2). Phase 2 re-scores the frozen legacy model
in the new evaluation harness for a fair comparison.

## Dependencies at the snapshot

numpy, pandas, polars, pyarrow, scikit-learn, lightgbm, optuna, typer, rich, prompt_toolkit,
plotly, streamlit, requests, pyyaml, matplotlib (Python ≥ 3.10).
