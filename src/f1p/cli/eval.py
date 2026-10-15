"""``f1p eval`` — rolling-origin evaluation of models against baselines."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Annotated, Any

import polars as pl
import typer
from rich.console import Console
from rich.table import Table

from f1p.core.config import get_settings
from f1p.eval.harness import METRICS, EvalResult, better

app = typer.Typer(help="Evaluate models with rolling-origin backtests.", no_args_is_help=True)
console = Console()

EVAL_TABLES = [
    "events",
    "results",
    "grid",
    "qualifying",
    "driver_standings",
    "constructor_standings",
    "entries",
    "constructor_seasons",
    "weather_forecast",
    "race_pace",
    "practice",
    "sessions",
]


def available_models(target: str = "race") -> dict[str, Any]:
    """Baselines for ``target`` plus every registered model that forecasts it."""
    from f1p.eval.baselines import default_baselines
    from f1p.eval.harness import model_targets

    models: dict[str, Any] = {m.name: m for m in default_baselines(target)}
    try:
        from f1p.models.registry import registered_models

        models.update({k: m for k, m in registered_models().items() if target in model_targets(m)})
    except ImportError:  # pragma: no cover - models package not present
        pass
    return models


def _summary_table(result: EvalResult, reference: str) -> Table:
    summary = result.summary()
    table = Table(title=f"rolling-origin evaluation (Δ vs {reference}, 95% CI)")
    for col in ("model", "cp", "events", *METRICS, "ece_top3"):
        table.add_column(col, justify="right" if col not in ("model", "cp") else "left")
    for row in summary.iter_rows(named=True):
        cells = [row["model"], row["checkpoint"], str(row["events"])]
        for metric in METRICS:
            value = row[metric]
            text = "-" if value is None else f"{value:.3f}"
            if row["model"] != reference and value is not None:
                cmp = result.compare(row["model"], reference, row["checkpoint"], metric)
                if cmp.n and cmp.significant:
                    colour = "green" if better(metric, cmp.mean_diff) else "red"
                    text = f"[{colour}]{text}[/{colour}]"
            cells.append(text)
        ece = row.get("ece_top3")
        cells.append("-" if ece is None else f"{ece:.3f}")
        table.add_row(*cells)
    return table


@app.command()
def run(
    models: Annotated[str, typer.Option(help="Comma-separated model names, or 'all'")] = "all",
    start: Annotated[int, typer.Option("--from", help="First evaluated season")] = 2014,
    end: Annotated[int, typer.Option("--to", help="Last evaluated season")] = 2026,
    history: Annotated[
        int | None, typer.Option(help="First season of history (default: from - 8)")
    ] = None,
    checkpoints: Annotated[
        str | None, typer.Option(help="Checkpoints, e.g. C1,C4 (default: the target's)")
    ] = None,
    target: Annotated[str, typer.Option(help="race or quali")] = "race",
    reference: Annotated[str, typer.Option(help="Model to compare against")] = "uniform",
    save: Annotated[bool, typer.Option(help="Write per-event metrics to artifacts/eval/")] = True,
) -> None:
    """Score models on every event in [--from, --to] with only then-public data."""
    from f1p.eval import harness
    from f1p.lake.pipeline import lake_for

    if target not in ("race", "quali"):
        console.print(f"[red]unknown target {target!r}[/red]; choose race or quali")
        raise typer.Exit(code=1)
    catalogue = available_models(target)
    names = list(catalogue) if models == "all" else [m.strip() for m in models.split(",")]
    unknown = [n for n in names if n not in catalogue]
    if unknown:
        console.print(f"[red]unknown model(s) {unknown}[/red]; available: {', '.join(catalogue)}")
        raise typer.Exit(code=1)
    if reference not in names:
        names.append(reference)
    settings = get_settings()
    with lake_for(settings, read_only=True) as lake:
        if not lake.exists:
            console.print("[red]No lake — run `f1p data update` first.[/red]")
            raise typer.Exit(code=1)
        present = set(lake.tables())
        tables = lake.read_all([n for n in EVAL_TABLES if n in present])
        snap = lake.latest_snapshot()
    result = harness.run(
        tables,
        [catalogue[n] for n in names],
        seasons=(start, end),
        history_from=history,
        checkpoints=tuple(c.strip() for c in checkpoints.split(",")) if checkpoints else None,
        target=target,
    )
    console.print(_summary_table(result, reference))
    if save and result.per_event.height:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out = settings.paths.artifacts / "eval" / stamp
        out.mkdir(parents=True, exist_ok=True)
        result.per_event.write_parquet(out / "per_event.parquet")
        meta = {
            "seasons": [start, end],
            "target": target,
            "models": names,
            "reference": reference,
            "lake_snapshot": snap.snapshot_id if snap else None,
            "summary": result.summary().to_dicts(),
        }
        (out / "summary.json").write_text(json.dumps(meta, indent=2, default=str))
        console.print(f"saved → {out}")


@app.command()
def title(
    start: Annotated[int, typer.Option("--from", help="First season")] = 2010,
    end: Annotated[int, typer.Option("--to", help="Last season")] = 2025,
    sims: Annotated[int, typer.Option(help="Simulated seasons per snapshot")] = 5_000,
    save: Annotated[bool, typer.Option(help="Write snapshots to artifacts/eval/")] = True,
) -> None:
    """Backtest championship projections: a snapshot after every round of every season."""
    from f1p.lake.pipeline import lake_for
    from f1p.models import title_backtest as tb

    settings = get_settings()
    with lake_for(settings, read_only=True) as lake:
        if not lake.exists:
            console.print("[red]No lake — run `f1p data update` first.[/red]")
            raise typer.Exit(code=1)
        present = set(lake.tables())
        tables = lake.read_all([n for n in EVAL_TABLES if n in present])
        snap = lake.latest_snapshot()
    df = tb.run(
        tables,
        (start, end),
        n_sims=sims,
        progress=lambda s, n: console.print(f"  {s}: {n} rounds", style="dim"),
    )
    if df.is_empty():
        console.print("[red]no completed seasons in range[/red]")
        raise typer.Exit(code=1)
    table = Table(title=f"title projections {start}-{end} (Δ vs uniform spread, season bootstrap)")
    for col in ("kind", "metric", "world_v1", "uniform", "Δ", "95% CI"):
        table.add_column(col, justify="left" if col in ("kind", "metric") else "right")
    for r in tb.summarise(df):
        c = r.comparison
        colour = "green" if c.significant and c.mean_diff < 0 else "red" if c.significant else ""
        delta = f"{c.mean_diff:+.3f}"
        table.add_row(
            r.kind,
            r.metric,
            f"{r.model:.3f}",
            f"{r.baseline:.3f}",
            f"[{colour}]{delta}[/{colour}]" if colour else delta,
            f"[{c.lo:+.3f}, {c.hi:+.3f}]",
        )
    console.print(table)
    calls = tb.called_from(df)
    t2 = Table(title="when the drivers' title was called (P(champion) ≥ 50% for good)")
    for col in ("season", "rounds", "called after round", "P pre-season"):
        t2.add_column(col, justify="right")
    for r in calls.iter_rows(named=True):
        call = "never" if r["called_after"] is None else str(r["called_after"])
        t2.add_row(str(r["season"]), str(r["rounds"]), call, f"{100 * r['p_preseason']:.0f}%")
    console.print(t2)
    if save:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out = settings.paths.artifacts / "eval" / f"title-{stamp}"
        out.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out / "snapshots.parquet")
        meta = {
            "seasons": [start, end],
            "n_sims": sims,
            "lake_snapshot": snap.snapshot_id if snap else None,
            "summary": [
                {
                    "kind": r.kind,
                    "metric": r.metric,
                    "world_v1": r.model,
                    "uniform": r.baseline,
                    "mean_diff": r.comparison.mean_diff,
                    "lo": r.comparison.lo,
                    "hi": r.comparison.hi,
                }
                for r in tb.summarise(df)
            ],
            "called": calls.to_dicts(),
        }
        (out / "summary.json").write_text(json.dumps(meta, indent=2, default=str))
        console.print(f"saved → {out}")


@app.command()
def models() -> None:
    """List the models the evaluator knows about."""
    for name in available_models():
        console.print(name)


def load_run(path: str) -> pl.DataFrame:
    return pl.read_parquet(f"{path}/per_event.parquet")
