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
    "entries",
    "constructor_seasons",
    "weather_forecast",
]


def available_models() -> dict[str, Any]:
    from f1p.eval.baselines import default_baselines

    models: dict[str, Any] = {m.name: m for m in default_baselines()}
    try:
        from f1p.models.registry import registered_models

        models.update(registered_models())
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
    checkpoints: Annotated[str, typer.Option(help="Checkpoints, e.g. C1,C4")] = "C1,C4",
    reference: Annotated[str, typer.Option(help="Model to compare against")] = "uniform",
    save: Annotated[bool, typer.Option(help="Write per-event metrics to artifacts/eval/")] = True,
) -> None:
    """Score models on every event in [--from, --to] with only then-public data."""
    from f1p.eval import harness
    from f1p.lake.pipeline import lake_for

    catalogue = available_models()
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
        checkpoints=tuple(c.strip() for c in checkpoints.split(",")),
    )
    console.print(_summary_table(result, reference))
    if save and result.per_event.height:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out = settings.paths.artifacts / "eval" / stamp
        out.mkdir(parents=True, exist_ok=True)
        result.per_event.write_parquet(out / "per_event.parquet")
        meta = {
            "seasons": [start, end],
            "models": names,
            "reference": reference,
            "lake_snapshot": snap.snapshot_id if snap else None,
            "summary": result.summary().to_dicts(),
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
