"""``f1p model`` — forecast an event and inspect the world model's beliefs."""

from __future__ import annotations

import json
from typing import Annotated

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from f1p.cli.eval import EVAL_TABLES
from f1p.core.config import get_settings

app = typer.Typer(help="Forecast races and inspect model beliefs.", no_args_is_help=True)
console = Console()


def _tables() -> dict:  # type: ignore[type-arg]
    from f1p.lake.pipeline import lake_for

    with lake_for(get_settings(), read_only=True) as lake:
        if not lake.exists:
            console.print("[red]No lake — run `f1p data update` first.[/red]")
            raise typer.Exit(code=1)
        present = set(lake.tables())
        return lake.read_all([n for n in EVAL_TABLES if n in present])


def _pct(p: float) -> str:
    return "<0.1%" if p < 0.001 else f"{100 * p:.1f}%"


@app.command()
def forecast(
    event: Annotated[
        str | None, typer.Argument(help="Event id, e.g. 2026-R18 (default: next race)")
    ] = None,
    checkpoint: Annotated[str, typer.Option(help="C1 (pre-weekend) or C4 (pre-race)")] = "C1",
    model: Annotated[str, typer.Option(help="Model name")] = "world_v1",
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON instead of a table")] = False,
) -> None:
    """Forecast a race: win, podium, points and finish probabilities for every car."""
    from f1p.cli.eval import available_models
    from f1p.eval.context import next_event
    from f1p.eval.harness import forecast_event

    tables = _tables()
    event_id = event or next_event(tables)
    if event_id is None:
        console.print("[red]No upcoming event in the lake.[/red]")
        raise typer.Exit(code=1)
    catalogue = available_models()
    if model not in catalogue:
        console.print(f"[red]unknown model {model!r}[/red]; available: {', '.join(catalogue)}")
        raise typer.Exit(code=1)
    fc, ctx = forecast_event(tables, catalogue[model], event_id, checkpoint)
    if fc is None:
        console.print(f"[red]{model} cannot forecast {event_id} at {checkpoint}.[/red]")
        raise typer.Exit(code=1)
    p_finish = fc.p_finish if fc.p_finish is not None else np.ones(fc.n)
    order = np.argsort(-fc.p_win)
    rows = [
        {
            "driver": fc.drivers[i],
            "constructor": ctx.constructors[i],
            "p_win": float(fc.p_win[i]),
            "p_podium": float(fc.p_top(3)[i]),
            "p_points": float(fc.p_top(10)[i]),
            "p_finish": float(p_finish[i]),
            "expected_position": float(fc.expected_position[i]),
        }
        for i in order
    ]
    if as_json:
        payload = {
            "event_id": event_id,
            "checkpoint": checkpoint,
            "as_of": ctx.as_of.isoformat(),
            "model": model,
            "meta": fc.meta,
            "drivers": rows,
        }
        typer.echo(json.dumps(payload, indent=2))
        return
    ev = tables["events"].filter(tables["events"]["event_id"] == event_id).row(0, named=True)
    table = Table(
        title=f"{ev['name']} ({event_id}) · {checkpoint} · {model}",
        caption=f"data public at {ctx.as_of:%Y-%m-%d %H:%M} UTC",
    )
    for col in ("#", "driver", "team", "win", "podium", "points", "finish", r"E\[pos]"):
        table.add_column(col, justify="left" if col in ("driver", "team") else "right")
    for k, r in enumerate(rows, 1):
        table.add_row(
            str(k),
            r["driver"],
            r["constructor"],
            _pct(r["p_win"]),
            _pct(r["p_podium"]),
            _pct(r["p_points"]),
            _pct(r["p_finish"]),
            f"{r['expected_position']:.1f}",
        )
    console.print(table)


@app.command()
def ratings(
    top: Annotated[int, typer.Option(help="Rows per kind")] = 15,
) -> None:
    """Current driver-skill and team-performance beliefs (mean +- sd, log-strength)."""
    from f1p.models.world_v1 import WorldV1, ratings_table

    df = ratings_table(WorldV1(), _tables(), top=top)
    for kind in ("team", "driver"):
        table = Table(title=f"{kind} beliefs")
        table.add_column(kind)
        table.add_column("mean", justify="right")
        table.add_column("sd", justify="right")
        for r in df.filter(df["kind"] == kind).iter_rows(named=True):
            table.add_row(r["id"], f"{r['mu']:+.2f}", f"{r['sigma']:.2f}")
        console.print(table)
