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
def championship(
    season: Annotated[int | None, typer.Option(help="Season (default: latest in the lake)")] = None,
    after_round: Annotated[
        int | None,
        typer.Option("--after-round", help="Project as of this round's result (0: pre-season)"),
    ] = None,
    as_of: Annotated[
        str | None, typer.Option("--as-of", help="Project as of a date/time, e.g. 2025-08-01")
    ] = None,
    sims: Annotated[int, typer.Option(help="Simulated seasons")] = 50_000,
    kind: Annotated[str, typer.Option(help="driver, constructor or both")] = "both",
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON instead of tables")] = False,
) -> None:
    """The projected final championship table — now, or as it looked at any point."""
    from datetime import UTC, datetime

    from f1p.models.season import simulate_season

    if after_round is not None and as_of is not None:
        console.print("[red]use --after-round or --as-of, not both[/red]")
        raise typer.Exit(code=1)
    tables = _tables() | _extra_tables(["constructor_standings"])
    last = season or int(tables["events"]["season"].max())  # type: ignore[arg-type]
    cutoff = None
    if as_of:
        cutoff = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
        cutoff = cutoff if cutoff.tzinfo else cutoff.replace(tzinfo=UTC)
    try:
        o = simulate_season(tables, last, n_sims=sims, after_round=after_round, as_of=cutoff)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    kinds = ("driver", "constructor") if kind == "both" else (kind,)

    def row(s):  # type: ignore[no-untyped-def]
        return {
            "id": s.id,
            "most_likely": s.most_likely + 1,
            "current": s.current,
            "expected": s.expected,
            "median": s.quantiles[2],
            "range50": [s.quantiles[1], s.quantiles[3]],
            "range90": [s.quantiles[0], s.quantiles[4]],
            "p_title": s.p_title,
            "p_top3": s.p_top3,
            "p_position": list(s.p_position),
        }

    if as_json:
        payload = {
            "season": last,
            "as_of": o.as_of.isoformat(),
            "after_round": o.after_round,
            "remaining": list(o.remaining),
            "n_sims": o.n_sims,
        }
        for k in kinds:
            payload[f"{k}s"] = [row(s) for s in (o.drivers if k == "driver" else o.constructors)]
        typer.echo(json.dumps(payload, indent=2))
        return
    for k in kinds:
        rows = o.drivers if k == "driver" else o.constructors
        label = "drivers" if k == "driver" else "constructors"
        table = Table(
            title=f"{last} {label} · projected after round {o.after_round}"
            f" · {len(o.remaining)} rounds to go",
            caption=f"data public at {o.as_of:%Y-%m-%d %H:%M} UTC · {o.n_sims:,} simulated seasons",
        )
        for col in ("#", k, "now", "expected", "median", "50%", "90%", "title", "top 3"):
            table.add_column(col, justify="left" if col == k else "right")
        for s in rows:
            q = s.quantiles
            table.add_row(
                str(s.most_likely + 1),
                s.id,
                f"{s.current:g}",
                f"{s.expected:.0f}",
                f"{q[2]:.0f}",
                f"{q[1]:.0f}-{q[3]:.0f}",
                f"{q[0]:.0f}-{q[4]:.0f}",
                _pct(s.p_title),
                _pct(s.p_top3),
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


@app.command()
def publish(
    season: Annotated[int | None, typer.Option(help="Season (default: latest in the lake)")] = None,
    out: Annotated[
        str | None, typer.Option(help="Output directory (default artifacts/marts)")
    ] = None,
    leaderboard: Annotated[
        bool, typer.Option(help="Also rebuild the 2014+ leaderboard (a few minutes)")
    ] = True,
) -> None:
    """Write the JSON marts the web app reads (forecasts, results, ratings, model card)."""
    from pathlib import Path

    from f1p import publish as pub
    from f1p.lake.pipeline import lake_for

    with lake_for(get_settings(), read_only=True) as lake:
        snap = lake.latest_snapshot()
    tables = _tables()
    tables |= _extra_tables(
        ["drivers", "constructors", "weather_observed", "constructor_standings"]
    )
    last = season or int(tables["events"]["season"].max())  # type: ignore[arg-type]
    out_dir = Path(out) if out else get_settings().paths.artifacts / "marts"
    files = pub.publish(
        tables,
        out_dir,
        season=last,
        snapshot=snap.snapshot_id if snap else None,
        leaderboard_from=2014 if leaderboard else None,
    )
    console.print(f"published {len(files)} files → {out_dir}")


def _extra_tables(names: list[str]) -> dict:  # type: ignore[type-arg]
    from f1p.lake.pipeline import lake_for

    with lake_for(get_settings(), read_only=True) as lake:
        present = set(lake.tables())
        return lake.read_all([n for n in names if n in present])
