"""``f1p data`` — build, inspect and check the lake."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import polars as pl
import typer
from rich.console import Console
from rich.table import Table

from f1p.core.config import get_settings
from f1p.lake.quality import GateResult, QualityError

app = typer.Typer(help="Build, inspect and check the data lake.", no_args_is_help=True)
console = Console()


def _gate_table(gates: list[GateResult]) -> Table:
    table = Table(title="data-quality gates")
    table.add_column("gate")
    table.add_column("result")
    table.add_column("detail")
    for g in gates:
        table.add_row(g.name, "[green]PASS[/green]" if g.passed else "[red]FAIL[/red]", g.summary)
    return table


@app.command()
def update(
    tag: Annotated[
        str | None, typer.Option(help="Pin an F1DB release tag, e.g. v2026.17.0")
    ] = None,
    file: Annotated[
        Path | None, typer.Option(help="Use a local f1db-sqlite.zip or f1db.db (offline)")
    ] = None,
    force: Annotated[
        bool, typer.Option(help="Rebuild even if current; commit despite failed gates")
    ] = False,
) -> None:
    """Fetch F1DB, curate it, run the quality gates and commit a lake snapshot."""
    from f1p.lake import pipeline

    try:
        report = pipeline.update(get_settings(), tag=tag, local_file=file, force=force)
    except QualityError as exc:
        console.print(_gate_table(exc.results))
        console.print(
            "[red]Gates failed — nothing was committed. Re-run with --force to override.[/red]"
        )
        raise typer.Exit(code=1) from exc
    if not report.changed:
        console.print(
            f"Lake already current with F1DB {report.version} (covers {report.coverage})."
        )
        return
    console.print(_gate_table(report.gates))
    for note in report.notes:
        console.print(f"[yellow]note:[/yellow] {note}")
    snap = report.snapshot.snapshot_id if report.snapshot else "?"
    console.print(
        f"[bold green]Committed snapshot {snap}[/bold green] — F1DB {report.version}, "
        f"covers {report.coverage}, {sum(report.row_counts.values()):,} rows in "
        f"{len(report.row_counts)} tables."
    )


@app.command()
def weather(
    start: Annotated[int, typer.Option("--from", help="First season")] = 2000,
    end: Annotated[int | None, typer.Option("--to", help="Last season (default: current)")] = None,
) -> None:
    """Fetch race weather (Open-Meteo): climatology, pre-race forecasts, observed conditions."""
    from datetime import UTC, datetime

    from f1p.lake import pipeline

    last = end or datetime.now(UTC).year
    counts = pipeline.update_weather(get_settings(), seasons=(start, last))
    console.print(
        f"Weather {start}-{last}: "
        + ", ".join(f"{name} {rows:,} rows" for name, rows in counts.items())
    )


@app.command()
def pace(
    start: Annotated[int, typer.Option("--from", help="First season (2018+)")] = 2018,
    end: Annotated[int | None, typer.Option("--to", help="Last season (default: current)")] = None,
) -> None:
    """Build clean-lap race pace from the F1 live-timing archive (2018+)."""
    from datetime import UTC, datetime

    from f1p.lake import pipeline

    last = end or datetime.now(UTC).year
    rows = pipeline.update_race_pace(get_settings(), seasons=(start, last))
    console.print(f"race_pace {start}-{last}: {rows:,} rows")


@app.command()
def laps(
    start: Annotated[int, typer.Option("--from", help="First season (2018+)")] = 2018,
    end: Annotated[int | None, typer.Option("--to", help="Last season (default: current)")] = None,
) -> None:
    """Build the lap table (times, tyres, traffic, pit and SC flags) for every race 2018+."""
    from datetime import UTC, datetime

    from f1p.lake import pipeline

    last = end or datetime.now(UTC).year
    rows = pipeline.update_laps(get_settings(), seasons=(start, last))
    console.print(f"laps {start}-{last}: {rows:,} rows")


@app.command()
def status() -> None:
    """Show the lake's location, snapshots, provenance and table sizes."""
    from f1p.lake.pipeline import lake_for

    settings = get_settings()
    with lake_for(settings, read_only=True) as lake:
        if not lake.exists:
            console.print(f"No lake at {lake.root} yet — run `f1p data update`.")
            raise typer.Exit(code=1)
        snap = lake.latest_snapshot()
        console.print(f"lake: {lake.root}")
        if snap is not None:
            console.print(f"latest snapshot: {snap.snapshot_id} ({snap.committed_at})")
        log_df = lake.ingest_log()
        if log_df.height:
            last = log_df.row(-1, named=True)
            console.print(
                f"built from: {last['source']} {last['version']} at {last['committed_at']}"
            )
        table = Table(title="tables")
        table.add_column("table")
        table.add_column("rows", justify="right")
        for name in lake.tables():
            n = lake.connect().execute(f'SELECT count(*) FROM lake."{name}"').fetchone()
            table.add_row(name, f"{n[0]:,}" if n else "?")
        console.print(table)


@app.command()
def check() -> None:
    """Re-run every data-quality gate against the current snapshot."""
    from f1p.lake import pipeline

    gates = pipeline.check(get_settings())
    console.print(_gate_table(gates))
    if not all(g.passed for g in gates):
        for g in gates:
            if not g.passed:
                console.print(f"[red]{g.name}[/red]")
                console.print(g.failures.head(20))
        raise typer.Exit(code=1)


@app.command()
def show(
    table: Annotated[str, typer.Argument(help="Curated table name, e.g. results")],
    event: Annotated[str | None, typer.Option(help="Filter by event id, e.g. 2026-R17")] = None,
    session: Annotated[str | None, typer.Option(help="Filter by session, e.g. race")] = None,
    limit: Annotated[int, typer.Option(help="Rows to print")] = 25,
    snapshot: Annotated[int | None, typer.Option(help="Read at an older snapshot")] = None,
) -> None:
    """Print rows from a curated table (optionally at an older snapshot)."""
    from f1p.lake.pipeline import lake_for

    with lake_for(get_settings(), read_only=True) as lake:
        if not lake.exists:
            console.print("[red]No lake — run `f1p data update` first.[/red]")
            raise typer.Exit(code=1)
        if table not in lake.tables():
            console.print(
                f"[red]unknown table {table!r}[/red]; available: {', '.join(lake.tables())}"
            )
            raise typer.Exit(code=1)
        df = lake.read(table, snapshot=snapshot)
    if event and "event_id" in df.columns:
        df = df.filter(pl.col("event_id") == event)
    if session and "session" in df.columns:
        df = df.filter(pl.col("session") == session)
    with pl.Config(tbl_rows=limit, tbl_width_chars=200, tbl_cols=14):
        console.print(df.head(limit))
