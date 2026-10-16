"""``f1p dev`` — the localhost job server behind the web app's dev panel (``/dev``)."""

from __future__ import annotations

import time
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from f1p.core.config import get_settings

app = typer.Typer(help="Dev panel: run checks and the M6 checklist from the web app.")
console = Console()

DEFAULT_PORT = 8765


@app.command()
def serve(
    port: Annotated[int, typer.Option(help="Port on 127.0.0.1")] = DEFAULT_PORT,
    parallel: Annotated[int, typer.Option(help="Jobs allowed to run at the same time")] = 1,
) -> None:
    """Serve the dev panel's API until Ctrl-C. Open the web app's /dev page to use it."""
    from f1p.devpanel.jobs import JobManager
    from f1p.devpanel.server import serve as start

    settings = get_settings()
    manager = JobManager(
        settings.paths.artifacts / "dev", cwd=settings.paths.root, parallel=parallel
    )
    try:
        server = start(manager, port)
    except OSError as exc:
        console.print(f"[red]could not listen on 127.0.0.1:{port}: {exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(
        f"dev panel API on http://127.0.0.1:{port} — open the web app's /dev page "
        f"(logs in {settings.paths.artifacts / 'dev'}). Ctrl-C to stop."
    )
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        console.print("stopping: running jobs are terminated")
    finally:
        manager.shutdown()
        server.shutdown()


@app.command("tasks")
def list_tasks() -> None:
    """List the tasks the panel can run."""
    from f1p.devpanel.tasks import PIPELINES, TASKS

    table = Table(title="dev panel tasks")
    for col in ("id", "group", "title", "steps"):
        table.add_column(col)
    for t in TASKS:
        steps = "\n".join(" ".join(s) for s in t.steps) or f"[dim]manual: {t.manual}[/dim]"
        table.add_row(t.id, t.group, t.title, steps)
    console.print(table)
    for name, ids in PIPELINES.items():
        console.print(f"pipeline {name}: {' → '.join(ids)}")
