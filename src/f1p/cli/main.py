"""``f1p`` — the f1predict command line.

Phase 0 ships the foundation commands; later phases register theirs on the same ``app``.
"""

from __future__ import annotations

import platform
import sys
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from f1p import __version__
from f1p.cli import data as data_cli
from f1p.cli import eval as eval_cli
from f1p.cli import lap as lap_cli
from f1p.cli import lapseq as lapseq_cli
from f1p.cli import model as model_cli
from f1p.cli import silicon as silicon_cli
from f1p.cli import sim as sim_cli
from f1p.cli import tune as tune_cli
from f1p.core import points
from f1p.core.config import get_settings

app = typer.Typer(
    name="f1p",
    help="Probabilistic Formula 1 forecasting.",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(data_cli.app, name="data")
app.add_typer(eval_cli.app, name="eval")
app.add_typer(lap_cli.app, name="lap")
app.add_typer(lapseq_cli.app, name="lapseq")
app.add_typer(model_cli.app, name="model")
app.add_typer(silicon_cli.app, name="silicon")
app.add_typer(sim_cli.app, name="sim")
app.add_typer(tune_cli.app, name="tune")
console = Console()


def _is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


@app.command()
def version() -> None:
    """Print the installed version."""
    console.print(f"f1predict {__version__}")


@app.command()
def doctor() -> None:
    """Check the environment: Python, platform, paths and memory budget."""
    settings = get_settings()
    table = Table(title="f1p doctor", show_header=False)
    table.add_column("check", style="bold")
    table.add_column("value")
    table.add_row("version", __version__)
    table.add_row("python", platform.python_version())
    table.add_row("platform", f"{sys.platform} / {platform.machine()}")
    table.add_row(
        "apple silicon",
        "yes" if _is_apple_silicon() else "no (CPU fallbacks will be used)",
    )
    table.add_row("data dir", str(settings.paths.data))
    table.add_row("artifacts dir", str(settings.paths.artifacts))
    budget = settings.memory
    table.add_row(
        "memory budget",
        f"{budget.working_gb:g} GB working / {budget.reserved_gb:g} GB reserved "
        f"/ {budget.total_gb:g} GB total",
    )
    console.print(table)


@app.command("points")
def points_cmd(
    season: Annotated[int, typer.Argument(help="Championship season, e.g. 2026")],
    sprint: Annotated[bool, typer.Option("--sprint", help="Show the sprint scale")] = False,
    distance: Annotated[
        float, typer.Option(help="Share of race distance completed (0-1)", min=0.0, max=1.0)
    ] = 1.0,
) -> None:
    """Show the points scale for a season (and, optionally, a shortened race)."""
    session = points.Session.SPRINT if sprint else points.Session.RACE
    conditions = points.RaceConditions(distance_fraction=distance)
    try:
        scale = points.scale(season, session, conditions)
    except (NotImplementedError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    table = Table(title=f"{season} {session.value} points (distance {distance:.0%})")
    table.add_column("position", justify="right")
    table.add_column("points", justify="right")
    for pos, pts in enumerate(scale, start=1):
        table.add_row(str(pos), f"{pts:g}")
    console.print(table)
    if points.fastest_lap_bonus_applies(season, session, conditions):
        console.print("+1 for the fastest lap if classified in the top 10")


if __name__ == "__main__":  # pragma: no cover
    app()
