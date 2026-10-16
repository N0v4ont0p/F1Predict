"""``f1p tune`` — hyper-parameter searches over the models (plan M6 §C)."""

from __future__ import annotations

import json
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from f1p.core.config import get_settings

app = typer.Typer(help="Hyper-parameter search: run, select, report.", no_args_is_help=True)
console = Console()


def _names() -> tuple[str, ...]:
    from f1p.cli.eval import EVAL_TABLES

    return (*EVAL_TABLES, "circuit_geometry")


@app.command()
def run(
    space: Annotated[str, typer.Argument(help="combiner, filter, race_sim or quali")],
    trials: Annotated[int, typer.Option(help="Trials (resumes the study if it exists)")] = 100,
    workers: Annotated[
        int, typer.Option(help="Worker processes (0: one per performance core)")
    ] = 0,
    metric: Annotated[
        str, typer.Option(help="Per-event metric to optimise")
    ] = "finisher_pl_loglik",
    name: Annotated[str | None, typer.Option(help="Study name (default: the space)")] = None,
) -> None:
    """Search ``space`` on the tune split (2014-2021), pruning by season."""
    from f1p.eval import runner
    from f1p.tuning import search
    from f1p.tuning.spaces import SPACES

    if space not in SPACES:
        console.print(f"[red]unknown space {space!r}[/red]; choose {', '.join(SPACES)}")
        raise typer.Exit(code=1)
    settings = get_settings()
    study_name = name or space
    path = search.study_path(settings.paths.artifacts, study_name)
    study = search.optimize(
        path,
        study_name,
        space,
        trials=trials,
        workers=workers or runner.cpu_workers(),
        metric=metric,
        cache_dir=settings.paths.artifacts / "cache" / "prepared",
        table_names=_names(),
    )
    best = study.best_trial
    console.print(
        f"{len(study.trials)} trials; best #{best.number}: {best.value:.5f} {best.params}"
    )
    console.print(f"study → {path}")


@app.command()
def select(
    space: Annotated[str, typer.Argument(help="The space the study searched")],
    top: Annotated[int, typer.Option(help="Best tune trials to re-score on select")] = 5,
    metric: Annotated[str, typer.Option(help="Per-event metric")] = "finisher_pl_loglik",
    name: Annotated[str | None, typer.Option(help="Study name (default: the space)")] = None,
) -> None:
    """Re-score the best trials on select; choose the simplest within one SE (plan §C.4)."""
    from f1p.tuning import search
    from f1p.tuning.spaces import SPACES

    settings = get_settings()
    study_name = name or space
    study = search.open_study(
        search.study_path(settings.paths.artifacts, study_name), study_name, SPACES[space]
    )
    tables = search.load_tables(_names())
    sel = search.select(
        study,
        space,
        tables,
        top=top,
        metric=metric,
        cache_dir=settings.paths.artifacts / "cache" / "prepared",
    )
    table = Table(
        title=f"{study_name}: {sel.trials} trials ({sel.complete} complete, {sel.pruned} pruned)"
    )
    for c in ("trial", "tune loss", "select loss", "SE vs best", "complexity", "params"):
        table.add_column(c)
    for c in sel.candidates:
        mark = (
            " ← chosen" if c.number == sel.chosen.number else (" (hand-set)" if c.baseline else "")
        )
        table.add_row(
            f"#{c.number}{mark}",
            f"{c.tune:.5f}",
            f"{c.select:.5f}",
            f"{c.se:.5f}",
            f"{c.complexity:.2f}",
            json.dumps(
                {k: round(v, 4) if isinstance(v, float) else v for k, v in c.params.items()}
            ),
        )
    console.print(table)
    if sel.null:
        n = sel.null
        console.print(
            f"search optimism: best tune improvement {n['best_improvement']:.5f}, "
            f"expected max by chance {n['null_mean_max']:.5f} "
            f"(95%: {n['null_q95']:.5f}), p = {n['p']:.3f}"
        )
    out = search.save_selection(
        sel, settings.paths.artifacts / "tuning" / f"{study_name}-selection.json"
    )
    console.print(f"saved → {out}")
