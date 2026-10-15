"""``f1p silicon`` — what this machine offers, and how fast each unit runs our workloads."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from f1p.core.config import get_settings

app = typer.Typer(help="Apple silicon runtime: device info and benchmarks.", no_args_is_help=True)
console = Console()


@app.command()
def info() -> None:
    """CPU clusters, memory, GPU, Neural Engine, BLAS and available ML back ends."""
    from f1p.silicon.device import detect
    from f1p.silicon.kernels import preferred_sampler

    d = detect()
    table = Table(title="f1p silicon", show_header=False)
    table.add_column("item", style="bold")
    table.add_column("value")
    table.add_row("chip", d.chip)
    table.add_row("os / arch", f"{d.os} / {d.arch}")
    table.add_row("apple silicon", "yes" if d.apple_silicon else "no")
    table.add_row("cpu clusters", ", ".join(f"{c.cores}x {c.name}" for c in d.clusters))
    table.add_row("memory", f"{d.memory_gb:g} GB")
    table.add_row("gpu cores", str(d.gpu_cores or "-"))
    table.add_row("neural engine", "yes" if d.neural_engine else "no")
    table.add_row("blas", d.blas)
    table.add_row("backends", ", ".join(k for k, v in d.backends.items() if v) or "numpy only")
    table.add_row("race sampler", preferred_sampler())
    console.print(table)


@app.command()
def bench(
    quick: Annotated[bool, typer.Option(help="Small sizes (seconds, for CI)")] = False,
    sustained: Annotated[
        float, typer.Option(help="Minutes to loop each workload (0 = burst timing)")
    ] = 0.0,
    save: Annotated[bool, typer.Option(help="Write JSON to artifacts/bench/")] = True,
    baseline: Annotated[
        str | None, typer.Option(help="Compare against a saved bench JSON (flags >15% drops)")
    ] = None,
) -> None:
    """Time the pipeline's workloads on every available back end."""
    import json
    from dataclasses import asdict
    from pathlib import Path

    from rich.markup import escape

    from f1p import benchmarks as b

    results, info = b.run(quick=quick, sustained_minutes=sustained)
    mode = f"sustained {sustained:g} min" if sustained else ("quick" if quick else "burst")
    table = Table(title=f"f1p silicon bench ({mode})")
    cols = ["benchmark", "backend", "size", "time/call", "throughput"]
    if sustained:
        cols += ["p95", "worst", "last/first"]
    for col in cols:
        table.add_column(
            col, justify="left" if col in ("benchmark", "backend", "size") else "right"
        )
    for r in results:
        row = [
            r.name,
            r.backend,
            r.detail,
            f"{r.seconds * 1e3:,.1f} ms",
            f"{r.throughput:,.1f} {r.unit}",
        ]
        if sustained:
            s = r.sustained
            row += [f"{s['p95']:,.1f}", f"{s['worst']:,.1f}", f"{s['last_over_first']:.2f}"]
        table.add_row(*row)
    console.print(table)
    for k, v in info.items():
        console.print(f"{k}: {v}")
    if save:
        path = b.save(results, get_settings().paths.artifacts / "bench", info)
        console.print(f"saved → {path}")
    if baseline:
        ref = json.loads(Path(baseline).read_text())["results"]
        drops = b.compare([asdict(r) for r in results], ref)
        if drops:
            console.print("[red]regressions vs baseline:[/red]")
            for d in drops:
                console.print(f"  {escape(d)}")
            raise typer.Exit(code=1)
        console.print("[green]no regression vs baseline[/green]")
