"""``f1p sim`` — calibration and checks of the race simulator v2 (plan M6 §B.4/§B.6)."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Annotated, Any

import polars as pl
import typer
from rich.console import Console
from rich.table import Table

from f1p.core.config import get_settings

app = typer.Typer(
    help="Race simulator v2: calibration and predictive checks.", no_args_is_help=True
)
console = Console()

SIM_TABLES = [
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
    "laps",
    "practice",
    "sessions",
    "circuit_geometry",
]


def _tables() -> dict[str, Any]:
    from f1p.lake.pipeline import lake_for

    with lake_for(get_settings(), read_only=True) as lake:
        if not lake.exists:
            console.print("[red]No lake — run `f1p data update` first.[/red]")
            raise typer.Exit(code=1)
        present = set(lake.tables())
        return lake.read_all([n for n in SIM_TABLES if n in present])


def _races(split: str, every: int, sims: int) -> tuple[Any, list[Any], dict[str, Any]]:
    from f1p.eval import splits
    from f1p.models import sim_calibrate as sc
    from f1p.models.race_sim_v2 import RaceSimV2, load_knobs

    if split not in splits.SPLITS:
        console.print(f"[red]unknown split {split!r}[/red]")
        raise typer.Exit(code=1)
    lo, hi = splits.SPLITS[split]  # type: ignore[index]
    tables = _tables()
    with_laps = set(tables["laps"]["event_id"].unique().to_list())
    events = (
        tables["events"]
        .filter(pl.col("season").is_between(lo, hi) & pl.col("event_id").is_in(list(with_laps)))
        .sort("season", "round")["event_id"]
        .to_list()
    )[:: max(every, 1)]
    model = RaceSimV2(knobs=load_knobs())
    races = sc.calibration_races(tables, events, model, sims=sims)
    return model, races, tables


@app.command()
def calibrate(
    split: Annotated[
        str, typer.Option(help="Races to calibrate on (searches: tune/select)")
    ] = "tune",
    every: Annotated[int, typer.Option(help="Use every n-th race")] = 2,
    sims: Annotated[int, typer.Option(help="Simulated races per real race per evaluation")] = 300,
    evals: Annotated[int, typer.Option(help="Objective evaluations (Nelder-Mead)")] = 80,
    save: Annotated[bool, typer.Option(help="Write artifacts/sim_v2/knobs.json")] = True,
) -> None:
    """Fit the simulator's global knobs by simulated method of moments (plan §B.4)."""
    from f1p.eval import splits
    from f1p.models import sim_calibrate as sc

    lo, hi = splits.SPLITS.get(split, (0, 0))  # type: ignore[call-overload]
    if split != "holdout":
        splits.search_seasons((lo, hi))
    else:
        console.print("[red]calibration never uses the holdout[/red]")
        raise typer.Exit(code=1)
    model, races, _ = _races(split, every, sims)
    console.print(f"{len(races)} races, {sims} simulations each")
    cal = sc.calibrate(races, start=model.knobs, max_evals=evals)
    title = f"race_sim_v2 calibration ({split})"
    table = Table(title=f"{title}: distance {cal.start_distance:.1f} → {cal.distance:.1f}")
    for c in ("summary", "observed", "simulated"):
        table.add_column(c)
    for k in sc.SUMMARIES:
        table.add_row(k, f"{cal.observed[k]:.3f}", f"{cal.simulated[k]:.3f}")
    console.print(table)
    console.print({k: round(v, 4) for k, v in asdict(cal.knobs).items()})
    if save:
        out = get_settings().paths.artifacts / "sim_v2"
        out.mkdir(parents=True, exist_ok=True)
        (out / "knobs.json").write_text(json.dumps(cal.to_dict(), indent=1))
        console.print(f"saved → {out / 'knobs.json'}")


@app.command()
def ppc(
    split: Annotated[str, typer.Option(help="Races to check")] = "select",
    every: Annotated[int, typer.Option(help="Use every n-th race")] = 1,
    sims: Annotated[int, typer.Option(help="Simulated races per real race")] = 500,
) -> None:
    """Posterior-predictive checks per circuit (plan §B.6 gate 1: every BH-adjusted p > 0.05)."""
    from f1p.models import sim_calibrate as sc

    model, races, _ = _races(split, every, sims)
    df = sc.ppc(races, model.knobs)
    failed = df.filter(pl.col("p_bh") <= 0.05)
    table = Table(title=f"race_sim_v2 PPC ({split}): {failed.height} of {df.height} checks fail")
    for c in ("statistic", "circuits", "worst circuit", "observed", "simulated", "min p (BH)"):
        table.add_column(c)
    for (stat,), g in df.group_by("statistic", maintain_order=True):
        worst = g.sort("p_bh").row(0, named=True)
        table.add_row(
            str(stat),
            str(g.height),
            str(worst["circuit"]),
            f"{worst['observed']:.3f}",
            f"{worst['simulated']:.3f}",
            f"{worst['p_bh']:.3f}",
        )
    console.print(table)
    out = get_settings().paths.artifacts / "sim_v2"
    out.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out / f"ppc-{split}.parquet")
    console.print(f"saved → {out / f'ppc-{split}.parquet'}")
