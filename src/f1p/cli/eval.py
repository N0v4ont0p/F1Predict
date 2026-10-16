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
    "laps",
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
        str | None,
        typer.Option(help="Checkpoints, e.g. C1,C4; C5 = in the race (default: the target's)"),
    ] = None,
    target: Annotated[str, typer.Option(help="race or quali")] = "race",
    reference: Annotated[str, typer.Option(help="Model to compare against")] = "uniform",
    save: Annotated[bool, typer.Option(help="Write per-event metrics to artifacts/eval/")] = True,
    workers: Annotated[
        int, typer.Option(help="Worker processes (0: one per performance core)")
    ] = 0,
    cache: Annotated[
        bool, typer.Option(help="Reuse prepared signals from artifacts/cache/prepared")
    ] = True,
    split: Annotated[
        str | None,
        typer.Option(
            help="tune (2014-21), select (2022-23) or holdout (2024-26); sets the seasons"
        ),
    ] = None,
    rescore: Annotated[
        bool, typer.Option(help="Allow a second holdout scoring of the same item (recorded)")
    ] = False,
) -> None:
    """Score models on every event in [--from, --to] with only then-public data."""
    from f1p.eval import harness, runner, splits
    from f1p.lake.pipeline import lake_for
    from f1p.silicon.compute import compute_log

    if target not in ("race", "quali"):
        console.print(f"[red]unknown target {target!r}[/red]; choose race or quali")
        raise typer.Exit(code=1)
    if split is not None:
        if split not in splits.SPLITS:
            console.print(f"[red]unknown split {split!r}[/red]; choose {', '.join(splits.SPLITS)}")
            raise typer.Exit(code=1)
        start, end = splits.SPLITS[split]  # type: ignore[index]
    catalogue = available_models(target)
    names = list(catalogue) if models == "all" else [m.strip() for m in models.split(",")]
    unknown = [n for n in names if n not in catalogue]
    if unknown:
        console.print(f"[red]unknown model(s) {unknown}[/red]; available: {', '.join(catalogue)}")
        raise typer.Exit(code=1)
    if reference not in names:
        names.append(reference)
    settings = get_settings()
    chosen = [catalogue[n] for n in names]
    ledger = splits.HoldoutLedger(settings.paths.artifacts / "ledger" / "holdout.jsonl")
    sees_holdout = splits.touches_holdout((start, end))
    items = [(m.name, runner.model_signature(m)) for m in chosen]
    if split == "holdout" and not rescore:
        done = [n for n in ledger.check(items, target) if n != reference]
        if done:
            console.print(
                f"[red]already scored on the holdout: {done}[/red] — the holdout is scored once "
                "per item; pass --rescore to score again (it is recorded)."
            )
            raise typer.Exit(code=1)
    elif sees_holdout and split != "holdout":
        console.print(
            f"[yellow]seasons {start}-{end} include the holdout {splits.HOLDOUT}: "
            "use --split tune/select for searches and model choices.[/yellow]"
        )
    with lake_for(settings, read_only=True) as lake:
        if not lake.exists:
            console.print("[red]No lake — run `f1p data update` first.[/red]")
            raise typer.Exit(code=1)
        present = set(lake.tables())
        tables = lake.read_all([n for n in EVAL_TABLES if n in present])
        snap = lake.latest_snapshot()
    n_workers = workers or runner.cpu_workers()
    with compute_log("eval run", workers=n_workers, cache=cache) as compute:
        result = harness.run(
            tables,
            chosen,
            seasons=(start, end),
            history_from=history,
            checkpoints=tuple(c.strip() for c in checkpoints.split(",")) if checkpoints else None,
            target=target,
            workers=n_workers,
            cache_dir=settings.paths.artifacts / "cache" / "prepared" if cache else None,
        )
    console.print(_summary_table(result, reference))
    console.print(
        f"[dim]{compute.wall_s:.1f}s wall, {compute.cpu_s:.0f}s cpu, {n_workers} workers, "
        f"peak {max(compute.peak_rss_mb, compute.peak_child_rss_mb):.0f} MB[/dim]"
    )
    if sees_holdout:
        kind = ("rescore" if rescore else "holdout") if split == "holdout" else "adhoc"
        summary = result.summary()
        for name, sig in items:
            rows = summary.filter(pl.col("model") == name).to_dicts()
            ledger.record(
                name, sig, target=target, seasons=(start, end), kind=kind, summary={"rows": rows}
            )
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
            "split": split,
            "signatures": dict(items),
            "compute": compute.to_dict(),
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
def pace(
    start: Annotated[int, typer.Option("--from", help="First season")] = 2019,
    end: Annotated[int, typer.Option("--to", help="Last season")] = 2026,
    window: Annotated[int, typer.Option(help="Races in the lap model's window")] = 24,
) -> None:
    """Held-out pace: predict each car's race-pace deficit from earlier races only."""
    from f1p.eval.bootstrap import paired_bootstrap
    from f1p.lake.pipeline import lake_for
    from f1p.models.lap_model import pace_backtest

    with lake_for(get_settings(), read_only=True) as lake:
        present = set(lake.tables())
        if not {"laps", "race_pace"} <= present:
            console.print("[red]needs `f1p data pace` and `f1p data laps` first[/red]")
            raise typer.Exit(code=1)
        tables = lake.read_all([n for n in EVAL_TABLES if n in present])
    df = pace_backtest(tables, (start, end), window=window)
    summary = (
        df.group_by("predictor")
        .agg(pl.len().alias("events"), pl.col("rmse").mean(), pl.col("spearman").mean())
        .sort("rmse")
    )
    table = Table(title=f"held-out race pace {start}-{end} (pp of a lap; Δ vs ew_race_pace)")
    for col in ("predictor", "events", "RMSE", "Spearman", "Δ RMSE", "95% CI"):
        table.add_column(col, justify="left" if col == "predictor" else "right")
    ref = df.filter(pl.col("predictor") == "ew_race_pace").select("event_id", "rmse")
    for r in summary.iter_rows(named=True):
        j = (
            df.filter(pl.col("predictor") == r["predictor"])
            .select("event_id", "rmse")
            .join(ref, on="event_id", suffix="_ref")
        )
        c = paired_bootstrap(j["rmse"].to_numpy(), j["rmse_ref"].to_numpy())
        table.add_row(
            r["predictor"],
            str(r["events"]),
            f"{r['rmse']:.3f}",
            f"{r['spearman']:.3f}",
            f"{c.mean_diff:+.3f}",
            f"[{c.lo:+.3f}, {c.hi:+.3f}]",
        )
    console.print(table)


STACK_V2 = "world_v1,race_sim_v2,tabicl_pointwise,lapseq,lgbm_rank"


@app.command()
def stack(
    members: Annotated[
        str, typer.Option(help="Comma-separated members (default: stack v2's, plan §E)")
    ] = STACK_V2,
    start: Annotated[int, typer.Option("--from", help="First evaluated season")] = 2022,
    end: Annotated[int, typer.Option("--to", help="Last evaluated season")] = 2023,
    training: Annotated[int, typer.Option(help="Seasons the pool weights are fitted on")] = 4,
    reference: Annotated[str, typer.Option(help="Model to compare against")] = "world_v1",
    checkpoints: Annotated[str, typer.Option(help="Checkpoints, e.g. C1,C4")] = "C1,C4",
    recalibrate: Annotated[
        bool,
        typer.Option(help="Also report an isotonic recalibration of the pooled P(win)"),
    ] = False,
    split: Annotated[
        str | None,
        typer.Option(help="select (2022-23) or holdout (2024-26, scored once); sets the seasons"),
    ] = None,
    rescore: Annotated[
        bool, typer.Option(help="Allow a second holdout scoring of the same stack (recorded)")
    ] = False,
    workers: Annotated[
        int, typer.Option(help="Worker processes (0: one per performance core)")
    ] = 1,
) -> None:
    """Stack member models (log-linear pool, weights fitted on earlier seasons, P(win)
    recalibrated) and compare. On the holdout, each stack (its members' signatures and its
    settings) is scored once, through the holdout ledger."""
    from f1p.eval import harness, runner, splits
    from f1p.lake.pipeline import lake_for
    from f1p.models.lapseq import digest
    from f1p.models.stacking import stack_result

    if split is not None:
        if split not in splits.SPLITS:
            console.print(f"[red]unknown split {split!r}[/red]; choose {', '.join(splits.SPLITS)}")
            raise typer.Exit(code=1)
        start, end = splits.SPLITS[split]  # type: ignore[index]
    catalogue = available_models()
    names = [m.strip() for m in members.split(",")]
    if reference not in names:
        names.append(reference)
    unknown = [n for n in names if n not in catalogue]
    if unknown:
        console.print(f"[red]unknown model(s) {unknown}[/red]; available: {', '.join(catalogue)}")
        raise typer.Exit(code=1)
    cps = tuple(c.strip() for c in checkpoints.split(","))
    settings = get_settings()
    ledger = splits.HoldoutLedger(settings.paths.artifacts / "ledger" / "holdout.jsonl")
    item = "stack:" + "+".join(names)
    signature = digest(
        tuple((n, runner.model_signature(catalogue[n])) for n in names),
        training,
        cps,
        recalibrate,
    )
    sees_holdout = splits.touches_holdout((start, end))
    if split == "holdout" and not rescore and ledger.check([(item, signature)], "race"):
        console.print(
            "[red]this stack was already scored on the holdout[/red] — pass --rescore to "
            "score it again (it is recorded)."
        )
        raise typer.Exit(code=1)
    if sees_holdout and split != "holdout":
        console.print(
            f"[yellow]seasons {start}-{end} include the holdout {splits.HOLDOUT}: "
            "use --split select for model choices.[/yellow]"
        )
    with lake_for(settings, read_only=True) as lake:
        present = set(lake.tables())
        tables = lake.read_all([n for n in EVAL_TABLES if n in present])
    first = start - training
    result = harness.run(
        tables,
        [catalogue[n] for n in names],
        seasons=(first, end),
        history_from=first - 8,
        checkpoints=cps,
        keep_forecasts=True,
        workers=workers or runner.cpu_workers(),
        cache_dir=settings.paths.artifacts / "cache" / "prepared",
    )
    stacked, fits = stack_result(
        tables,
        result,
        names,
        seasons=(start, end),
        training_seasons=training,
        recalibrate=recalibrate,
    )
    if stacked.per_event.is_empty():
        console.print("[red]no event had a forecast from every member[/red]")
        raise typer.Exit(code=1)
    console.print(_summary_table(stacked, reference))
    for (season, cp), fit in sorted(fits.items()):
        weights = ", ".join(f"{m} {w:.2f}" for m, w in fit.weights.items())
        console.print(f"  {season} {cp}: {weights}", style="dim")
    summary = stacked.summary()
    if sees_holdout:
        kind = ("rescore" if rescore else "holdout") if split == "holdout" else "adhoc"
        ledger.record(
            item,
            signature,
            target="race",
            seasons=(start, end),
            kind=kind,
            summary={"rows": summary.to_dicts()},
        )
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = settings.paths.artifacts / "eval" / f"stack-{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    stacked.per_event.write_parquet(out / "per_event.parquet")
    meta = {
        "seasons": [start, end],
        "split": split,
        "members": names,
        "checkpoints": list(cps),
        "training_seasons": training,
        "recalibrate": recalibrate,
        "signature": signature,
        "weights": {f"{s} {c}": f.weights for (s, c), f in sorted(fits.items())},
        "summary": summary.to_dicts(),
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
