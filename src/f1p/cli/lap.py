"""``f1p lap`` — the Bayesian lap model ``lap_v2`` (plan M6 §A): fit, validate, bench, SBC.

These are the long jobs of the M6 plan. ``validate`` and ``sbc`` are resumable: each race (or
simulation batch) is written as it finishes, and a rerun with the same settings skips it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from f1p.core.config import get_settings

app = typer.Typer(
    help="Bayesian lap model (lap_v2): fit, validate, bench, SBC.", no_args_is_help=True
)
console = Console()

LAP_TABLES = [
    "events",
    "results",
    "entries",
    "laps",
    "race_pace",
    "constructor_seasons",
    "circuits",
    "circuit_geometry",
    "weather_observed",
]


def _tables() -> dict[str, Any]:
    from f1p.lake.pipeline import lake_for

    with lake_for(get_settings(), read_only=True) as lake:
        if not lake.exists:
            console.print("[red]No lake — run `f1p data update` first.[/red]")
            raise typer.Exit(code=1)
        present = set(lake.tables())
        return lake.read_all([n for n in LAP_TABLES if n in present])


def _config(sampler: str, draws: int, tune: int, chains: int, adaptation: str, seed: int) -> Any:
    from f1p.models.lap_v2 import SAMPLERS, SampleConfig

    if sampler not in SAMPLERS:
        console.print(f"[red]unknown sampler {sampler!r}[/red]; choose from {', '.join(SAMPLERS)}")
        raise typer.Exit(code=1)
    return SampleConfig(
        sampler=sampler, draws=draws, tune=tune, chains=chains, adaptation=adaptation, seed=seed
    )


def _tag(*parts: Any) -> str:
    return hashlib.sha256(repr(parts).encode()).hexdigest()[:12]


Sampler = Annotated[
    str, typer.Option(help="nutpie, nutpie-flow, nutpie-mclmc, numpyro, blackjax, pymc")
]
Draws = Annotated[int, typer.Option(help="Draws per chain")]
Tune = Annotated[int, typer.Option(help="Tuning steps per chain")]
Chains = Annotated[int, typer.Option(help="Chains (one per performance core)")]
Adaptation = Annotated[str, typer.Option(help="nutpie mass matrix: diag or low_rank")]
Window = Annotated[int, typer.Option(help="Races in the fitted window")]
Evolution = Annotated[bool, typer.Option(help="Include the track-evolution term (plan §A.5)")]
Store = Annotated[
    str,
    typer.Option(
        help="Posterior store under artifacts/posteriors/lap_v2/ (read by world_v1_lapv2)"
    ),
]


@app.command()
def fit(
    event: Annotated[str, typer.Argument(help="The race the window forecasts, e.g. 2024-R03")],
    window: Window = 24,
    sampler: Sampler = "nutpie",
    draws: Draws = 1000,
    tune: Tune = 1000,
    chains: Chains = 4,
    adaptation: Adaptation = "diag",
    evolution: Evolution = True,
    seed: int = 7,
    save: Annotated[bool, typer.Option(help="Write the posterior to the store")] = True,
    store: Store = "default",
) -> None:
    """Fit the window before EVENT. With the race's laps in the lake they are held out and
    scored; otherwise its entry list defines the forecast."""
    import numpy as np
    import polars as pl

    from f1p.models import lap_validate as lv
    from f1p.models.lap_data import build_window
    from f1p.models.lap_v2 import V2Priors, car_pace_draws, save_posterior
    from f1p.models.lap_v2 import fit as fit_v2

    tables = _tables()
    cfg = _config(sampler, draws, tune, chains, adaptation, seed)
    priors = V2Priors(track_evolution=evolution)
    has_laps = tables["laps"].filter(pl.col("event_id") == event).height > 0
    if has_laps:
        w = lv.holdout_window(tables, event, window)
    else:
        ev = tables["events"].filter(pl.col("event_id") == event)
        if ev.height == 0:
            console.print(f"[red]unknown event {event!r}[/red]")
            raise typer.Exit(code=1)
        s, r = int(ev["season"][0]), int(ev["round"][0])
        ent = tables["entries"].filter(
            (pl.col("season") == s) & (pl.col("round") == r) & ~pl.col("test_driver")
        )
        laps, _ = lv.window_before(tables, event, window)
        entries = [
            (str(d), str(c)) for d, c in ent.select("driver_id", "constructor_id").iter_rows()
        ]
        w = build_window(laps, tables, next_event=event, entries=entries)
    if w is None:
        console.print("[red]not enough laps before this race[/red]")
        raise typer.Exit(code=1)
    console.print(w.summary())
    post = fit_v2(w, cfg, priors=priors)
    d = post.diagnostics
    console.print(
        f"{'[green]converged[/green]' if d.ok else '[yellow]NOT converged[/yellow]'}: "
        f"R-hat {d.max_rhat:.3f}, ESS bulk/tail {d.min_ess_bulk:.0f}/{d.min_ess_tail:.0f}, "
        f"{d.divergences} divergences, E-BFMI {d.min_bfmi:.2f}, "
        f"{d.wall_s:.0f}s sampling + {d.compile_s:.0f}s compile ({d.ess_per_s:.2f} ESS/s)"
    )
    drivers = [x for x, _ in w.entries]
    pace = 100.0 * car_pace_draws(post, drivers)
    pace = pace - np.nanmean(pace, axis=1, keepdims=True)
    truth = lv.realised_pace(tables, event) if has_laps else {}
    table = Table(title=f"pace at {event} (% of a lap vs the field, lower = faster)")
    for col in ("driver", "team", "mean", "80% interval", "actual"):
        table.add_column(col, justify="left" if col in ("driver", "team") else "right")
    order = np.argsort(np.nanmean(pace, axis=0))
    for i in order:
        lo, hi = np.nanquantile(pace[:, i], [0.1, 0.9])
        act = truth.get(drivers[i])
        table.add_row(
            drivers[i],
            w.entries[i][1],
            f"{np.nanmean(pace[:, i]):+.3f}",
            f"[{lo:+.3f}, {hi:+.3f}]",
            "" if act is None else f"{act:+.3f}",
        )
    console.print(table)
    if has_laps:
        d2, d1 = lv.v2_lap_density(post), lv.v1_lap_density(w, tables)
        ok = np.isfinite(d1) & np.isfinite(d2)
        console.print(
            f"held-out elpd per lap: v2 {d2[ok].mean():.4f}, v1 {d1[ok].mean():.4f} "
            f"({int(ok.sum())} laps)"
        )
    if save:
        out = save_posterior(
            post,
            get_settings().paths.artifacts / "posteriors" / "lap_v2" / store / event,
            priors=priors,
        )
        console.print(f"saved → {out}")


@app.command()
def validate(
    split: Annotated[
        str, typer.Option(help="tune, select or holdout (scored once, logged)")
    ] = "select",
    every: Annotated[int, typer.Option(help="Score every n-th race (1: all)")] = 1,
    window: Window = 24,
    sampler: Sampler = "nutpie",
    draws: Draws = 1000,
    tune: Tune = 1000,
    chains: Chains = 4,
    adaptation: Adaptation = "diag",
    evolution: Evolution = True,
    seed: int = 7,
    rescore: Annotated[bool, typer.Option(help="Allow a second holdout scoring (logged)")] = False,
    store: Store = "default",
) -> None:
    """Held-out validation over a split (plan §A.4/§A.7): per race, lap_v2 against v1 on the
    next race's laps, car-pace coverage and convergence. Resumable."""
    import polars as pl

    from f1p.eval import splits
    from f1p.models import lap_validate as lv
    from f1p.models.lap_v2 import V2Priors

    if split not in splits.SPLITS:
        console.print(f"[red]unknown split {split!r}[/red]")
        raise typer.Exit(code=1)
    lo, hi = splits.SPLITS[split]  # type: ignore[index]
    cfg = _config(sampler, draws, tune, chains, adaptation, seed)
    priors = V2Priors(track_evolution=evolution)
    tag = _tag(asdict(priors), asdict(cfg), window)
    settings = get_settings()
    ledger = splits.HoldoutLedger(settings.paths.artifacts / "ledger" / "holdout.jsonl")
    item = ("lap_v2", tag)
    if split == "holdout" and not rescore and ledger.check([item], "laps"):
        console.print("[red]lap_v2 with these settings was already scored on the holdout[/red]")
        raise typer.Exit(code=1)
    tables = _tables()
    with_laps = set(tables["laps"]["event_id"].unique().to_list())
    events = (
        tables["events"]
        .filter(pl.col("season").is_between(lo, hi) & pl.col("event_id").is_in(list(with_laps)))
        .sort("season", "round")["event_id"]
        .to_list()
    )[:: max(every, 1)]
    out = settings.paths.artifacts / "lap_v2" / "validate" / tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "settings.json").write_text(
        json.dumps({"priors": asdict(priors), "cfg": asdict(cfg), "window": window, "split": split})
    )
    post_dir = settings.paths.artifacts / "posteriors" / "lap_v2" / store
    rows = []
    for i, ev in enumerate(events, 1):
        path = out / f"{ev}.json"
        if path.exists():
            rows.append(json.loads(path.read_text()))
            continue
        console.print(f"[{i}/{len(events)}] {ev}")
        row = lv.validate_event(
            tables, ev, cfg=cfg, priors=priors, window=window, save_dir=post_dir
        )
        if row is None:
            continue
        path.write_text(json.dumps(row))
        rows.append(row)
        console.print(
            f"  elpd/lap v2 {row['elpd_v2_per_lap']:.4f} v1 {row['elpd_v1_per_lap']:.4f}, "
            f"80% coverage {row['cov80_in']}/{row['cov80_n']}, "
            f"R-hat {row['diag_max_rhat']:.3f}, {row['diag_wall_s']:.0f}s"
        )
    summary = lv.summarise(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    table = Table(title=f"lap_v2 validation — {split} ({len(rows)} races)", show_header=False)
    for k, v in summary.items():
        table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
    console.print(table)
    if split == "holdout":
        ledger.record(
            *item,
            target="laps",
            seasons=(lo, hi),
            kind="rescore" if rescore else "holdout",
            summary=summary,
        )
    console.print(f"saved → {out}")


def _bench_one(event: str, window: int, cfg: Any, evolution: bool) -> dict[str, Any]:
    """One sampler, in its own process (so its peak memory is its own)."""
    import resource
    import sys

    from f1p.models import lap_validate as lv
    from f1p.models.lap_v2 import V2Priors
    from f1p.models.lap_v2 import fit as fit_v2

    tables = _tables()
    w = lv.holdout_window(tables, event, window)
    assert w is not None
    post = fit_v2(w, cfg, priors=V2Priors(track_evolution=evolution))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak = rss / 2**20 if sys.platform == "darwin" else rss / 2**10
    return {**post.diagnostics.to_dict(), "peak_mb": round(peak, 1)}


@app.command()
def bench(
    event: Annotated[
        str, typer.Argument(help="Held-out race whose window is sampled")
    ] = "2024-R03",
    samplers: Annotated[
        str, typer.Option(help="Comma-separated samplers")
    ] = "nutpie,numpyro,blackjax,nutpie-mclmc",
    window: Window = 24,
    draws: Draws = 1000,
    tune: Tune = 1000,
    chains: Annotated[str, typer.Option(help="Chain counts to try, e.g. 4,8")] = "4",
    evolution: Evolution = True,
) -> None:
    """The sampler bake-off (plan §A.5): ESS/s on the car-pace quantities, convergence and
    peak memory per sampler and chain count, each run in a fresh process."""
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    rows = []
    for n_chains in (int(c) for c in chains.split(",")):
        for s in (x.strip() for x in samplers.split(",")):
            cfg = _config(s, draws, tune, n_chains, "diag", 7)
            console.print(f"{s} x{n_chains} …")
            with ProcessPoolExecutor(1, mp_context=get_context("spawn")) as pool:
                try:
                    r = pool.submit(_bench_one, event, window, cfg, evolution).result()
                except Exception as exc:  # a sampler that fails is a result too
                    r = {"error": f"{type(exc).__name__}: {exc}"[:200]}
            rows.append({"sampler": s, "chains": n_chains, **r})
    table = Table(title=f"lap_v2 sampler bake-off — window before {event}")
    cols = (
        "sampler",
        "chains",
        "ess_per_s",
        "min_ess_bulk",
        "max_rhat",
        "divergences",
        "wall_s",
        "compile_s",
        "peak_mb",
        "ok",
    )
    for c in cols:
        table.add_column(c)
    for r in rows:
        if "error" in r:
            table.add_row(r["sampler"], str(r["chains"]), f"[red]{r['error']}[/red]")
            continue
        table.add_row(*(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols))
    console.print(table)
    out = get_settings().paths.artifacts / "lap_v2" / "bench"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{event}-{_tag(rows)}.json"
    path.write_text(json.dumps(rows, indent=1, default=str))
    console.print(f"saved → {path}")


@app.command()
def sbc(
    event: Annotated[
        str, typer.Argument(help="Race whose preceding window is the template")
    ] = "2024-R03",
    window: Annotated[int, typer.Option(help="Races in the template window")] = 4,
    sims: Annotated[int, typer.Option(help="Simulated datasets (plan: ~200)")] = 200,
    draws: Draws = 500,
    tune: Tune = 500,
    chains: Chains = 2,
    evolution: Evolution = True,
) -> None:
    """Simulation-based calibration (plan §A.4) on the design of a real window: ranks of the
    true values among posterior draws must be uniform."""
    from f1p.models import lap_validate as lv
    from f1p.models.lap_v2 import V2Priors

    tables = _tables()
    w = lv.holdout_window(tables, event, window)
    if w is None:
        raise typer.Exit(code=1)
    cfg = _config("nutpie", draws, tune, chains, "diag", 0)
    res = lv.sbc(w, sims, cfg=cfg, priors=V2Priors(track_evolution=evolution))
    out = get_settings().paths.artifacts / "lap_v2" / "sbc"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{event}-w{window}-n{sims}"
    res.ranks.write_parquet(out / f"{stem}.parquet")
    (out / f"{stem}.json").write_text(json.dumps(res.p_values, indent=1))
    table = Table(title=f"SBC — {sims} simulations (chi-square p, uniform ranks)")
    table.add_column("parameter")
    table.add_column("p")
    for k, p in sorted(res.p_values.items(), key=lambda kv: kv[1]):
        table.add_row(k, f"{p:.3f}")
    console.print(table)
    console.print("[green]uniform[/green]" if res.ok() else "[red]NOT uniform[/red]")
    console.print(f"saved → {out / stem}")
