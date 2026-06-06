"""f1predict — industry-grade command line interface (Typer + Rich).

Subcommand groups: top-level verbs (train/predict/backtest/simulate/whatif/dashboard/
report/profile-memory) plus ``data`` and ``model`` groups. Every command renders rich
output by default and supports ``--format json`` for scripting.
"""
from __future__ import annotations

import json as _json
from pathlib import Path
from typing import Optional

import typer
import typer.core
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn

from ..config import load_config
from ..utils.logging import console
from . import render

ACCENT_RED = render.ACCENT

app = typer.Typer(
    add_completion=False,
    rich_markup_mode="rich",
    help="🏎️  [bold red]f1predict[/bold red] — F1 prediction platform: train, predict, "
         "simulate, what-if & an extravagant dashboard.",
)
data_app = typer.Typer(help="Data pipeline commands.")


class ModelGroup(typer.core.TyperGroup):
    """A Typer group where an unknown subcommand is treated as a model name to activate.

    This lets [bold]f1predict model ensemble_default[/bold] select a model directly while
    still supporting real subcommands like [bold]model leaderboard[/bold].
    """

    def resolve_command(self, ctx, args):
        # If the first token isn't an option or a known subcommand, treat it as a
        # model name and route to the hidden `_select` command.
        if args and not args[0].startswith("-") and self.get_command(ctx, args[0]) is None:
            select = self.get_command(ctx, "_select")
            if select is not None:
                return "_select", select, list(args)
        return super().resolve_command(ctx, args)


model_app = typer.Typer(
    cls=ModelGroup, invoke_without_command=True,
    help="List, inspect and switch trained models. Run `f1predict model` to see them all.",
)
app.add_typer(data_app, name="data")
app.add_typer(model_app, name="model")


@app.callback(invoke_without_command=True)
def _root(ctx: typer.Context):
    """🏎️  f1predict — run with no command to enter the interactive shell."""
    if ctx.invoked_subcommand is not None:
        return
    import sys
    # Only drop into the REPL when attached to a real terminal — pipes, CI and the
    # test runner get help text instead of a hung interactive prompt.
    if sys.stdin.isatty() and sys.stdout.isatty():
        from .shell import run_shell
        run_shell()
        raise typer.Exit()
    console.print(ctx.get_help())


def _parse_overrides(items: list[str] | None) -> dict:
    out = {}
    for it in items or []:
        if "=" in it:
            k, v = it.split("=", 1)
            try:
                v = _json.loads(v)
            except _json.JSONDecodeError:
                pass
            out[k.strip()] = v
    return out


def _cfg(config: Optional[str], override: Optional[list[str]]):
    return load_config(config, _parse_overrides(override))


def _refresh_seasons(cfg) -> list[int]:
    """Seasons worth re-fetching in a *smart* update: the in-progress year plus any season
    that previously fell back to synthetic data (so we retry it for real results)."""
    import datetime
    from ..data_pipeline import data_status

    start = int(cfg.get("data.start_season", 2014))
    end = int(cfg.get("data.end_season", 2026))
    this_year = datetime.date.today().year
    seasons: set[int] = set()
    # The current calendar year is "in progress" — its results change race to race.
    if start <= this_year <= end:
        seasons.add(this_year)
    # Anything that resolved to synthetic earlier deserves a real-data retry.
    info = data_status(cfg)
    for s in info.get("synthetic_seasons", []) or []:
        if start <= int(s) <= end:
            seasons.add(int(s))
    return sorted(seasons)


def _run_update(cfg, full: bool = False, offline: bool = False) -> dict:
    """Refresh datasets to the latest available and return a summary.

    *Smart* (default): re-fetch only the in-progress season (and any prior synthetic
    fallbacks), then refresh weather + schedule. *Full* (``full=True``): re-fetch every
    season. Regulations are static code, so there's nothing to download for them.
    """
    from ..data_pipeline import (build_master, load_master, enrich_weather, data_status,
                                  clear_caches, clear_schedule_cache)
    from ..whatif import clear_feature_cache
    from ..models import clear_model_cache

    before = data_status(cfg)
    before_rows = int(before.get("rows", 0))
    before_races = int(before.get("races", 0))

    targets = None if full else _refresh_seasons(cfg)
    scope = "all seasons" if full else (
        "seasons " + ", ".join(map(str, targets)) if targets else "nothing stale")

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  TimeElapsedColumn(), console=console, transient=True) as p:
        task = p.add_task(f"Refreshing results ({scope})…", total=None)
        df = build_master(cfg, source="auto", force_refresh=full, refresh_seasons=targets)
        p.update(task, description="Refreshing weather…")
        enrich_weather(df, cfg, online=not offline)

    # Drop every in-process cache so the very next command sees the fresh data.
    clear_caches(); clear_feature_cache(); clear_model_cache(); clear_schedule_cache()

    after = data_status(cfg)
    return {
        "scope": scope,
        "rows": int(after.get("rows", len(df))),
        "rows_added": int(after.get("rows", len(df))) - before_rows,
        "races": int(after.get("races", 0)),
        "races_added": int(after.get("races", 0)) - before_races,
        "seasons": after.get("seasons", []),
        "synthetic_seasons": after.get("synthetic_seasons", []),
    }


# --------------------------------------------------------------------------- train
@app.command()
def update(
    full: bool = typer.Option(
        False, "--all", "--full",
        help="Re-fetch [bold]every[/bold] season (slow; hammers the API). Default refreshes "
             "only the in-progress season + weather + schedule."),
    offline: bool = typer.Option(False, "--offline", help="Skip live weather (climatology only)."),
    config: Optional[str] = typer.Option(None),
    override: Optional[list[str]] = typer.Option(None, "--override", "-o"),
):
    """Update every dataset to the latest available — results, weather & schedule.

    By default this is [bold]smart[/bold]: it re-fetches the in-progress season (whose results
    change race to race) plus any season that previously fell back to synthetic data, then
    refreshes measured weather and the sprint-aware schedule. Pass [bold]--all[/bold] to
    re-download every season from scratch.

    [dim]Regulations & rules are encoded in the app and always current, so there's nothing to
    download for them.[/dim]
    """
    cfg = _cfg(config, override)
    master = cfg.path("paths.master_dataset")
    if not master.exists():
        render.banner("update · first build")
        console.print(Panel(
            "[bold yellow]No dataset yet.[/bold yellow] Building it for the first time "
            "(downloads & caches real F1 data)…",
            border_style="yellow", expand=False,
        ))
    render.banner("update · refreshing to latest")
    summary = _run_update(cfg, full=full, offline=offline)

    seasons = summary["seasons"]
    span = f"{seasons[0]}–{seasons[-1]}" if seasons else "—"
    console.print(Panel(
        f"[bold]{summary['rows']:,}[/bold] rows · [bold]{summary['races']:,}[/bold] races · "
        f"seasons [cyan]{span}[/cyan]\n"
        f"refreshed: [cyan]{summary['scope']}[/cyan]\n"
        f"new since last build: [green]+{summary['rows_added']:,}[/green] rows · "
        f"[green]+{summary['races_added']:,}[/green] races"
        + (f"\n[yellow]synthetic fallback still in:[/yellow] "
           f"{', '.join(map(str, summary['synthetic_seasons']))}"
           if summary["synthetic_seasons"] else ""),
        title="✓ up to date", border_style=ACCENT_RED, expand=False,
    ))
    console.print("[dim]Regulations & rules are built-in and current. "
                  "Retrain to fold new results into the models: [bold]f1predict train[/bold].[/dim]")


@app.command()
def train(
    config: Optional[str] = typer.Option(None, help="Path to a YAML config."),
    preset: Optional[str] = typer.Option(
        None, "--preset", "-p",
        help="Training preset: test|light|medium|deep|max (scales accuracy & time)."),
    model: Optional[str] = typer.Option(None, help="Family: ensemble|random_forest|extra_trees|lightgbm|histgb|gbr."),
    tune: bool = typer.Option(False, "--tune", help="Run Optuna hyperparameter search."),
    n_trials: int = typer.Option(40, help="Optuna trials when --tune (overridden by --preset)."),
    profile_memory: bool = typer.Option(False, "--profile-memory", help="Report peak memory."),
    compare_all: bool = typer.Option(False, "--compare-all", help="Benchmark all families."),
    session: str = typer.Option(
        "race", "--session", "-s",
        help="What to predict: [cyan]race[/cyan]|[cyan]qualifying[/cyan]|[cyan]sprint[/cyan]."),
    all_sessions: bool = typer.Option(
        False, "--all-sessions",
        help="Train race, qualifying AND sprint models in one go (~3x longer)."),
    experiment_name: str = typer.Option("default", help="Name logged to the experiment tracker."),
    override: Optional[list[str]] = typer.Option(None, "--override", "-o", help="key=value overrides."),
):
    """Train the model. Run with no [bold]--preset[/bold] to get a guided picker.

    Running [bold]f1predict train[/bold] checks your dataset is built, shows a beautiful
    overview of every preset (and how long each takes), then asks you to choose one.

    [bold]Presets[/bold] (each does real time-series-CV tuning on the full real dataset):
      • [cyan]test[/cyan]   ~5-10 min  · 2-model ensemble, 15 trials
      • [cyan]light[/cyan]  ~15 min    · 3-model ensemble, 30 trials
      • [cyan]medium[/cyan] ~40 min    · 4-model ensemble, 60 trials
      • [cyan]deep[/cyan]   ~1.5 hr    · 5-model ensemble, 120 trials
      • [cyan]max[/cyan]    ~3-5 hr    · 5-model ensemble, 250 trials
    """
    from ..models import train as do_train
    from ..models.train import PRESETS, preset_overview_rows, SESSIONS

    cfg = _cfg(config, override)

    session = (session or "race").lower()
    if session not in SESSIONS:
        console.print(f"[red]Unknown session[/red] '{session}'. Choose: {', '.join(SESSIONS)}.")
        raise typer.Exit(1)
    sessions = list(SESSIONS) if all_sessions else [session]

    # 1) Urge the user to have the dataset in place before spending time training.
    master = cfg.path("paths.master_dataset")
    if not master.exists():
        render.banner("training")
        console.print(Panel(
            "[bold yellow]No dataset found.[/bold yellow]\n\n"
            f"Expected master dataset at [cyan]{master}[/cyan].\n"
            "Build it first (downloads & caches real F1 data, 2014-2026):\n\n"
            "    [bold]f1predict data build[/bold]\n",
            title="Set up your data first", border_style="yellow", expand=False,
        ))
        raise typer.Exit(1)

    # 2) Force a preset choice. Presets do real time-series-CV tuning and are the only path
    #    to an accurate model — so if the user didn't pick one (and isn't running an explicit
    #    family / --compare-all benchmark), show the overview and make them choose.
    explicit_family = compare_all or model is not None
    if preset is None and not explicit_family:
        import sys
        # A preset is mandatory and chosen interactively. On a non-TTY (pipe / CI / a
        # script) there's no one to answer the prompt — refuse loudly instead of letting
        # Rich silently fall back to the default and kick off a multi-hour train.
        if not sys.stdin.isatty():
            console.print(
                "[yellow]A training preset is required.[/yellow] Re-run with one, e.g.\n"
                "    [bold]f1predict train --preset test[/bold]   [dim](test|light|medium|deep|max)[/dim]"
            )
            raise typer.Exit(1)
        render.banner("training · choose a preset")
        rows = preset_overview_rows(cfg)
        try:
            preset = render.pick_preset(rows)
        except (EOFError, KeyboardInterrupt):
            console.print(
                "\n[yellow]No preset chosen.[/yellow] Re-run with one, e.g. "
                "[bold]f1predict train --preset test[/bold]."
            )
            raise typer.Exit(1)

    if preset is not None and preset not in PRESETS:
        console.print(
            f"[red]Unknown preset[/red] '{preset}'. Choose one of: "
            f"{', '.join(PRESETS)}."
        )
        raise typer.Exit(1)

    label = f"preset={preset}" if preset else (model or "ensemble")
    sess_label = "all sessions" if all_sessions else session
    render.banner(f"training · {experiment_name} · {label} · {sess_label}")
    if preset:
        info = next(r for r in preset_overview_rows(cfg) if r["preset"] == preset)
        console.print(Panel(
            f"[bold]{preset}[/bold] · est. [cyan]{info['eta']}[/cyan] · "
            f"{info['ensemble']} · {info['trials']} trials · {info['cv_folds']}-fold CV"
            + (f" · [cyan]{len(sessions)}[/cyan] sessions" if len(sessions) > 1 else "") + "\n"
            "[dim]Grab a coffee — this runs real cross-validated tuning.[/dim]",
            border_style=ACCENT_RED, expand=False,
        ))

    results: list[tuple[str, dict]] = []
    for sess in sessions:
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                      BarColumn(),
                      TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                      TimeElapsedColumn(),
                      TextColumn("[dim]ETA[/dim]"), TimeRemainingColumn(),
                      console=console, transient=False) as p:
            task = p.add_task(f"Training {sess}…", total=1.0)
            try:
                res = do_train(cfg, family=model, preset=preset, do_tune=tune, n_trials=n_trials,
                               profile_memory=profile_memory, experiment_name=experiment_name,
                               compare_all=compare_all, session=sess,
                               progress=lambda f, l: p.update(task, completed=f,
                                                              description=f"[{sess}] {l}"))
            except ValueError as exc:
                # e.g. no sprint weekends in the data — skip rather than abort the whole run.
                p.stop()
                console.print(f"[yellow]Skipping {sess}:[/yellow] {exc}")
                continue
        results.append((sess, res))
        console.print(render.metrics_panel(res["metrics"],
                                            title=f"{sess.title()} · best: {res['family']}"))
        if compare_all:
            import pandas as pd
            bench = pd.DataFrame(res["benchmark"])[
                ["family", "mae", "winner_accuracy", "podium_accuracy", "peak_rss_mb", "train_seconds"]
            ]
            console.print(render.df_table(bench, title=f"Benchmark · {sess}"))
        console.print(f"[green]✓[/green] {sess} model saved -> {res['model_path']}")

    if not results:
        console.print("[red]No models trained.[/red]")
        raise typer.Exit(1)


# --------------------------------------------------------------------------- predict
@app.command()
def predict(
    race: Optional[str] = typer.Argument(
        None,
        help="Race reference, e.g. [cyan]abudhabi2026[/cyan], [cyan]suzuka26[/cyan], "
             "[cyan]japan27[/cyan], [cyan]madring26[/cyan], [cyan]monaco26[/cyan]."),
    session: str = typer.Argument(
        "race",
        help="Session to predict: [cyan]race[/cyan]|[cyan]qualifying[/cyan]|[cyan]sprint[/cyan] "
             "(default race). Sprint only works on sprint weekends."),
    season: Optional[int] = typer.Option(None, help="Season year (alternative to the race ref)."),
    round: Optional[int] = typer.Option(None, help="Round number (alternative to the race ref)."),
    upcoming: bool = typer.Option(False, "--upcoming", help="Predict the next/most-recent race."),
    force: bool = typer.Option(
        False, "--force",
        help="Skip the 'update data first?' prompt and predict on the current data."),
    format: str = typer.Option("rich", help="rich|json|table."),
    config: Optional[str] = typer.Option(None),
    override: Optional[list[str]] = typer.Option(None, "--override", "-o"),
):
    """Predict a race, qualifying or sprint result with win/podium/points probabilities.

    Accepts a flexible [bold]race reference[/bold] — country, city, circuit or GP name plus a
    2- or 4-digit year: [cyan]abudhabi2026[/cyan], [cyan]uae2027[/cyan], [cyan]suzuka26[/cyan],
    [cyan]japan27[/cyan], [cyan]madring26[/cyan], [cyan]spain26[/cyan], [cyan]vegas26[/cyan].
    Add a [bold]session[/bold] to predict qualifying or the sprint:
    [cyan]f1predict predict suzuka26 qualifying[/cyan], [cyan]f1predict predict miami26 sprint[/cyan].

    By default you're offered a quick data refresh first (so predictions use the latest
    results); pass [bold]--force[/bold] to skip that and predict immediately. Future races
    (no results yet) are predicted from the latest form & ELO snapshot.
    """
    import sys
    from ..models import load_production, predict_future_race
    from ..models.train import SESSIONS
    from ..data_pipeline import load_master, parse_racref, weekend_has_sprint, sprint_rounds
    from ..whatif import get_feature_frame, get_race, upcoming_race

    cfg = _cfg(config, override)

    session = (session or "race").lower()
    if session not in SESSIONS:
        console.print(f"[red]Unknown session[/red] '{session}'. Choose: {', '.join(SESSIONS)}.")
        raise typer.Exit(1)

    # Offer to refresh data first — unless --force, or there's no TTY to answer the prompt.
    if not force and sys.stdin.isatty():
        if typer.confirm("Update data to the latest first?", default=False):
            render.banner("predict · refreshing data")
            _run_update(cfg, full=False, offline=False)

    master = load_master(cfg)
    circuit_id = None
    label = None

    if race:
        try:
            rr = parse_racref(race, cfg)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1)
        season, round, circuit_id, label = rr.season, rr.round, rr.circuit_id, rr.label

    feat_df, _ = get_feature_frame(master, cfg)
    if upcoming or (season is None and race is None):
        season, round = upcoming_race(feat_df)

    # Sprint is only valid on sprint weekends — refuse early with a helpful list.
    if session == "sprint" and not weekend_has_sprint(season, round, circuit_id, cfg, master):
        rounds = sprint_rounds(season, cfg, master)
        if rounds:
            listing = ", ".join(
                (f"R{r['round']} {r['circuit_id']}" if r["round"] is not None else r["circuit_id"])
                for r in rounds)
            hint = f"\n{season} sprint weekends: [cyan]{listing}[/cyan]"
        else:
            hint = f"\nNo sprint weekends are known for {season}."
        console.print(Panel(
            f"[yellow]{label or circuit_id or season} is not a sprint weekend.[/yellow]"
            f"{hint}\n\n[dim]Run a race or qualifying prediction instead, e.g. "
            f"[bold]f1predict predict {race or 'suzuka26'}[/bold].[/dim]",
            title="No sprint here", border_style="yellow", expand=False,
        ))
        raise typer.Exit(1)

    pred = load_production(cfg, session=session)

    sess_tag = {"race": "Race", "qualifying": "Qualifying", "sprint": "Sprint"}[session]

    # Does this race already have real results in the master?
    have_results = False
    if round is not None:
        existing = master[(master["season"] == season) & (master["round"] == round)]
        pos_col = {"race": "position", "qualifying": "quali_position",
                   "sprint": "sprint_position"}[session]
        have_results = len(existing) > 0 and (existing[pos_col] > 0).any()

    if have_results:
        race_df = get_race(feat_df, season, round)
        tbl = pred.predict_race(race_df, season=season, session=session)
        title = f"{sess_tag} · {race_df['race_name'].iloc[0]} {season} · R{round}"
    else:
        if circuit_id is None:
            console.print("[red]Future race needs a race reference (e.g. abudhabi2026).[/red]")
            raise typer.Exit(1)
        tbl = predict_future_race(cfg, pred, season, round, circuit_id, master, session=session)
        title = (f"{sess_tag} · {label or circuit_id} {season}  "
                 f"[yellow](projected — no results yet)[/yellow]")

    if format == "json":
        cols = ["driver_id", "driver_code", "driver_name", "constructor_name", "predicted_rank",
                "pred_position", "p_win", "p_podium", "p_points", "expected_points"]
        cols = [c for c in cols if c in tbl.columns]
        console.print_json(data=tbl[cols].to_dict("records"))
        return
    render.banner(title)
    result_label = {"race": "Predicted Result", "qualifying": "Predicted Grid",
                    "sprint": "Predicted Sprint Result"}[session]
    console.print(render.race_table(tbl, title=result_label))


# --------------------------------------------------------------------------- backtest
@app.command()
def backtest(
    config: Optional[str] = typer.Option(None),
    model: Optional[str] = typer.Option(None, help="Model family."),
    min_races: int = typer.Option(2, help="Minimum training seasons before evaluating."),
    export: Optional[str] = typer.Option(None, help="Write per-race CSV to this path."),
    format: str = typer.Option("rich", help="rich|json."),
    override: Optional[list[str]] = typer.Option(None, "--override", "-o"),
):
    """Run a walk-forward backtest across eras."""
    from ..evaluation import backtest as do_backtest

    render.banner("backtest · walk-forward")
    cfg = _cfg(config, override)
    family = model or cfg.get("model.family", "random_forest")
    heavy = family == "ensemble"
    console.print(Panel(
        f"family [bold]{family}[/bold] · expanding-window, retrained once per evaluated season\n"
        + ("[yellow]⚠ the full ensemble retrains a 5-model stack every season — this can take "
           "several minutes.[/yellow]\n[dim]Pass [bold]--model random_forest[/bold] for a ~15 s "
           "backtest.[/dim]" if heavy else
           "[dim]Fast single-family backtest.[/dim]"),
        border_style=ACCENT_RED, expand=False,
    ))
    with Progress(SpinnerColumn(), BarColumn(), TextColumn("{task.description}"),
                  TimeElapsedColumn(), console=console, transient=False) as p:
        task = p.add_task("Backtesting…", total=1.0)
        res = do_backtest(cfg, family=model, min_train_seasons=min_races,
                          progress=lambda f, l: p.update(task, completed=f, description=l))

    if format == "json":
        console.print_json(data={"summary": res["summary"], "era_summary": res["era_summary"],
                                 "baseline": res["baseline"]})
        return
    console.print(render.metrics_panel(res["summary"], title="Overall"))
    console.print(render.metrics_panel(res["baseline"], title="Grid-only baseline"))
    if res["era_summary"]:
        import pandas as pd
        era = pd.DataFrame(res["era_summary"]).T.reset_index(names="era")
        console.print(render.df_table(era, title="By era"))
    if export:
        res["per_race"].to_csv(export, index=False)
        console.print(f"[green]✓[/green] per-race results -> {export}")


# --------------------------------------------------------------------------- simulate
@app.command()
def simulate(
    race: Optional[str] = typer.Argument(None, help="Race reference, e.g. abudhabi2026."),
    season: Optional[int] = typer.Option(None),
    round: Optional[int] = typer.Option(None),
    n_simulations: int = typer.Option(10000, "--n-simulations", help="Number of simulations."),
    weather: str = typer.Option("dry", help="dry|rain."),
    output: Optional[str] = typer.Option(None, help="Write full distribution to JSON/CSV."),
    config: Optional[str] = typer.Option(None),
    override: Optional[list[str]] = typer.Option(None, "--override", "-o"),
):
    """Monte Carlo simulate a race (winner/podium/points distributions)."""
    from ..models import load_production, predict_future_race
    from ..simulation import simulate_race
    from ..data_pipeline import load_master, parse_racref
    from ..whatif import get_feature_frame, get_race, upcoming_race

    cfg = _cfg(config, override)
    master = load_master(cfg)
    feat_df, _ = get_feature_frame(master, cfg)
    circuit_id = None
    if race:
        rr = parse_racref(race, cfg)
        season, round, circuit_id = rr.season, rr.round, rr.circuit_id
    if season is None or round is None and circuit_id is None:
        season, round = upcoming_race(feat_df)

    pred = load_production(cfg)
    have = False
    if round is not None:
        ex = master[(master["season"] == season) & (master["round"] == round)]
        have = len(ex) > 0 and (ex["position"] > 0).any()
    if have:
        race_df = get_race(feat_df, season, round)
    else:
        from ..models.future import build_future_race
        from ..features import build_features
        combined, round = build_future_race(cfg, season, round, circuit_id or "", master)
        fdf, _ = build_features(combined, cfg)
        race_df = fdf[(fdf["season"] == season) & (fdf["round"] == round)].reset_index(drop=True)
    mu = pred.predict_positions(race_df)

    from ..simulation import empirical_dnf_rates
    from ..features import era_summary
    dnf_prob = empirical_dnf_rates(race_df, master, int(season), cfg=cfg)
    es = era_summary(int(season))

    render.banner(f"Monte Carlo · {n_simulations:,} sims · {weather} · {es['label']}")
    with Progress(SpinnerColumn(), TextColumn("Simulating…"), console=console, transient=True) as p:
        p.add_task("sim", total=None)
        sim = simulate_race(race_df, mu, pred.residual_std, n_sim=n_simulations,
                            season=season, weather=weather, dnf_prob=dnf_prob, cfg=cfg)
    frame = sim.to_frame()
    console.print(render.df_table(
        frame[["driver_name", "p_win", "p_podium", "p_points", "p_dnf", "exp_points"]],
        title=f"{race_df['race_name'].iloc[0]} {season}"))
    mods = sim.meta.get("modifiers", {})
    console.print(
        f"[dim]Regulation modifiers — chaos ×{mods.get('chaos', 1):.2f} · "
        f"reliability ×{mods.get('reliability', 1):.2f} · "
        f"overtake-difficulty {mods.get('overtake_difficulty', 0):.2f} · "
        f"safety-car {mods.get('safety_car_rate', 0):.0%}[/dim]")
    console.print(
        f"[dim]Track & weather — rain {sim.meta.get('rain_prob', 0):.0%} · "
        f"tyre-stress {sim.meta.get('tyre_stress', 0):.2f} · "
        f"degradation {sim.meta.get('track_degradation', 0):.0%}[/dim]")
    if output:
        if output.endswith(".csv"):
            frame.to_csv(output, index=False)
        else:
            Path(output).write_text(_json.dumps(frame.to_dict("records"), indent=2))
        console.print(f"[green]✓[/green] distribution -> {output}")


# --------------------------------------------------------------------------- whatif
@app.command()
def whatif(
    season: Optional[int] = typer.Option(None),
    round: Optional[int] = typer.Option(None),
    driver: Optional[str] = typer.Option(None, help="Driver id or name to perturb."),
    grid: Optional[int] = typer.Option(None, help="Pin driver to this grid slot."),
    weather: str = typer.Option("dry", help="dry|rain."),
    recent_form_boost: float = typer.Option(0.0, "--recent-form-boost",
                                            help="Form delta for driver (positive=faster)."),
    config: Optional[str] = typer.Option(None),
    override: Optional[list[str]] = typer.Option(None, "--override", "-o"),
):
    """Run a what-if scenario and show the impact deltas."""
    from ..models import load_production
    from ..whatif import (Scenario, apply_scenario, compute_standings,
                          get_feature_frame, get_race, upcoming_race)
    from ..data_pipeline import load_master

    cfg = _cfg(config, override)
    df = load_master(cfg)
    feat_df, _ = get_feature_frame(df, cfg)
    if season is None or round is None:
        season, round = upcoming_race(feat_df)
    race = get_race(feat_df, season, round)
    pred = load_production(cfg)

    did = None
    if driver:
        match = race[(race["driver_id"] == driver) | (race["driver_name"] == driver)]
        if not match.empty:
            did = match["driver_id"].iloc[0]
    sc = Scenario(name="cli", weather=weather)
    if did:
        if grid:
            sc.grid_override[did] = grid
        if recent_form_boost:
            sc.form_boost[did] = -recent_form_boost  # positive boost => lower (better) position

    st = compute_standings(df, season, round)
    standings = dict(zip(st["driver_id"], st["points"]))
    res = apply_scenario(pred, race, sc, season=season, standings=standings, cfg=cfg)

    render.banner(f"What-if · {sc.describe()}")
    console.print(render.df_table(
        res["delta"][["driver_name", "p_win", "p_win_delta", "p_podium_delta",
                      "exp_points", "exp_points_delta"]],
        title="Scenario impact (Δ vs baseline)"))


# --------------------------------------------------------------------------- dashboard
@app.command()
def dashboard(
    port: int = typer.Option(8501, help="Port for the Streamlit app."),
):
    """Launch the extravagant Streamlit dashboard."""
    import subprocess
    import sys
    from ..config import PROJECT_ROOT

    app_path = PROJECT_ROOT / "src" / "dashboard" / "app.py"
    render.banner(f"launching dashboard on :{port}")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path),
                    "--server.port", str(port)])


# --------------------------------------------------------------------------- report
@app.command()
def report(
    season: Optional[int] = typer.Option(None),
    round: Optional[int] = typer.Option(None),
    output: str = typer.Option("reports/race_report.html", help="Output HTML path."),
    config: Optional[str] = typer.Option(None),
    override: Optional[list[str]] = typer.Option(None, "--override", "-o"),
):
    """Generate a race report (HTML)."""
    from ..reporting import build_race_report

    cfg = _cfg(config, override)
    render.banner("report")
    path = build_race_report(cfg, season, round, output)
    console.print(f"[green]✓[/green] report -> {path}")


# --------------------------------------------------------------------------- profile-memory
@app.command(name="profile-memory")
def profile_memory(
    command: str = typer.Option("train", help="Command to profile (currently: train)."),
    config: Optional[str] = typer.Option(None),
):
    """Profile peak memory & time of a heavy command."""
    from ..models import train as do_train
    from ..utils.profiling import profile as prof_ctx

    render.banner(f"profiling · {command}")
    cfg = _cfg(config, None)
    with prof_ctx(command) as pr:
        if command.startswith("train"):
            do_train(cfg, experiment_name="profile")
    console.print(render.metrics_panel(
        {"mae": pr.peak_rss_mb}, title="profile"))  # reuse panel
    console.print(f"peak RSS: [bold]{pr.peak_rss_mb:.0f} MB[/bold] · {pr.seconds:.1f}s")


# --------------------------------------------------------------------------- weather
@app.command()
def weather(
    race: Optional[str] = typer.Argument(None, help="Race ref, e.g. [cyan]suzuka26[/cyan]."),
    season: Optional[int] = typer.Option(None),
    round: Optional[int] = typer.Option(None),
    offline: bool = typer.Option(False, "--offline", help="Use climatology only (no API)."),
    config: Optional[str] = typer.Option(None),
    override: Optional[list[str]] = typer.Option(None, "--override", "-o"),
):
    """Show weather & track-environment for a race (Open-Meteo API + circuit climatology).

    Past races pull measured conditions from the free Open-Meteo archive (cached locally);
    future races fall back to the circuit's researched race-month climatology. Also prints
    the static track-physics stats (degradation, abrasion, length, downforce, altitude).
    """
    from rich.table import Table
    from ..data_pipeline import load_master, parse_racref, get_race_weather
    from ..features.tracks import track_env, track_stats

    cfg = _cfg(config, override)
    master = load_master(cfg)
    circuit_id, date, name = None, None, None
    if race:
        rr = parse_racref(race, cfg)
        season, round, circuit_id = rr.season, rr.round, rr.circuit_id
    if circuit_id is None and season is not None and round is not None:
        row = master[(master["season"] == season) & (master["round"] == round)]
        if len(row):
            circuit_id = str(row["circuit_id"].iloc[0])
            date = row["date"].iloc[0]
            name = str(row["race_name"].iloc[0])
    if circuit_id is None:
        console.print("[red]Could not resolve a race.[/red] Try e.g. "
                      "[bold]f1predict weather suzuka26[/bold].")
        raise typer.Exit(1)
    if date is None:
        hit = master[(master["circuit_id"] == circuit_id)]
        date = hit["date"].iloc[-1] if len(hit) else None
        name = name or (str(hit["race_name"].iloc[-1]) if len(hit) else circuit_id)

    render.banner(f"weather · {name or circuit_id} {season or ''}".strip())
    wx = get_race_weather(circuit_id, date, cfg, online=not offline)
    env = track_env(circuit_id)

    wt = Table(title="Race-day weather", header_style=f"bold {ACCENT_RED}", expand=False)
    wt.add_column("Metric"); wt.add_column("Value", justify="right")
    src = wx.get("wx_source", "climatology")
    temp = wx.get("wx_temp_c"); hum = wx.get("wx_humidity")
    rain = wx.get("wx_rain_prob"); wind = wx.get("wx_wind_kmh")
    wt.add_row("Air temperature", f"{temp:.1f} °C" if temp is not None else "—")
    wt.add_row("Relative humidity", f"{hum:.0%}" if hum is not None else "—")
    wt.add_row("Rain probability", f"{rain:.0%}" if rain is not None else "—")
    wt.add_row("Max wind", f"{wind:.0f} km/h" if wind is not None else "—")
    wt.add_row("Source", "Open-Meteo (measured)" if src == "open-meteo" else "Climatology (prior)")
    console.print(wt)

    st = track_stats(circuit_id)
    tt = Table(title="Track environment", header_style=f"bold {ACCENT_RED}", expand=False)
    tt.add_column("Stat"); tt.add_column("Value", justify="right")
    tt.add_row("Circuit length", f"{env['lap_km']:.3f} km")
    tt.add_row("Tyre degradation", f"{st['trk_degradation']:.0%}")
    tt.add_row("Surface abrasion", f"{st['trk_abrasion']:.0%}")
    tt.add_row("Downforce demand", f"{st['trk_downforce']:.0%}")
    tt.add_row("Altitude", f"{env['altitude_m']:.0f} m")
    tt.add_row("Coordinates", f"{env['lat']:.3f}, {env['lon']:.3f}")
    console.print(tt)


# --------------------------------------------------------------------------- regulations
@app.command()
def regulations(
    season: int = typer.Argument(2026, help="Season to describe, e.g. [cyan]2026[/cyan]."),
):
    """Show the regulation era, key rule changes, and the simulation modifiers in effect.

    F1's rule-sets reshape the competitive order — this surfaces exactly how f1predict
    adapts: which era a season belongs to, how much constructor history it trusts, and the
    chaos / reliability / overtaking modifiers the Monte-Carlo engine applies.
    """
    from ..features import era_summary
    from rich.table import Table

    es = era_summary(season)
    render.banner(f"regulations · {season}")
    reset = "[bold yellow]RESET YEAR[/bold yellow]" if es["is_reset_year"] else "stable rule-set"
    console.print(Panel(
        f"[bold]{es['label']}[/bold]\n"
        f"era id [cyan]{es['era']}[/cyan] · {reset} · "
        f"{es['seasons_since_reset']} season(s) since reset\n"
        f"constructor-history trust [bold]{es['constructor_trust']:.0%}[/bold] "
        "[dim](low early in a reset → model leans on driver skill)[/dim]",
        border_style=ACCENT_RED, expand=False,
    ))
    changes = Table(title="Key rule changes", header_style=f"bold {ACCENT_RED}", expand=False)
    changes.add_column("•", style=ACCENT_RED)
    changes.add_column("Change")
    for c in es["changes"]:
        changes.add_row("•", c)
    console.print(changes)

    mt = Table(title="Simulation modifiers", header_style=f"bold {ACCENT_RED}", expand=False)
    mt.add_column("Modifier")
    mt.add_column("Value", justify="right")
    mt.add_column("Effect", style="dim")
    effects = {
        "chaos": "race-order variance (higher = more upsets)",
        "reliability": "DNF-rate multiplier (higher = more retirements)",
        "overtake_difficulty": "0..1 — higher means the grid order sticks",
        "safety_car_rate": "P(order-bunching safety car) per race",
        "tire_wear_var": "strategy-driven variability",
    }
    for k, eff in effects.items():
        v = es["modifiers"].get(k, 0.0)
        disp = f"{v:.0%}" if k == "safety_car_rate" else f"×{v:.2f}" if k != "overtake_difficulty" else f"{v:.2f}"
        mt.add_row(k, disp, eff)
    console.print(mt)


# --------------------------------------------------------------------------- explain
@app.command()
def explain(
    top: int = typer.Option(20, help="How many top features to show."),
    config: Optional[str] = typer.Option(None),
):
    """Explain the active model: which features drive its predictions (offline, no SHAP).

    Aggregates impurity-based importances across the ensemble's tree members and groups them
    into intuitive families (driver form, ELO, track, regulations, tyres, interactions) so
    you can see, in particular, how much the model leans on regulation-aware signals.
    """
    import numpy as np
    from rich.table import Table
    from ..models import load_production

    cfg = _cfg(config, None)
    pred = load_production(cfg)
    cols = pred.feature_cols
    model = pred.model

    members = []
    if hasattr(model, "named_estimators_"):
        members = [e for e in model.named_estimators_.values()
                   if hasattr(e, "feature_importances_")]
    elif hasattr(model, "feature_importances_"):
        members = [model]
    if not members:
        console.print("[yellow]This model exposes no tree importances to explain.[/yellow]")
        raise typer.Exit(0)

    imp = np.zeros(len(cols))
    for est in members:
        fi = np.asarray(est.feature_importances_, dtype=float)
        if fi.sum() > 0:
            imp += fi / fi.sum()
    imp /= max(len(members), 1)

    render.banner(f"explain · {pred.family} · {len(members)} tree member(s)")

    def family(name: str) -> str:
        if "_x_" in name or name.endswith(("_sq", "_log")):
            return "Interactions"
        if name.startswith("elo"):
            return "ELO skill"
        if name.startswith("reg"):
            return "Regulations"
        if name.startswith("trk"):
            return "Track"
        if name.startswith(("f_drv", "f_con")):
            return "Form"
        if name.startswith(("f_grid", "f_pole", "f_quali", "f_front", "f_top", "f_pit")):
            return "Grid / Quali"
        return "Other"

    groups: dict[str, float] = {}
    for name, v in zip(cols, imp):
        groups[family(name)] = groups.get(family(name), 0.0) + v

    gt = Table(title="Importance by feature family", header_style=f"bold {ACCENT_RED}", expand=False)
    gt.add_column("Family")
    gt.add_column("Share", justify="right")
    gt.add_column("", width=24)
    for fam, v in sorted(groups.items(), key=lambda kv: kv[1], reverse=True):
        bar = "█" * int(round(v * 24)) + "·" * (24 - int(round(v * 24)))
        gt.add_row(fam, f"{v:.1%}", f"[{ACCENT_RED}]{bar}[/{ACCENT_RED}]")
    console.print(gt)

    order = np.argsort(imp)[::-1][:top]
    ft = Table(title=f"Top {top} features", header_style=f"bold {ACCENT_RED}", expand=False)
    ft.add_column("#", justify="right")
    ft.add_column("Feature")
    ft.add_column("Family", style="dim")
    ft.add_column("Importance", justify="right")
    for i, idx in enumerate(order, start=1):
        ft.add_row(str(i), cols[idx], family(cols[idx]), f"{imp[idx]:.3f}")
    console.print(ft)


# --------------------------------------------------------------------------- data group
@data_app.command("status")
def data_status_cmd(config: Optional[str] = typer.Option(None)):
    """Show data freshness, coverage and cold-start info."""
    from ..data_pipeline import data_status

    cfg = _cfg(config, None)
    info = data_status(cfg)
    render.banner("data status")
    if not info.get("exists"):
        console.print("[yellow]No master dataset. Run [bold]f1predict data build[/bold].[/yellow]")
        return
    console.print_json(data=info)


@data_app.command("build")
def data_build_cmd(
    source: str = typer.Option("auto", help="auto|jolpica|synthetic."),
    force_refresh: bool = typer.Option(
        False, "--force-refresh",
        help="Ignore the per-season cache and re-fetch every season from the API."),
    config: Optional[str] = typer.Option(None),
    override: Optional[list[str]] = typer.Option(None, "--override", "-o"),
):
    """Build the master dataset (jolpica with synthetic fallback).

    [dim]Tip: to just pull the latest results after a race, use [bold]f1predict update[/bold]
    — it refreshes only the in-progress season plus weather.[/dim]
    """
    from ..data_pipeline import build_master

    cfg = _cfg(config, override)
    render.banner(f"building master · source={source}"
                  + (" · force-refresh" if force_refresh else ""))
    with Progress(SpinnerColumn(), TextColumn("Ingesting…"), TimeElapsedColumn(),
                  console=console, transient=True) as p:
        p.add_task("build", total=None)
        df = build_master(cfg, source=source, force_refresh=force_refresh)
    console.print(f"[green]✓[/green] {len(df):,} rows across "
                  f"{df['season'].nunique()} seasons.")


@data_app.command("weather")
def data_weather_cmd(
    seasons: Optional[str] = typer.Option(None, help="Season range, e.g. [cyan]2018-2026[/cyan]."),
    offline: bool = typer.Option(False, "--offline", help="Climatology only (no API calls)."),
    config: Optional[str] = typer.Option(None),
    override: Optional[list[str]] = typer.Option(None, "--override", "-o"),
):
    """Enrich the master dataset with measured Open-Meteo weather (cached locally).

    Walks every race in the master, fetching race-day conditions from the free Open-Meteo
    archive (no API key) and caching them to [cyan]data/cache/weather.parquet[/cyan]. Races
    the API can't cover fall back to circuit climatology, so this is safe to run offline.
    """
    from ..data_pipeline import load_master, enrich_weather

    cfg = _cfg(config, override)
    master = load_master(cfg)
    if seasons:
        lo, _, hi = seasons.partition("-")
        lo, hi = int(lo), int(hi or lo)
        mask = master["season"].between(lo, hi)
    else:
        mask = master["season"].notna()
    sub = master[mask]
    n_races = sub[["season", "round"]].drop_duplicates().shape[0]
    render.banner(f"weather enrich · {n_races:,} races")
    with Progress(SpinnerColumn(), TextColumn("Fetching weather…"),
                  TimeElapsedColumn(), console=console, transient=True) as p:
        task = p.add_task("wx", total=1.0)
        enriched = enrich_weather(sub, cfg, online=not offline,
                                  progress=lambda frac, msg: p.update(task, completed=frac))
    measured_rows = (enriched.get("wx_source") == "open-meteo") if "wx_source" in enriched else None
    measured_races = (enriched[measured_rows][["season", "round"]].drop_duplicates().shape[0]
                      if measured_rows is not None else 0)
    console.print(f"[green]✓[/green] enriched [bold]{n_races:,}[/bold] races · "
                  f"[bold]{measured_races:,}[/bold] with measured weather, rest climatology.")
    console.print("[dim]Cached to data/cache/weather.parquet — reused by predict / simulate.[/dim]")


@model_app.callback()
def model_main(ctx: typer.Context, config: Optional[str] = typer.Option(None)):
    """List trained models. Run [bold]f1predict model <name>[/bold] to switch the active one."""
    if ctx.invoked_subcommand is not None:
        return
    from ..models import list_models

    cfg = _cfg(config, None)
    models = list_models(cfg)
    render.banner("models")
    if not models:
        console.print(Panel(
            "[yellow]No trained models yet.[/yellow]\n\n"
            "Train one to get started:\n\n    [bold]f1predict train[/bold]",
            title="Nothing here yet", border_style="yellow", expand=False,
        ))
        raise typer.Exit(0)

    console.print(render.models_table(models))
    active = next((m for m in models if m["is_production"]), None)
    tip = Text.assemble(
        ("Tip  ", "bold white on bright_red"),
        ("  switch the active model with  ", "dim"),
        ("f1predict model <name>", "bold cyan"),
        ("   e.g.  ", "dim"),
        (f"f1predict model {models[0]['name']}", "cyan"),
    )
    console.print(tip)
    if active:
        console.print(f"[dim]Active model:[/dim] [bold green]{active['name']}[/bold green] "
                      "[dim](● used by predict / simulate / whatif)[/dim]")
    console.print("[dim]Delete a model with [bold]f1predict model delete <name>[/bold].[/dim]")


@model_app.command("_select", hidden=True)
def model_select(
    name: str = typer.Argument(..., help="Model name (easy/partial match allowed)."),
    config: Optional[str] = typer.Option(None),
):
    """Activate a trained model by (partial) name — used by `f1predict model <name>`."""
    from ..models import set_production

    cfg = _cfg(config, None)
    render.banner("model · switch")
    try:
        rec = set_production(name, cfg)
    except ValueError as e:
        console.print(f"[red]✗[/red] {e}")
        console.print("[dim]Run [bold]f1predict model[/bold] to see all models.[/dim]")
        raise typer.Exit(1)
    met = rec.get("metrics", {})
    wa = met.get("winner_accuracy")
    summary = (
        f"[bold green]✓ now active:[/bold green] [bold]{rec['name']}[/bold]\n"
        f"[dim]family[/dim] {rec['family']}  ·  [dim]preset[/dim] {rec['preset']}"
    )
    if isinstance(wa, (int, float)):
        summary += f"  ·  [dim]winner acc[/dim] {wa:.0%}"
    summary += "\n[dim]predict / simulate / whatif will now use this model.[/dim]"
    console.print(Panel(summary, border_style=ACCENT_RED, expand=False))


@model_app.command("use")
def model_use(
    name: str = typer.Argument(..., help="Model name (easy/partial match allowed)."),
    config: Optional[str] = typer.Option(None),
):
    """Activate a trained model (explicit alias for `f1predict model <name>`)."""
    model_select(name=name, config=config)


@model_app.command("delete")
def model_delete(
    names: list[str] = typer.Argument(
        ..., help="Model name(s) to delete (easy/partial match), or "
                  "[bold]clear[/bold] / [bold]all[/bold] to delete every model."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
    config: Optional[str] = typer.Option(None),
):
    """Delete one or more trained models from the store.

    Removes each model's [cyan].joblib[/cyan] + [cyan].card.json[/cyan]. If you delete the
    active model, the newest remaining model is promoted automatically. Asks for confirmation
    unless [bold]--yes[/bold] is given.

    Use [bold]f1predict model delete clear[/bold] (or [bold]all[/bold]) to wipe every trained
    model in one go.
    """
    from ..models import resolve_model, delete_model, list_models
    from rich.table import Table

    cfg = _cfg(config, None)
    render.banner("model · delete")

    all_models = list_models(cfg)
    clear_all = any(n.strip().lower() in ("clear", "all", "*", "everything") for n in names)

    targets, seen = [], set()
    if clear_all:
        if not all_models:
            console.print("[yellow]No trained models to delete.[/yellow]")
            raise typer.Exit(0)
        targets = list(all_models)
    else:
        for q in names:
            try:
                rec = resolve_model(q, cfg)
            except ValueError as e:
                console.print(f"[red]✗[/red] {e}")
                raise typer.Exit(1)
            if rec["file"] in seen:
                continue
            seen.add(rec["file"])
            targets.append(rec)

    tbl = Table(header_style=f"bold {ACCENT_RED}", expand=False)
    tbl.add_column("Model"); tbl.add_column("Family"); tbl.add_column("Size", justify="right")
    tbl.add_column("Active", justify="center")
    for r in targets:
        tbl.add_row(r["name"], r["family"], f"{r['size_mb']:.1f} MB",
                    "[green]●[/green]" if r["is_production"] else "")
    console.print(tbl)

    wipes_all = len(all_models) == len(targets)
    if wipes_all:
        console.print("[bold red]⚠ This deletes EVERY trained model in the store.[/bold red] "
                      "[dim]You'll need to [bold]f1predict train[/bold] again before predicting.[/dim]")
    if any(r["is_production"] for r in targets) and not wipes_all:
        console.print("[yellow]⚠ One target is the active model — the newest remaining "
                      "model will be promoted automatically.[/yellow]")

    if not yes:
        if clear_all or wipes_all:
            label = f"ALL {len(targets)} models"
        elif len(targets) == 1:
            label = targets[0]["name"]
        else:
            label = f"these {len(targets)} models"
        if not typer.confirm(f"Permanently delete {label}?"):
            console.print("[dim]Aborted — nothing deleted.[/dim]")
            raise typer.Exit(0)

    new_prod = None
    for r in targets:
        res = delete_model(r["name"], cfg)
        console.print(f"[green]✓[/green] deleted [bold]{r['name']}[/bold] "
                      f"[dim]({', '.join(res['removed']) or 'no files'})[/dim]")
        if res["new_production"]:
            new_prod = res["new_production"]

    remaining = list_models(cfg)
    if new_prod and remaining:
        console.print(f"[dim]Active model is now[/dim] [bold green]{new_prod}[/bold green].")
    elif not remaining:
        console.print("[yellow]No models left — train one with "
                      "[bold]f1predict train[/bold].[/yellow]")


@model_app.command("leaderboard")
def model_leaderboard(config: Optional[str] = typer.Option(None)):
    """Show all logged experiments with metrics & memory footprint."""
    from ..models import load_experiments
    import pandas as pd

    cfg = _cfg(config, None)
    exps = load_experiments(cfg)
    render.banner("experiment leaderboard")
    if not exps:
        console.print("[yellow]No experiments yet. Run [bold]f1predict train[/bold].[/yellow]")
        return
    rows = []
    for e in exps:
        m = e.get("metrics", {})
        rows.append({
            "time": e.get("timestamp", ""), "name": e.get("experiment_name", ""),
            "family": e.get("family", ""), "mae": m.get("mae", float("nan")),
            "winner_acc": m.get("winner_accuracy", float("nan")),
            "podium_acc": m.get("podium_accuracy", float("nan")),
            "commit": e.get("git_commit", ""),
        })
    df = pd.DataFrame(rows).sort_values("mae")
    console.print(render.df_table(df, title="Experiments (best MAE first)"))


@app.command()
def setup(
    build_data: bool = typer.Option(False, "--build-data", help="Also build the real master dataset now."),
):
    """Show how to run [bold]f1predict[/bold] from anywhere, and optionally build data."""
    import sys
    from ..config import PROJECT_ROOT

    venv_bin = Path(sys.executable).parent
    render.banner("f1predict setup")
    console.print(
        "[bold]Run the CLI[/bold] in any of these ways:\n\n"
        f"  1. Activate the venv, then use [cyan]f1predict[/cyan] directly:\n"
        f"     [green]source {PROJECT_ROOT}/.venv/bin/activate[/green]\n"
        f"     [green]f1predict --help[/green]\n\n"
        f"  2. Call the installed entry-point without activating:\n"
        f"     [green]{venv_bin}/f1predict --help[/green]\n\n"
        f"  3. Add a permanent shell alias (zsh):\n"
        f"     [green]echo 'alias f1predict=\"{venv_bin}/f1predict\"' >> ~/.zshrc[/green]\n"
        f"     [green]source ~/.zshrc[/green]\n"
    )
    console.print(
        "[bold]Typical first run[/bold]:\n"
        "  [green]f1predict data build[/green]            # fetch real F1 data (cached)\n"
        "  [green]f1predict train --preset test[/green]   # train an accurate model (~5-10 min)\n"
        "  [green]f1predict predict abudhabi2026[/green]  # predict a race\n"
        "  [green]f1predict dashboard[/green]             # launch the UI\n"
    )
    if build_data:
        from ..data_pipeline import build_master
        render.banner("building master dataset")
        with Progress(SpinnerColumn(), TextColumn("Ingesting real data…"),
                      TimeElapsedColumn(), console=console, transient=True) as p:
            p.add_task("build", total=None)
            df = build_master(cfg=_cfg(None, None), source="auto")
        console.print(f"[green]✓[/green] {len(df):,} rows across {df['season'].nunique()} seasons.")


@app.command()
def shell(
    config: Optional[str] = typer.Option(None, help="Path to a YAML config."),
):
    """Launch the interactive f1predict shell (persistent session, history & tab-complete).

    Type commands without the [bold]f1predict[/bold] prefix — [cyan]predict suzuka26[/cyan],
    [cyan]simulate vegas26 --weather rain[/cyan], [cyan]model[/cyan]. Heavy data/model state
    stays warm so every command after the first is instant. [bold]Tab[/bold] completes
    commands, races, presets & models; [bold]↑/↓[/bold] recalls history; [cyan]/help[/cyan]
    lists meta-commands.
    """
    from .shell import run_shell
    run_shell(config=config)


@app.command(hidden=True)
def repl(config: Optional[str] = typer.Option(None)):
    """Alias for [bold]shell[/bold]."""
    from .shell import run_shell
    run_shell(config=config)


@app.command()
def version():
    """Show version."""
    from .. import __version__
    console.print(f"f1predict [bold red]{__version__}[/bold red]")


if __name__ == "__main__":
    app()
