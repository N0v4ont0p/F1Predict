"""``f1p lapseq`` — the sequence model over laps (plan M6 §D.2): train, and the next-lap gate."""

from __future__ import annotations

from typing import Annotated, Any

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="lapseq: the sequence model over laps.", no_args_is_help=True)
console = Console()


def _model(size: str, pretrain: bool, sims: int | None, epochs: int | None) -> Any:
    from dataclasses import replace

    from f1p.models import lapseq as lq
    from f1p.models.lapseq_model import LapSeq

    sizes = {"small": lq.SMALL, "full": lq.FULL}
    if size not in sizes:
        console.print(f"[red]unknown size {size!r}[/red]; choose {', '.join(sizes)}")
        raise typer.Exit(code=1)
    cfg = sizes[size]
    if sims is not None:
        cfg = replace(cfg, sims_per_race=sims)
    if epochs is not None:
        cfg = replace(cfg, finetune_epochs=epochs)
    return LapSeq(config=cfg, pretrain=pretrain)


def _tables() -> dict[str, Any]:
    from f1p.cli.eval import EVAL_TABLES
    from f1p.tuning.search import load_tables

    return load_tables(tuple(EVAL_TABLES))  # the eval run's tables: the networks are shared


Size = Annotated[str, typer.Option(help="small (1.1M parameters) or full (6.3M)")]
Pretrain = Annotated[bool, typer.Option(help="Pretrain on race_sim_v2 simulations")]
Sims = Annotated[int | None, typer.Option(help="Simulated races per real race")]
Epochs = Annotated[int | None, typer.Option(help="Fine-tuning epochs (at most)")]


@app.command()
def train(
    season: Annotated[int, typer.Argument(help="Forecast season (trains on the ones before)")],
    size: Size = "small",
    pretrain: Pretrain = True,
    sims: Sims = None,
    epochs: Epochs = None,
) -> None:
    """Train (or load) the network that forecasts ``season``."""
    from f1p.models import lapseq as lq

    model = _model(size, pretrain, sims, epochs)
    tables = _tables()
    where = model._dir(tables, season)
    net = model.network(tables, season)
    if net is None:
        console.print(f"[yellow]too few training races before {season}[/yellow]")
        raise typer.Exit(code=1)
    _, cfg, meta = lq.load(where)
    table = Table(title=f"lapseq {season}: {cfg.params():,} parameters")
    for c in ("phase", "epoch", "train", "val", "s"):
        table.add_column(c)
    for row in meta["log"]:
        val = "" if row["val"] is None else f"{row['val']:.4f}"
        table.add_row(
            row["phase"], str(row["epoch"]), f"{row['train']:.4f}", val, f"{row['seconds']:.0f}"
        )
    console.print(table)
    terms = ", ".join(f"{k} {v:.4f}" for k, v in meta.get("val_terms", {}).items())
    console.print(f"validation: {terms}")
    console.print(f"network → {where}")


@app.command()
def nextlap(
    season: Annotated[int, typer.Argument(help="A tune/select season to score")],
    size: Size = "small",
    pretrain: Pretrain = True,
    sims: Sims = None,
    epochs: Epochs = None,
) -> None:
    """Gate 1 (plan §D.2): the next-lap log score on ``season``'s races, against an in-race
    running mean of each car's clean laps with a Student-t fitted on the training races.
    Also the pit, retirement and neutralisation heads against their base rates."""
    from scipy import stats

    from f1p.eval.splits import HOLDOUT, split_of
    from f1p.models import lapseq as lq
    from f1p.models.lapseq_data import tokens
    from f1p.models.lapseq_model import race_seq, training_contexts

    if split_of(season) == "holdout":
        console.print(f"[red]{season} is in the holdout {HOLDOUT}: scored once, by eval[/red]")
        raise typer.Exit(code=1)
    model = _model(size, pretrain, sims, epochs)
    tables = _tables()
    net = model.network(tables, season)
    if net is None:
        console.print(f"[yellow]too few training races before {season}[/yellow]")
        raise typer.Exit(code=1)

    def races(ctxs: list[Any]) -> list[Any]:
        return [s for c in ctxs if (s := race_seq(c, outcome=True)) is not None]

    train_races = races(training_contexts(tables, season, model.training_seasons))
    test_races = races(training_contexts(tables, season + 1, 1))
    # the baseline's Student-t: residual minus the car's running mean, on the training races
    dev = np.concatenate([_running_dev(tokens(r))[0] for r in train_races])
    df, _, scale = stats.t.fit(dev, floc=0.0)
    rows: dict[str, list[float]] = {k: [] for k in ("lapseq", "baseline")}
    heads: dict[str, list[tuple[float, float]]] = {"pit": [], "retire": [], "neutral": []}
    base_rate = {
        "pit": float(np.nanmean(np.concatenate([tokens(r).next_pit.ravel() for r in train_races]))),
        "retire": float(
            np.nanmean(np.concatenate([tokens(r).next_out.ravel() for r in train_races]))
        ),
        "neutral": float(np.nanmean(np.concatenate([tokens(r).next_neutral for r in train_races]))),
    }
    for r in test_races:
        tok = tokens(r)
        raw = net.heads(lq.collate([tok]))
        nu = float(np.asarray(raw.pop("nu")).reshape(-1)[0])
        out = {k: v[0] for k, v in raw.items()}
        mu, sigma = out["mu"], np.exp(out["log_sigma"])
        ok = np.isfinite(tok.next_res)
        rows["lapseq"] += stats.t.logpdf(tok.next_res[ok], nu, mu[ok], sigma[ok]).tolist()
        _, loc = _running_dev(tok)
        rows["baseline"] += stats.t.logpdf(tok.next_res[ok], df, loc[ok], scale).tolist()
        for name, key, target in (
            ("pit", "pit", tok.next_pit),
            ("retire", "retire", tok.next_out),
        ):
            p = 1.0 / (1.0 + np.exp(-out[key]))
            m = np.isfinite(target)
            heads[name] += list(zip(p[m].tolist(), target[m].tolist(), strict=True))
        p = 1.0 / (1.0 + np.exp(-out["neutral"]))
        m = np.isfinite(tok.next_neutral)
        heads["neutral"] += list(zip(p[m].tolist(), tok.next_neutral[m].tolist(), strict=True))

    a, b = np.asarray(rows["lapseq"]), np.asarray(rows["baseline"])
    d = a - b
    se = d.std(ddof=1) / np.sqrt(d.size)
    table = Table(title=f"next-lap log score, {season} ({len(test_races)} races, {a.size} laps)")
    for c in ("model", "mean log density", "Δ vs baseline (95% CI)"):
        table.add_column(c)
    table.add_row("lapseq", f"{a.mean():.4f}", f"{d.mean():+.4f} ± {1.96 * se:.4f}")
    table.add_row("running mean + Student-t", f"{b.mean():.4f}", "")
    console.print(table)
    heads_table = Table(title="next-lap events: log loss (lapseq vs base rate)")
    for c in ("event", "rate", "lapseq", "base rate"):
        heads_table.add_column(c)
    for name, pairs in heads.items():
        if not pairs:
            continue
        p, y = (np.asarray(v) for v in zip(*pairs, strict=True))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        q = np.clip(base_rate[name], 1e-6, 1 - 1e-6)
        ll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        lb = -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
        heads_table.add_row(name, f"{y.mean():.4f}", f"{ll:.4f}", f"{lb:.4f}")
    console.print(heads_table)


def _running_dev(tok: Any) -> tuple[np.ndarray, np.ndarray]:
    """(clean next-lap residual minus the car's running mean of its clean residuals so far,
    that running mean per token) — the in-race persistence baseline."""
    t, n = tok.next_res.shape
    loc = np.zeros((t, n))
    total = np.zeros(n)
    count = np.zeros(n)
    for i in range(t):
        loc[i] = np.where(count > 0, total / np.maximum(count, 1), 0.0)
        y = tok.next_res[i]
        ok = np.isfinite(y)
        total = total + np.where(ok, y, 0.0)
        count = count + ok
    ok = np.isfinite(tok.next_res)
    return (tok.next_res - loc)[ok], loc
