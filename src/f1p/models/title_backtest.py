"""Title backtest: how well did the season simulator call championships (plan item 4)?

For every season, a projection is made after every round (pre-season included, the final
round excluded — the title is decided by then) using only what was public at that moment
(:func:`f1p.models.season.simulate_season` with ``after_round``). Each snapshot is scored on

* **title log-loss** — ``-log P(actual champion)``, clipped at ``1 / n_sims``;
* **position Brier** — the mean over competitors of the squared error of their whole
  final-position distribution against the place they actually finished.

The baseline is the same simulator with every car equally strong and never retiring:
"current points plus a uniform spread". Snapshots within a season are correlated, so the
comparison bootstraps whole seasons.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import polars as pl

from f1p.eval.bootstrap import Comparison, paired_bootstrap
from f1p.eval.context import Tables
from f1p.eval.harness import EventForecaster
from f1p.models.season import SeasonOutlook, Standing, simulate_season
from f1p.models.world_v1 import WorldV1

KINDS = ("driver", "constructor")
MODELS = ("world_v1", "uniform")


def final_order(tables: Tables, season: int, kind: str) -> dict[str, int]:
    """0-based final championship place of every classified competitor."""
    name, key = (
        ("driver_standings", "driver_id")
        if kind == "driver"
        else ("constructor_standings", "constructor_id")
    )
    st = tables[name].filter(pl.col("season") == season)
    if st.is_empty():
        return {}
    last = st.filter(pl.col("round") == st["round"].max()).sort("display_order")
    return {str(x): i for i, x in enumerate(last[key].to_list())}


def score_snapshot(
    rows: tuple[Standing, ...], actual: dict[str, int], n_sims: int
) -> dict[str, float]:
    champion = next((k for k, v in actual.items() if v == 0), None)
    p_champ = next((s.p_title for s in rows if s.id == champion), 0.0)
    floor = 1.0 / n_sims
    briers = []
    for s in rows:
        if s.id not in actual:
            continue
        p = np.asarray(s.p_position)
        truth = np.zeros_like(p)
        if actual[s.id] < p.size:
            truth[actual[s.id]] = 1.0
        briers.append(float(((p - truth) ** 2).sum()))
    return {
        "p_champion": p_champ,
        "title_logloss": float(-np.log(max(p_champ, floor))),
        "position_brier": float(np.mean(briers)) if briers else float("nan"),
        "champion_favourite": float(
            bool(rows) and max(rows, key=lambda s: s.p_title).id == champion
        ),
    }


def run(
    tables: Tables,
    seasons: tuple[int, int],
    *,
    n_sims: int = 5_000,
    seed: int = 2026,
    progress: Callable[[int, int], None] | None = None,
) -> pl.DataFrame:
    """One row per (season, after_round, kind, model) snapshot."""
    rows = []
    for season in range(seasons[0], seasons[1] + 1):
        raced = (
            tables["results"]
            .filter((pl.col("season") == season) & (pl.col("session") == "race"))["round"]
            .unique()
            .sort()
            .to_list()
        )
        if not raced:
            continue
        final_round = int(tables["events"].filter(pl.col("season") == season)["round"].max())  # type: ignore[arg-type]
        if raced[-1] != final_round:
            continue  # season still running: no final standings to score against
        actual = {k: final_order(tables, season, k) for k in KINDS}
        model = WorldV1()
        forecaster = EventForecaster(tables, model, "C1")
        for after in [0, *raced[:-1]]:
            outlooks: dict[str, SeasonOutlook] = {
                "world_v1": simulate_season(
                    tables,
                    season,
                    model=model,
                    after_round=after,
                    n_sims=n_sims,
                    seed=seed,
                    forecaster=forecaster,
                ),
                "uniform": simulate_season(
                    tables, season, after_round=after, n_sims=n_sims, seed=seed, uniform=True
                ),
            }
            for name, o in outlooks.items():
                for kind in KINDS:
                    standings = o.drivers if kind == "driver" else o.constructors
                    rows.append(
                        {
                            "season": season,
                            "after_round": after,
                            "rounds": final_round,
                            "kind": kind,
                            "model": name,
                            **score_snapshot(standings, actual[kind], n_sims),
                        }
                    )
        if progress is not None:
            progress(season, len(raced))
    return pl.DataFrame(rows)


@dataclass(frozen=True, slots=True)
class TitleSummary:
    kind: str
    metric: str
    model: float
    baseline: float
    comparison: Comparison


def summarise(df: pl.DataFrame) -> list[TitleSummary]:
    out = []
    for kind in KINDS:
        sub = df.filter(pl.col("kind") == kind).sort("season", "after_round")
        a = sub.filter(pl.col("model") == "world_v1")
        b = sub.filter(pl.col("model") == "uniform")
        if a.is_empty():
            continue
        for metric in ("title_logloss", "position_brier"):
            x = a[metric].to_numpy()
            y = b[metric].to_numpy()
            cmp = paired_bootstrap(x, y, clusters=a["season"].to_numpy())
            out.append(TitleSummary(kind, metric, float(np.nanmean(x)), float(np.nanmean(y)), cmp))
    return out


def called_from(df: pl.DataFrame, kind: str = "driver", threshold: float = 0.5) -> pl.DataFrame:
    """Per season: the first round after which the model's P(actual champion) stayed at or
    above ``threshold`` for good (``None`` if it never did before the finale)."""
    out = []
    sub = df.filter((pl.col("kind") == kind) & (pl.col("model") == "world_v1"))
    for season in sorted(set(sub["season"].to_list())):
        s = sub.filter(pl.col("season") == season).sort("after_round")
        p = s["p_champion"].to_numpy()
        rounds = s["after_round"].to_list()
        call = None
        for i in range(len(p) - 1, -1, -1):
            if p[i] < threshold:
                break
            call = rounds[i]
        out.append(
            {
                "season": season,
                "rounds": int(s["rounds"][0]),
                "called_after": call,
                "p_preseason": float(p[0]),
            }
        )
    return pl.DataFrame(out)
