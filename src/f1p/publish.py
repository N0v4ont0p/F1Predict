"""Publish marts (plan Phase 11): versioned JSON the web app and API read.

``f1p publish`` writes, under ``<artifacts>/marts/``:

* ``index.json`` — the season calendar: for every round its status (raced / next /
  upcoming), the favourite at each checkpoint, the actual winner and the winner log-loss.
* ``events/<event_id>.json`` — every car's full finishing-position distribution at C1 and
  C4, P(win / podium / points / finish), the race-weather signal, and — once raced — the
  result and how each model scored; plus a ``quali`` block: the qualifying forecast at C1
  and C2 (pole, front row, Q3 and knockout odds), the qualifying result and its scores.
* ``ratings.json`` — current driver and team beliefs with their trajectories this season.
* ``championship.json`` — the projected final table (drivers and constructors) now.
* ``championship_history.json`` — the projected final table after every completed round.
* ``model.json`` — the rolling-origin leaderboard against the baselines with bootstrap
  intervals, reliability diagrams and the fitted stage-temperature curve.

Every payload carries ``schema_version``, the lake snapshot and the generation time.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from f1p.eval import harness
from f1p.eval.baselines import default_baselines
from f1p.eval.context import Tables, build_contexts, live_context, next_event
from f1p.eval.forecast import Forecast
from f1p.eval.metrics import reliability_table
from f1p.models.quali_v1 import QualiV1, knockout_probs
from f1p.models.season import QUANTILES, SeasonOutlook, Standing, simulate_season
from f1p.models.world_v1 import WorldV1

SCHEMA_VERSION = 1
CHECKPOINTS = ("C1", "C4")
QUALI_CHECKPOINTS = ("C1", "C2")
METRICS = ("winner_logloss", "winner_hit", "podium_brier", "points_brier", "finisher_pl_loglik")


def short_name(grand_prix_id: str | None, fallback: str) -> str:
    """A neutral display name ('United States') from the F1DB grand-prix id."""
    if not grand_prix_id:
        return fallback
    return grand_prix_id.replace("_", " ").title().replace("Sao Paulo", "São Paulo")


def _r(x: Any, nd: int = 4) -> float:
    return round(float(x), nd) if math.isfinite(float(x)) else 0.0


@dataclass
class Lookups:
    drivers: dict[str, dict[str, Any]]
    teams: dict[str, str]
    lineage: dict[tuple[int, str], str]

    @classmethod
    def from_tables(cls, tables: Tables) -> Lookups:
        drivers = {
            r["driver_id"]: r
            for r in tables["drivers"]
            .select("driver_id", "name", "last_name", "abbreviation", "permanent_number")
            .iter_rows(named=True)
        }
        teams = dict(tables["constructors"].select("constructor_id", "name").iter_rows())
        lineage = {
            (int(s), str(c)): str(lin or c)
            for s, c, lin in tables["constructor_seasons"]
            .select("season", "constructor_id", "lineage_id")
            .iter_rows()
        }
        return cls(drivers, teams, lineage)

    def driver(self, d: str) -> dict[str, Any]:
        r = self.drivers.get(d, {})
        return {
            "driver_id": d,
            "name": r.get("name") or d.replace("_", " ").title(),
            "code": r.get("abbreviation") or d[:3].upper(),
            "number": r.get("permanent_number"),
        }

    def team(self, season: int, constructor_id: str) -> dict[str, Any]:
        return {
            "constructor_id": constructor_id,
            "team": self.teams.get(constructor_id, constructor_id.replace("_", " ").title()),
            "lineage": self.lineage.get((season, constructor_id), constructor_id),
        }


def forecast_payload(
    fc: Forecast, season: int, constructors: Sequence[str], look: Lookups
) -> dict[str, Any]:
    p_finish = fc.p_finish if fc.p_finish is not None else np.ones(fc.n)
    order = np.argsort(-fc.p_win, kind="stable")
    rows = []
    for rank, i in enumerate(order.tolist(), 1):
        rows.append(
            {
                "rank": rank,
                **look.driver(fc.drivers[i]),
                **look.team(season, str(constructors[int(i)])),
                "p_win": _r(fc.p_win[i]),
                "p_podium": _r(fc.p_top(3)[i]),
                "p_points": _r(fc.p_top(10)[i]),
                "p_finish": _r(p_finish[i]),
                "expected_position": _r(fc.expected_position[i], 2),
                "position_probs": [_r(p) for p in fc.position_probs[i]],
                "why": {k: _r(v[i], 3) for k, v in (fc.components or {}).items()},
            }
        )
    return {"model": fc.model, "meta": {k: _r(v) for k, v in fc.meta.items()}, "cars": rows}


def outcome_payload(tables: Tables, event_id: str, look: Lookups) -> list[dict[str, Any]] | None:
    r = tables["results"].filter((pl.col("event_id") == event_id) & (pl.col("session") == "race"))
    if r.is_empty():
        return None
    r = r.sort(pl.col("position").is_null(), "position", "display_order")
    season = int(r["season"][0])
    return [
        {
            **look.driver(row["driver_id"]),
            **look.team(season, row["constructor_id"]),
            "position": row["position"],
            "classified": row["classified"],
            "status": row["status"],
            "grid": row["grid"],
            "points": row["points"],
        }
        for row in r.iter_rows(named=True)
    ]


def quali_outcome_payload(
    tables: Tables, event_id: str, season: int, look: Lookups
) -> list[dict[str, Any]] | None:
    q = tables["qualifying"].filter(
        (pl.col("event_id") == event_id) & (pl.col("session") == "quali")
    )
    if q.is_empty():
        return None
    q = q.sort(pl.col("position").is_null(), "position", "display_order")
    return [
        {
            **look.driver(row["driver_id"]),
            **look.team(season, row["constructor_id"]),
            "position": row["position"],
            "q1_ms": row["q1_ms"],
            "q2_ms": row["q2_ms"],
            "q3_ms": row["q3_ms"],
        }
        for row in q.iter_rows(named=True)
    ]


def quali_forecast_payload(
    fc: Forecast, season: int, constructors: Sequence[str], look: Lookups
) -> dict[str, Any]:
    """:func:`forecast_payload` plus pole / front-row / knockout probabilities per car."""
    out = forecast_payload(fc, season, constructors, look)
    ko = knockout_probs(fc, season)
    index = {d: i for i, d in enumerate(fc.drivers)}
    front = fc.p_top(2)
    for car in out["cars"]:
        i = index[car["driver_id"]]
        car["p_front_row"] = _r(front[i])
        if ko is not None:
            car["p_q3"] = _r(ko.q3[i])
            car["p_out_q2"] = _r(ko.out_q2[i])
            car["p_out_q1"] = _r(ko.out_q1[i])
    return out


def _weather(tables: Tables, event_id: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "weather_forecast" in tables:
        w = tables["weather_forecast"].filter(pl.col("event_id") == event_id)
        for row in w.iter_rows(named=True):
            out[row["kind"]] = {
                "wet_prob": _r(row["wet_prob"]),
                "precip_mm": _r(row["precip_mm"] or 0.0, 2),
                "temp_c": None if row["temp_c"] is None else _r(row["temp_c"], 1),
            }
    if "weather_observed" in tables:
        o = tables["weather_observed"].filter(pl.col("event_id") == event_id)
        if o.height:
            row = o.row(0, named=True)
            out["observed"] = {
                "wet": row["wet"],
                "precip_mm": _r(row["precip_mm"], 2),
                "temp_c": None if row["temp_c"] is None else _r(row["temp_c"], 1),
            }
    return out


def build_season(
    tables: Tables, season: int, model: WorldV1, *, history_from: int
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """The season calendar and one payload per event."""
    look = Lookups.from_tables(tables)
    baselines = [m for m in default_baselines() if m.name in ("quali", "standings", "grid")]
    result = harness.run(
        tables,
        [model, *baselines],
        seasons=(season, season),
        history_from=history_from,
        keep_forecasts=True,
    )
    contexts = {
        (c.event_id, c.checkpoint): (c, o)
        for c, o, _ in build_contexts(tables, seasons=(season, season))
    }
    qmodel = QualiV1()
    qbase = [m for m in default_baselines("quali") if m.name in ("last_quali", "practice")]
    qresult = harness.run(
        tables,
        [qmodel, *qbase],
        seasons=(season, season),
        history_from=history_from,
        keep_forecasts=True,
        target="quali",
    )
    qcontexts = {
        (c.event_id, c.checkpoint): c
        for c, _, _ in build_contexts(tables, seasons=(season, season), target="quali")
    }
    upcoming = next_event(tables)
    events = tables["events"].filter(pl.col("season") == season).sort("round")
    per_event: dict[str, dict[str, Any]] = {}
    calendar = []
    for ev in events.iter_rows(named=True):
        eid = ev["event_id"]
        name = short_name(ev["grand_prix_id"], ev["name"])
        status = "raced" if (eid, "C1") in contexts else ("next" if eid == upcoming else "upcoming")
        payload: dict[str, Any] = {
            "event_id": eid,
            "season": season,
            "round": ev["round"],
            "name": name,
            "official_name": ev["name"],
            "circuit_id": ev["circuit_id"],
            "race_date": ev["race_date"].isoformat(),
            "race_start_utc": (
                ev["race_start_utc"].isoformat() if ev["race_start_utc"] is not None else None
            ),
            "sprint": ev["has_sprint"],
            "status": status,
            "weather": _weather(tables, eid),
            "checkpoints": {},
            "scores": {},
        }
        for cp in CHECKPOINTS:
            if status == "raced":
                fc = result.forecasts.get((model.name, cp, eid))
                ctx, _outcome = contexts[(eid, cp)]
                if fc is None:
                    continue
                payload["checkpoints"][cp] = forecast_payload(fc, season, ctx.constructors, look)
                payload["checkpoints"][cp]["as_of"] = ctx.as_of.isoformat()
                payload["scores"][cp] = {
                    m.name: _scores(result, m.name, cp, eid)
                    for m in [model, *baselines]
                    if (m.name, cp, eid) in result.forecasts
                }
            elif status == "next":
                live = _live(tables, model, eid, cp, history_from)
                if live is not None:
                    fc, ctx = live
                    payload["checkpoints"][cp] = forecast_payload(
                        fc, season, ctx.constructors, look
                    )
                    payload["checkpoints"][cp]["as_of"] = ctx.as_of.isoformat()
        payload["result"] = outcome_payload(tables, eid, look) if status == "raced" else None
        quali: dict[str, Any] = {"checkpoints": {}, "scores": {}, "result": None}
        for cp in QUALI_CHECKPOINTS:
            if (eid, cp) in qcontexts:
                fc = qresult.forecasts.get((qmodel.name, cp, eid))
                if fc is None:
                    continue
                qctx = qcontexts[(eid, cp)]
                quali["checkpoints"][cp] = quali_forecast_payload(
                    fc, season, qctx.constructors, look
                )
                quali["checkpoints"][cp]["as_of"] = qctx.as_of.isoformat()
                quali["scores"][cp] = {
                    m.name: _scores(qresult, m.name, cp, eid)
                    for m in [qmodel, *qbase]
                    if (m.name, cp, eid) in qresult.forecasts
                }
            elif status == "next":
                live = _live(tables, qmodel, eid, cp, history_from, target="quali")
                if live is not None:
                    fc, qctx = live
                    quali["checkpoints"][cp] = quali_forecast_payload(
                        fc, season, qctx.constructors, look
                    )
                    quali["checkpoints"][cp]["as_of"] = qctx.as_of.isoformat()
        if any((eid, cp) in qcontexts for cp in QUALI_CHECKPOINTS):
            quali["result"] = quali_outcome_payload(tables, eid, season, look)
        payload["quali"] = quali
        per_event[eid] = payload
        calendar.append(_calendar_row(payload))
    index = {"season": season, "next_event": upcoming, "events": calendar}
    return index, per_event


def _scores(result: harness.EvalResult, model: str, cp: str, eid: str) -> dict[str, float]:
    row = result.per_event.filter(
        (pl.col("model") == model) & (pl.col("checkpoint") == cp) & (pl.col("event_id") == eid)
    )
    if row.is_empty():
        return {}
    r = row.row(0, named=True)
    return {m: _r(r[m]) for m in METRICS if r.get(m) is not None}


def _live(
    tables: Tables,
    model: WorldV1,
    eid: str,
    cp: str,
    history_from: int,
    *,
    target: str = "race",
) -> tuple[Forecast, Any] | None:
    """Forecast an unraced event at ``cp`` — only if that checkpoint's information exists."""
    ctx = live_context(tables, eid, cp, target=target)
    if cp == "C4":
        grid = ctx.known("grid").filter(pl.col("event_id") == eid)
        if grid.is_empty():
            return None  # qualifying not run yet: a "C4" forecast would just be C1 again
    if cp == "C2" and "practice" in tables:
        practice = ctx.known("practice").filter(pl.col("event_id") == eid)
        if practice.is_empty():
            return None  # no practice yet: a "C2" forecast would just be C1 again
    fc, ctx = harness.forecast_event(
        tables, model, eid, cp, history_seasons=ctx.season - history_from, target=target
    )
    return (fc, ctx) if fc is not None else None


def _calendar_row(p: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        k: p[k]
        for k in ("event_id", "round", "name", "circuit_id", "race_date", "sprint", "status")
    }
    for cp, fc in p["checkpoints"].items():
        top = fc["cars"][0]
        row[f"favourite_{cp}"] = {
            "driver_id": top["driver_id"],
            "code": top["code"],
            "p_win": top["p_win"],
        }
    if p.get("result"):
        w = p["result"][0]
        row["winner"] = {"driver_id": w["driver_id"], "code": w["code"], "team": w["team"]}
        for cp in CHECKPOINTS:
            s = p["scores"].get(cp, {}).get("world_v1")
            if s:
                row[f"winner_logloss_{cp}"] = s.get("winner_logloss")
                cars = {c["driver_id"]: c for c in p["checkpoints"][cp]["cars"]}
                row[f"p_winner_{cp}"] = cars.get(w["driver_id"], {}).get("p_win")
    wet = p["weather"].get("observed", {}).get("wet")
    if wet is not None:
        row["wet"] = wet
    return row


def build_ratings(tables: Tables, season: int, model: WorldV1) -> dict[str, Any]:
    look = Lookups.from_tables(tables)
    st = model._state(tables)
    season_steps = [i for i, r in enumerate(st.races) if r.season == season and r.is_race]

    def series(hist: Any) -> list[dict[str, Any]]:
        out = []
        for step, (mu, var, _) in zip(hist.steps, hist.states, strict=True):
            if step in season_steps:
                out.append({"event_id": st.races[step].event_id, "mu": _r(mu), "sd": _r(var**0.5)})
        return out

    drivers = []
    for d, b in st.filter.drivers.items():
        if b.season == season:
            h = st.filter.driver_hist.get(d)
            drivers.append(
                {
                    **look.driver(d),
                    "mu": _r(b.mu),
                    "sd": _r(b.var**0.5),
                    "series": series(h) if h else [],
                }
            )
    teams = []
    for t, b in st.filter.teams.items():
        if b.season == season:
            h = st.filter.team_hist.get(t)
            teams.append(
                {
                    "lineage": t,
                    "team": look.teams.get(t, t.replace("_", " ").title()),
                    "mu": _r(b.mu),
                    "sd": _r(b.var**0.5),
                    "series": series(h) if h else [],
                }
            )
    drivers.sort(key=lambda r: -r["mu"])
    teams.sort(key=lambda r: -r["mu"])
    return {"season": season, "drivers": drivers, "teams": teams}


def _standing_row(s: Standing, *, positions: int | None = None) -> dict[str, Any]:
    q = s.quantiles
    probs = s.p_position if positions is None else s.p_position[:positions]
    return {
        "current": _r(s.current, 1),
        "p_title": _r(s.p_title),
        "p_top3": _r(s.p_top3),
        "expected": _r(s.expected, 1),
        "quantiles": [_r(x, 1) for x in q],
        "most_likely": s.most_likely + 1,
        "p_position": [_r(p, 4) for p in probs],
    }


def _identity(o: SeasonOutlook, look: Lookups, kind: str, ident: str) -> dict[str, Any]:
    if kind == "driver":
        row = look.driver(ident)
        if ident in o.team_of:
            row |= look.team(o.season, o.team_of[ident])
        return row
    return look.team(o.season, ident)


def build_championship(
    tables: Tables, season: int, *, n_sims: int = 20_000, outlook: SeasonOutlook | None = None
) -> dict[str, Any]:
    """The projected final table now: every driver and team with points now, expected and
    median final points, 50% and 90% ranges, P(title), P(top 3), the full final-position
    distribution and the jointly most likely final order."""
    look = Lookups.from_tables(tables)
    o = outlook or simulate_season(tables, season, n_sims=n_sims)

    def rows(standings: tuple[Standing, ...], kind: str) -> list[dict[str, Any]]:
        return [_identity(o, look, kind, s.id) | _standing_row(s) for s in standings]

    return {
        "season": season,
        "as_of": o.as_of.isoformat(),
        "after_round": o.after_round,
        "remaining": list(o.remaining),
        "sprints_remaining": o.sprints,
        "n_sims": o.n_sims,
        "quantiles": list(QUANTILES),
        "drivers": rows(o.drivers, "driver"),
        "constructors": rows(o.constructors, "constructor"),
    }


def build_championship_history(
    tables: Tables, season: int, *, n_sims: int = 5_000, model: WorldV1 | None = None
) -> dict[str, Any] | None:
    """One projected final table per completed round of ``season`` (pre-season included),
    each made with only what was public right after that round's result."""
    look = Lookups.from_tables(tables)
    events = tables["events"].filter(pl.col("season") == season).sort("round")
    raced = sorted(
        set(
            tables["results"]
            .filter((pl.col("season") == season) & (pl.col("session") == "race"))["round"]
            .to_list()
        )
    )
    if not raced:
        return None
    model = model or WorldV1()
    forecaster = harness.EventForecaster(tables, model, "C1")
    identities: dict[str, dict[str, dict[str, Any]]] = {"driver": {}, "constructor": {}}
    snapshots = []
    for after in [0, *raced]:
        o = simulate_season(
            tables, season, model=model, after_round=after, n_sims=n_sims, forecaster=forecaster
        )
        snap: dict[str, Any] = {
            "after_round": after,
            "as_of": o.as_of.isoformat(),
            "remaining": len(o.remaining),
        }
        for kind, standings in (("driver", o.drivers), ("constructor", o.constructors)):
            key = "driver_id" if kind == "driver" else "constructor_id"
            for s in standings:
                identities[kind][s.id] = _identity(o, look, kind, s.id)
            snap[f"{kind}s"] = [{key: s.id} | _standing_row(s) for s in standings]
        snapshots.append(snap)
    return {
        "season": season,
        "n_sims": n_sims,
        "quantiles": list(QUANTILES),
        "rounds": [
            {
                "round": ev["round"],
                "event_id": ev["event_id"],
                "name": short_name(ev["grand_prix_id"], ev["name"]),
            }
            for ev in events.iter_rows(named=True)
        ],
        "drivers": identities["driver"],
        "constructors": identities["constructor"],
        "snapshots": snapshots,
    }


def build_model_card(
    tables: Tables, model: WorldV1, *, seasons: tuple[int, int], history_from: int
) -> dict[str, Any]:
    """Leaderboard (with bootstrap CIs vs the best baseline) and reliability diagrams."""
    baselines = default_baselines()
    result = harness.run(tables, [model, *baselines], seasons=seasons, history_from=history_from)
    summary = result.summary()
    board = []
    for row in summary.iter_rows(named=True):
        entry = {"model": row["model"], "checkpoint": row["checkpoint"], "events": row["events"]}
        entry |= {m: _r(row[m]) for m in METRICS if row.get(m) is not None}
        if row.get("ece_top3") is not None:
            entry["ece_top3"] = _r(row["ece_top3"])
        board.append(entry)
    reference = {"C1": "standings", "C4": "quali"}
    comparisons = {}
    for cp, ref in reference.items():
        comparisons[cp] = {}
        for m in METRICS:
            c = result.compare(model.name, ref, cp, m)
            comparisons[cp][m] = {
                "reference": ref,
                "mean_diff": _r(c.mean_diff),
                "lo": _r(c.lo),
                "hi": _r(c.hi),
                "significant": c.significant,
            }
    reliability = {}
    for (name, cp, k), (p, y) in result.calibration.items():
        if name == model.name:
            reliability[f"{cp}_top{k}"] = [
                {"predicted": _r(a), "observed": _r(b), "count": n}
                for a, b, n in reliability_table(p, y)
            ]
    latest = max(model._weights) if model._weights else None
    curve = {}
    if latest is not None:
        for cp in CHECKPOINTS:
            fit = model._weights.get((latest[0], cp))
            if fit is not None:
                curve[cp] = {
                    "weights": {n: _r(v) for n, v in zip(model.features[cp], fit.w, strict=True)},
                    "stage_temperature": [_r(float(np.exp(t))) for t in fit.theta[:22]],
                    "rain_gamma": _r(fit.gamma),
                }
    return {
        "model": model.name,
        "seasons": list(seasons),
        "leaderboard": board,
        "comparisons": comparisons,
        "reliability": reliability,
        "combiner": curve,
    }


def publish(
    tables: Tables,
    out_dir: Path,
    *,
    season: int,
    snapshot: int | None,
    leaderboard_from: int | None = 2014,
    history_seasons: int = 8,
    championship_history: bool = True,
) -> list[Path]:
    """Write every mart; returns the files written."""
    stamp = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "lake_snapshot": snapshot,
    }
    model = WorldV1()
    index, events = build_season(tables, season, model, history_from=season - history_seasons)
    written = []

    def write(rel: str, payload: dict[str, Any]) -> None:
        path = out_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**stamp, **payload}, separators=(",", ":")))
        written.append(path)

    write("index.json", index)
    for eid, payload in events.items():
        write(f"events/{eid}.json", payload)
    write("ratings.json", build_ratings(tables, season, model))
    if "constructor_standings" in tables:
        if index.get("next_event"):
            write("championship.json", build_championship(tables, season))
        if championship_history:
            hist = build_championship_history(tables, season)
            if hist is not None:
                write("championship_history.json", hist)
    if leaderboard_from is not None:
        card = build_model_card(
            tables,
            WorldV1(),
            seasons=(leaderboard_from, season),
            history_from=leaderboard_from - history_seasons,
        )
        write("model.json", card)
    return written
