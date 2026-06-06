"""Deterministic synthetic F1 data generator (offline fallback / tests only).

Real jolpica data is the default everywhere. This seeded generator exists purely so the
test-suite and a fully-offline machine still have a schema-correct dataset. It uses a
latent skill/pace model and now emits the enriched v2 columns (qualifying, sprint, dnf).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.points import points_for_position
from .schema import RAW_COLUMNS

_DRIVERS = [
    ("verstappen", "VER", "Max Verstappen"), ("hamilton", "HAM", "Lewis Hamilton"),
    ("leclerc", "LEC", "Charles Leclerc"), ("norris", "NOR", "Lando Norris"),
    ("russell", "RUS", "George Russell"), ("sainz", "SAI", "Carlos Sainz"),
    ("perez", "PER", "Sergio Perez"), ("alonso", "ALO", "Fernando Alonso"),
    ("piastri", "PIA", "Oscar Piastri"), ("gasly", "GAS", "Pierre Gasly"),
    ("ocon", "OCO", "Esteban Ocon"), ("stroll", "STR", "Lance Stroll"),
    ("albon", "ALB", "Alexander Albon"), ("tsunoda", "TSU", "Yuki Tsunoda"),
    ("hulkenberg", "HUL", "Nico Hulkenberg"), ("bottas", "BOT", "Valtteri Bottas"),
    ("ricciardo", "RIC", "Daniel Ricciardo"), ("zhou", "ZHO", "Guanyu Zhou"),
    ("magnussen", "MAG", "Kevin Magnussen"), ("sargeant", "SAR", "Logan Sargeant"),
]
_CONSTRUCTORS = [
    ("red_bull", "Red Bull"), ("mercedes", "Mercedes"), ("ferrari", "Ferrari"),
    ("mclaren", "McLaren"), ("aston_martin", "Aston Martin"), ("alpine", "Alpine"),
    ("williams", "Williams"), ("rb", "RB"), ("sauber", "Sauber"), ("haas", "Haas"),
]
_CIRCUITS = [
    ("bahrain", "Bahrain GP", "Bahrain"), ("jeddah", "Saudi Arabian GP", "Saudi Arabia"),
    ("albert_park", "Australian GP", "Australia"), ("imola", "Emilia Romagna GP", "Italy"),
    ("miami", "Miami GP", "USA"), ("monaco", "Monaco GP", "Monaco"),
    ("catalunya", "Spanish GP", "Spain"), ("villeneuve", "Canadian GP", "Canada"),
    ("red_bull_ring", "Austrian GP", "Austria"), ("silverstone", "British GP", "UK"),
    ("hungaroring", "Hungarian GP", "Hungary"), ("spa", "Belgian GP", "Belgium"),
    ("zandvoort", "Dutch GP", "Netherlands"), ("monza", "Italian GP", "Italy"),
    ("marina_bay", "Singapore GP", "Singapore"), ("suzuka", "Japanese GP", "Japan"),
    ("americas", "United States GP", "USA"), ("rodriguez", "Mexico City GP", "Mexico"),
    ("interlagos", "Sao Paulo GP", "Brazil"), ("yas_marina", "Abu Dhabi GP", "UAE"),
]


def generate(seasons, n_drivers: int = 20, rounds_per_season: int = 20, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    drivers = _DRIVERS[:n_drivers]
    skill = {d[0]: rng.normal(0, 1) for d in drivers}
    pace = {c[0]: rng.normal(0, 1) for c in _CONSTRUCTORS}
    assign = {drivers[i][0]: _CONSTRUCTORS[i // 2][0] for i in range(len(drivers))}
    code = {d[0]: d[1] for d in drivers}
    name = {d[0]: d[2] for d in drivers}

    rows: list[dict] = []
    for season in sorted(seasons):
        for k in skill:
            skill[k] += rng.normal(0, 0.15)
        for k in pace:
            pace[k] += rng.normal(0, 0.25)
        circuits = _CIRCUITS[:rounds_per_season]
        for rnd, (circ_id, race_name, country) in enumerate(circuits, start=1):
            strength = {d: skill[d] + pace[assign[d]] + rng.normal(0, 0.4) for d in skill}
            quali = sorted(strength, key=lambda d: strength[d] + rng.normal(0, 0.3), reverse=True)
            qpos = {d: i + 1 for i, d in enumerate(quali)}
            race_score = {d: -qpos[d] + strength[d] * 0.8 + rng.normal(0, 1.1) for d in strength}
            dnf = {d: rng.random() < 0.06 for d in strength}
            order = sorted([d for d in strength if not dnf[d]], key=lambda d: race_score[d], reverse=True)
            dnfs = [d for d in strength if dnf[d]]
            date = f"{season}-{(rnd % 12) + 1:02d}-{((rnd * 2) % 28) + 1:02d}"
            base = {"season": season, "round": rnd, "date": date, "race_name": race_name,
                    "circuit_id": circ_id, "country": country, "locality": country}
            for fp, d in enumerate(order, start=1):
                rows.append({**base, "driver_id": d, "driver_code": code[d], "driver_name": name[d],
                             "constructor_id": assign[d], "constructor_name": dict(_CONSTRUCTORS)[assign[d]],
                             "quali_position": qpos[d], "quali_best_ms": 80000 + qpos[d] * 150 + rng.normal(0, 50),
                             "grid": qpos[d], "position": fp, "status": "Finished",
                             "points": points_for_position(fp, season, "R"), "laps": 58,
                             "finished": 1, "dnf": 0, "sprint_position": 0,
                             "sprint_points": 0.0, "has_sprint": 0})
            for d in dnfs:
                rows.append({**base, "driver_id": d, "driver_code": code[d], "driver_name": name[d],
                             "constructor_id": assign[d], "constructor_name": dict(_CONSTRUCTORS)[assign[d]],
                             "quali_position": qpos[d], "quali_best_ms": 80000 + qpos[d] * 150,
                             "grid": qpos[d], "position": 0,
                             "status": rng.choice(["Engine", "Accident", "Gearbox", "Hydraulics"]),
                             "points": 0.0, "laps": int(rng.integers(5, 50)), "finished": 0,
                             "dnf": 1, "sprint_position": 0, "sprint_points": 0.0, "has_sprint": 0})
    return _coerce(pd.DataFrame(rows, columns=list(RAW_COLUMNS)))


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    for col, dtype in RAW_COLUMNS.items():
        if col not in df:
            df[col] = 0 if dtype in ("int32", "float32") else ""
        if dtype == "int32":
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int32")
        elif dtype == "float32":
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype("float32")
        else:
            df[col] = df[col].astype("string").fillna("")
    return df[list(RAW_COLUMNS)]
