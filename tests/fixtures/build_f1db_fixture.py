"""Build ``tests/fixtures/f1db_subset.db.gz`` from a full F1DB SQLite release.

    uv run python tests/fixtures/build_f1db_fixture.py path/to/f1db.db

The subset keeps the full 2021 and 2026 seasons (sprints, half points, a dropped-duplicate
pit stop) plus single events that exercise every special case the curation and the gates
handle: 1983 Brazil (vacant P2 after a DSQ), 1987 Austria (zero-lap "classified" rows),
2014 Abu Dhabi (double points) and 2022 Japan (full points despite 53% distance).
Standings are kept only for the complete seasons, where they can be reconciled.

Data: F1DB (https://github.com/f1db/f1db), CC BY 4.0.
"""

from __future__ import annotations

import gzip
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

FULL_SEASONS = (2021, 2026)
SINGLE_EVENTS = ((1983, 1), (1987, 10), (2014, 19), (2022, 18))

RACE_TABLES = (
    "race_result",
    "sprint_race_result",
    "qualifying_result",
    "sprint_qualifying_result",
    "starting_grid_position",
    "sprint_starting_grid_position",
    "free_practice_1_result",
    "free_practice_2_result",
    "free_practice_3_result",
    "pit_stop",
    "fastest_lap",
)
STANDINGS_TABLES = ("race_driver_standing", "race_constructor_standing")
ENTITY_TABLES = (
    "circuit",
    "circuit_layout",
    "driver",
    "constructor",
    "engine_manufacturer",
    "constructor_chronology",
)


def build(source: Path, dest: Path) -> None:
    seasons = ",".join(str(s) for s in FULL_SEASONS)
    singles = " OR ".join(f"(year={y} AND round={r})" for y, r in SINGLE_EVENTS)
    years = ",".join(str(y) for y in sorted({*FULL_SEASONS, *(y for y, _ in SINGLE_EVENTS)}))
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "f1db.db"
        out = sqlite3.connect(out_path)
        # Several F1DB "tables" are views over race_data; materialise each as a plain table.
        out.execute("ATTACH DATABASE ? AS src", (f"file:{source}?mode=ro",))
        selected = f"SELECT id FROM src.race WHERE year IN ({seasons}) OR {singles}"
        full = f"SELECT id FROM src.race WHERE year IN ({seasons})"

        def copy(table: str, where: str) -> None:
            out.execute(f"CREATE TABLE main.{table} AS SELECT * FROM src.{table} WHERE {where}")

        copy("race", f"id IN ({selected})")
        for t in RACE_TABLES:
            copy(t, f"race_id IN ({selected})")
        for t in STANDINGS_TABLES:
            copy(t, f"race_id IN ({full})")
        copy("season_entrant_driver", f"year IN ({years})")
        for t in ENTITY_TABLES:
            copy(t, "1=1")
        n_events = out.execute("SELECT count(*) FROM main.race").fetchone()[0]
        out.commit()
        out.execute("DETACH DATABASE src")
        out.execute("VACUUM")
        out.close()
        with out_path.open("rb") as fin, gzip.open(dest, "wb", compresslevel=9) as fout:
            shutil.copyfileobj(fin, fout)
    print(f"wrote {dest} ({dest.stat().st_size / 1e6:.2f} MB, {n_events} events)")


if __name__ == "__main__":
    build(Path(sys.argv[1]), Path(__file__).with_name("f1db_subset.db.gz"))
