from __future__ import annotations

import polars as pl
import pytest

from f1p.core.status import StatusClass
from f1p.lake import schema as S
from f1p.lake.curate_f1db import Curated, result_status


def _race(curated: Curated, event_id: str, session: str = "race") -> pl.DataFrame:
    return curated.tables["results"].filter(
        (pl.col("event_id") == event_id) & (pl.col("session") == session)
    )


def test_every_table_matches_its_spec(curated: Curated) -> None:
    assert set(curated.tables) == set(S.SPECS)
    for name, df in curated.tables.items():
        S.validate(df, S.SPECS[name])
        assert df.height > 0, name


def test_corrections_are_reported(curated: Curated) -> None:
    joined = " | ".join(curated.notes)
    assert "zero-lap" in joined
    assert "2026-R13" in joined


def test_ids_are_snake_case(curated: Curated) -> None:
    for col in ("driver_id", "constructor_id"):
        values = curated.tables["results"][col]
        assert not values.str.contains("-").any(), col
    assert "max_verstappen" in curated.tables["drivers"]["driver_id"].to_list()


def test_event_ids_and_fractions(curated: Curated) -> None:
    ev = curated.tables["events"]
    suzuka = ev.filter(pl.col("event_id") == "2022-R18").row(0, named=True)
    assert suzuka["circuit_id"] == "suzuka"
    assert suzuka["distance_fraction"] == pytest.approx(162.296 / 307.471)
    spa = ev.filter(pl.col("event_id") == "2021-R12").row(0, named=True)
    assert spa["distance_fraction"] < 0.05
    assert ev.filter(pl.col("event_id") == "2021-R10")["has_sprint"].item()


def test_lapped_finishers_are_classified_not_dnf(curated: Curated) -> None:
    r = curated.tables["results"]
    lapped = r.filter(pl.col("status") == StatusClass.LAPPED.value)
    assert lapped.height > 0
    assert lapped["classified"].all()


def test_zero_lap_retirement_declassified(curated: Curated) -> None:
    streiff = _race(curated, "1987-R10").filter(pl.col("driver_id") == "philippe_streiff")
    row = streiff.row(0, named=True)
    assert row["classified"] is False
    assert row["position"] is None
    assert row["status"] == StatusClass.DNF_INCIDENT.value


def test_pit_stop_duplicates_dropped(curated: Curated) -> None:
    ps = curated.tables["pit_stops"].filter(pl.col("event_id") == "2026-R13")
    assert ps.group_by("driver_id", "stop").len()["len"].max() == 1


def test_lineage_links_renamed_teams(curated: Curated) -> None:
    cs = curated.tables["constructor_seasons"]
    lineage = dict(
        cs.filter(pl.col("season") == 2021).select("constructor_id", "lineage_id").iter_rows()
    )
    assert lineage["alphatauri"] == "racing_bulls"
    assert lineage["alfa_romeo"] == "audi"
    assert lineage["aston_martin"] == "aston_martin"


def test_entries_explode_rounds(curated: Curated) -> None:
    e = curated.tables["entries"].filter((pl.col("season") == 2026) & ~pl.col("test_driver"))
    lawson = e.filter(pl.col("driver_id") == "liam_lawson")
    assert set(lawson.filter(pl.col("constructor_id") == "red_bull")["round"].to_list()) == {
        12,
        13,
        14,
    }
    # Reserve/test drivers without a race appearance have no round; a reserve who stood in
    # for a race keeps the rounds they raced.
    reserves = curated.tables["entries"].filter(pl.col("test_driver"))
    assert reserves["round"].null_count() > 0
    assert curated.tables["entries"].filter(~pl.col("test_driver"))["round"].null_count() == 0


def test_results_known_after_qualifying_and_before_next_event(curated: Curated) -> None:
    r = curated.tables["results"].filter(pl.col("session") == "race")
    q = curated.tables["qualifying"].filter(pl.col("session") == "quali")
    ev = curated.tables["events"].filter(pl.col("event_id") == "2026-R17").row(0, named=True)
    race_obs = r.filter(pl.col("event_id") == "2026-R17")["observed_at"].max()
    quali_obs = q.filter(pl.col("event_id") == "2026-R17")["observed_at"].max()
    assert quali_obs < ev["race_start_utc"] < race_obs  # type: ignore[operator]


@pytest.mark.parametrize(
    ("position", "text", "reason", "gap", "expected"),
    [
        (1, "1", None, 0, StatusClass.FINISHED),
        (12, "12", None, 2, StatusClass.LAPPED),
        (15, "15", "Engine", 3, StatusClass.DNF_MECHANICAL),  # classified late retirement
        (None, "DNF", "Collision", None, StatusClass.DNF_INCIDENT),
        (None, "DNF", None, None, StatusClass.DNF_OTHER),
        (None, "NC", None, None, StatusClass.NOT_CLASSIFIED),
        (None, "DSQ", "Illegal floor", None, StatusClass.DSQ),
        (None, "EX", None, None, StatusClass.DSQ),
        (None, "DNS", "Injury", None, StatusClass.DNS),
        (None, "DNQ", None, None, StatusClass.DNQ),
        (None, "DNPQ", None, None, StatusClass.DNPQ),
        (None, "DNF", "Withdrew", None, StatusClass.DNF_OTHER),
    ],
)
def test_result_status(
    position: int | None, text: str, reason: str | None, gap: int | None, expected: StatusClass
) -> None:
    assert result_status(position, text, reason, gap) is expected
