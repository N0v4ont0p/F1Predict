from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from f1p.lake.race_pace import lap_seconds, neutralised, pace_table, parse_laps
from f1p.sources.livetiming import read_stream


def _t(seconds: float) -> str:
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


def _stream(tmp_path: Path, name: str, rows: list[tuple[float, dict]]) -> Path:  # type: ignore[type-arg]
    path = tmp_path / f"{name}.jsonStream.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("﻿")
        for t, patch in rows:
            fh.write(f"{_t(t)}{json.dumps(patch)}\r\n")
    return path


def _race(
    cars: dict[str, float], laps: int, pit: dict[str, int] | None = None
) -> list[tuple[float, dict]]:  # type: ignore[type-arg]
    """Synthetic TimingData: each car laps at its own pace; optional pit stop (in-lap number)."""
    rows: list[tuple[float, dict]] = [(5.0, {"Lines": {c: {"InPit": True} for c in cars}})]  # type: ignore[type-arg]
    clock = dict.fromkeys(cars, 100.0)
    for lap in range(1, laps + 1):
        for car, pace in cars.items():
            secs = pace + (25.0 if pit and pit.get(car) == lap else 0.0)
            clock[car] += secs
            upd: dict = {"NumberOfLaps": lap}  # type: ignore[type-arg]
            if lap > 1:  # lap 1 has no time in the feed
                m, s = divmod(secs, 60)
                upd["LastLapTime"] = {"Value": f"{int(m)}:{s:06.3f}"}
            rows.append((clock[car], {"Lines": {car: upd}}))
            if pit and pit.get(car) == lap + 1:
                rows.append((clock[car] + 50, {"Lines": {car: {"InPit": True}}}))
            if pit and pit.get(car) == lap:
                rows.append((clock[car] + 1, {"Lines": {car: {"PitOut": True}}}))
    rows.sort(key=lambda r: r[0])
    return rows


def test_lap_seconds() -> None:
    assert lap_seconds("1:35.123") == pytest.approx(95.123)
    assert lap_seconds("59.5") == pytest.approx(59.5)
    assert lap_seconds("") is None
    assert lap_seconds("bad") is None


def test_parse_and_pace(tmp_path: Path) -> None:
    td = _stream(
        tmp_path, "TimingData", _race({"1": 90.0, "44": 90.9, "16": 91.8}, 20, pit={"44": 10})
    )
    laps = parse_laps(read_stream(td))
    by_car = {c: [lap for lap in laps if lap.car == c] for c in ("1", "44", "16")}
    assert len(by_car["1"]) == 19  # laps 2..20
    in_laps = [lap.lap for lap in by_car["44"] if lap.in_lap]
    out_laps = [lap.lap for lap in by_car["44"] if lap.out_lap]
    assert in_laps == [10]
    assert out_laps == [11]
    pace = pace_table(laps, []).sort("pace_deficit_pct")
    assert pace["car_number"].to_list() == ["1", "44", "16"]
    assert pace["pace_deficit_pct"][0] == 0.0
    assert pace["pace_deficit_pct"][1] == pytest.approx(1.0, abs=1e-6)
    assert pace["pace_deficit_pct"][2] == pytest.approx(2.0, abs=1e-6)
    # the pit-stop laps were excluded: the median is the true pace
    assert pace.filter(pace["car_number"] == "44")["median_lap_ms"][0] == 90900


def test_safety_car_laps_are_excluded(tmp_path: Path) -> None:
    ts = _stream(
        tmp_path,
        "TrackStatus",
        [(10.0, {"Status": "1"}), (500.0, {"Status": "4"}), (800.0, {"Status": "1"})],
    )
    intervals = neutralised(read_stream(ts))
    assert intervals == [(500.0, 800.0)]
    td = _stream(tmp_path, "TimingData", _race({"1": 90.0, "2": 91.0}, 15))
    laps = parse_laps(read_stream(td))
    clean = pace_table(laps, intervals)
    total = {c: sum(1 for lap in laps if lap.car == c) for c in ("1", "2")}
    assert all(n < total[c] for c, n in zip(clean["car_number"], clean["clean_laps"], strict=True))


def test_too_few_clean_laps_gives_no_row(tmp_path: Path) -> None:
    td = _stream(tmp_path, "TimingData", _race({"1": 90.0}, 4))
    assert pace_table(parse_laps(read_stream(td)), []).is_empty()


def _jolpica_api(cars: dict[str, float], laps: int, sc_laps: set[int], pit: dict[str, int]):  # type: ignore[no-untyped-def]
    """A MockTransport handler serving Jolpica laps / pitstops / results, paginated by 100."""
    import httpx

    timings = []
    for lap in range(1, laps + 1):
        for drv, pace in cars.items():
            secs = pace * (1.4 if lap in sc_laps else 1.0) + (25.0 if pit.get(drv) == lap else 0.0)
            m, s = divmod(secs, 60)
            timings.append((lap, {"driverId": drv, "time": f"{int(m)}:{s:06.3f}"}))

    def handler(request: httpx.Request) -> httpx.Response:
        q = dict(request.url.params)
        limit, offset = int(q.get("limit", 30)), int(q.get("offset", 0))
        path = request.url.path
        if path.endswith("laps.json"):
            page = timings[offset : offset + limit]
            by_lap: dict[int, list] = {}  # type: ignore[type-arg]
            for lap, t in page:
                by_lap.setdefault(lap, []).append(t)
            races = [{"Laps": [{"number": str(n), "Timings": ts} for n, ts in by_lap.items()]}]
            total = len(timings)
        elif path.endswith("pitstops.json"):
            races = [{"PitStops": [{"driverId": d, "lap": str(lap)} for d, lap in pit.items()]}]
            total = len(pit)
        else:
            races = [
                {
                    "Results": [
                        {"number": str(i + 1), "Driver": {"driverId": d}}
                        for i, d in enumerate(cars)
                    ]
                }
            ]
            total = len(cars)
        return httpx.Response(
            200, json={"MRData": {"total": str(total), "RaceTable": {"Races": races}}}
        )

    return handler


def test_jolpica_pace_with_pagination_pits_and_safety_car(tmp_path: Path) -> None:
    import httpx

    from f1p.lake.race_pace import jolpica_laps, jolpica_pace, neutral_laps_from_field
    from f1p.sources.http import HttpClient

    cars = {"a": 90.0, "b": 90.9, "c": 91.8, "d": 92.7, "e": 93.6, "f": 94.5}
    handler = _jolpica_api(cars, laps=30, sc_laps={12, 13, 14}, pit={"b": 20})
    http = HttpClient(cache_dir=tmp_path / "http", transport=httpx.MockTransport(handler))
    laps = jolpica_laps(http, 2022, 1)
    assert len(laps) == 30 * 6  # 180 rows across two pages, merged
    assert neutral_laps_from_field(laps) == {12, 13, 14}
    b = [lap for lap in laps if lap.car == "2"]
    assert [lap.lap for lap in b if lap.in_lap] == [20]
    assert [lap.lap for lap in b if lap.out_lap] == [21]
    pace = jolpica_pace(http, 2022, 1).sort("pace_deficit_pct")
    assert pace["car_number"].to_list() == ["1", "2", "3", "4", "5", "6"]
    assert pace["pace_deficit_pct"][1] == pytest.approx(1.0, abs=1e-6)  # SC and pit laps excluded
