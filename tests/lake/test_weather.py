from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx
import polars as pl
import pytest

from f1p.core.config import Settings
from f1p.lake import pipeline, weather
from f1p.lake.curate_f1db import Curated
from f1p.lake.weather import RaceSlot
from f1p.sources.http import DEFAULT_LIMITS, HttpClient, RateLimit


def _slot(**kw: object) -> RaceSlot:
    base: dict[str, object] = {
        "event_id": "2021-R12",
        "season": 2021,
        "circuit_id": "spa_francorchamps",
        "race_date": date(2021, 8, 29),
        "race_start_utc": None,
        "lat": 50.437,
        "lon": 5.971,
    }
    base.update(kw)
    return RaceSlot(**base)  # type: ignore[arg-type]


def test_race_window_uses_exact_start_when_known() -> None:
    start = datetime(2024, 7, 28, 13, 0, tzinfo=UTC)
    slot = _slot(race_start_utc=start, season=2024, race_date=date(2024, 7, 28))
    assert slot.window_start == start
    assert slot.window_end - slot.window_start == timedelta(hours=2)


def test_race_window_falls_back_to_local_afternoon_or_night() -> None:
    assert _slot().window_start == datetime(2021, 8, 29, 14, 0, tzinfo=UTC)  # lon 6 -> UTC+0
    sg = _slot(circuit_id="marina_bay", lat=1.29, lon=103.86, race_date=date(2019, 9, 22))
    assert sg.window_start == datetime(2019, 9, 22, 13, 0, tzinfo=UTC)  # 20:00 at UTC+7
    vegas = _slot(circuit_id="las_vegas", lat=36.1, lon=-115.2, race_date=date(2023, 11, 19))
    assert vegas.window_start == datetime(2023, 11, 19, 6, 0, tzinfo=UTC)  # Sat 22:00 PST
    bahrain = _slot(circuit_id="bahrain", lon=50.5, season=2013, race_date=date(2013, 4, 21))
    assert bahrain.window_start.hour == 11  # 14:00 local before the switch to night racing
    assert _slot(circuit_id="bahrain", lon=50.5, season=2014).window_start.hour == 15


def _hourly(start: date, days: int, wet_hours: set[int]) -> pl.DataFrame:
    times = [
        datetime.combine(start, datetime.min.time(), UTC) + timedelta(hours=h)
        for h in range(24 * days)
    ]
    return pl.DataFrame(
        {
            "time": times,
            "precipitation": [2.0 if h in wet_hours else 0.0 for h in range(len(times))],
            "temperature_2m": [18.0] * len(times),
            "wind_speed_10m": [5.0] * len(times),
        }
    )


def test_window_summary_and_wet_label() -> None:
    slot = _slot()
    hourly = _hourly(date(2021, 8, 28), 3, wet_hours={24 + 14, 24 + 15})
    summary = weather.window_summary(hourly, slot)
    assert summary == {"precip_mm": 4.0, "max_hourly_mm": 2.0, "temp_c": 18.0}
    assert summary is not None
    row = weather.observed_row(slot, summary)
    assert row["wet"] is True
    assert row["observed_at"] > slot.window_end
    dry = weather.window_summary(_hourly(date(2021, 8, 28), 3, {24 + 10}), slot)
    assert dry is not None
    assert dry["precip_mm"] == 0.0


def test_climatology_uses_only_prior_years() -> None:
    slot = _slot()
    days = pl.date_range(date(1980, 1, 1), date(2025, 12, 31), eager=True)
    # it rains every day from the race year on: climatology must not see that
    daily = pl.DataFrame(
        {
            "date": days,
            "precipitation_sum": [
                5.0 if d.year >= 2021 else (2.0 if d.day % 2 else 0.0) for d in days
            ],
            "temperature_2m_max": [20.0] * len(days),
        }
    )
    clim = weather.climatology(daily, slot)
    assert clim is not None
    assert 0.4 < clim["wet_prob"] < 0.6


def _client(tmp_path: Path, handler: Any) -> HttpClient:
    """A mock-backed client without the real per-host pacing (tests make many requests)."""
    fast = {h: RateLimit(per_second=1e6) for h in DEFAULT_LIMITS}
    return HttpClient(
        cache_dir=tmp_path / "http", transport=httpx.MockTransport(handler), limits=fast
    )


def _fake_api(request: httpx.Request) -> httpx.Response:
    if "open-meteo" not in str(request.url):
        return httpx.Response(404)  # no live-timing archive, no stations
    q = {k: v[0] for k, v in parse_qs(urlsplit(str(request.url)).query).items()}
    start, end = date.fromisoformat(q["start_date"]), date.fromisoformat(q["end_date"])
    n = (end - start).days + 1
    if "daily" in q:
        dates = [(start + timedelta(days=i)).isoformat() for i in range(n)]
        return httpx.Response(
            200,
            json={
                "daily": {
                    "time": dates,
                    "precipitation_sum": [1.5 if i % 3 == 0 else 0.0 for i in range(n)],
                    "temperature_2m_max": [22.0] * n,
                }
            },
        )
    times = [
        (datetime.combine(start, datetime.min.time()) + timedelta(hours=h)).strftime(
            "%Y-%m-%dT%H:%M"
        )
        for h in range(24 * n)
    ]
    wet = "historical-forecast" in str(request.url)
    return httpx.Response(
        200,
        json={
            "hourly": {
                "time": times,
                "precipitation": [0.8 if wet else 0.0] * len(times),
                "temperature_2m": [25.0] * len(times),
                "wind_speed_10m": [3.0] * len(times),
            }
        },
    )


def test_build_tables_from_api(curated: Curated, tmp_path: Path) -> None:
    http = _client(tmp_path, _fake_api)
    frames = weather.build(http, curated.tables, seasons=(2021, 2021), today=date(2026, 10, 15))
    fc, obs = frames["weather_forecast"], frames["weather_observed"]
    n_events = curated.tables["events"].filter(pl.col("season") == 2021).height
    assert obs.height == n_events
    assert not obs["wet"].any()
    assert set(fc["kind"].unique().to_list()) == {"climatology", "forecast"}
    f = fc.filter(pl.col("kind") == "forecast")
    assert f.height == n_events
    assert (f["wet_prob"] == 1.0).all()
    # stamped no later than the pre-race checkpoint, after the season began
    ev = curated.tables["events"].select("event_id", "race_date")
    j = f.join(ev, on="event_id")
    for at, race_date in j.select("observed_at", "race_date").iter_rows():
        assert at <= datetime.combine(race_date, datetime.min.time(), UTC)
    clim = fc.filter(pl.col("kind") == "climatology")
    latest = clim["observed_at"].max()
    assert isinstance(latest, datetime)
    assert latest <= datetime(2021, 1, 1, tzinfo=UTC)


def test_update_weather_commits_snapshot(
    isolated_settings: Settings, f1db_fixture: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import zipfile

    archive = tmp_path / "f1db-sqlite.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(f1db_fixture, "f1db.db")
    pipeline.update(isolated_settings, local_file=archive)
    http = _client(tmp_path, _fake_api)
    counts = pipeline.update_weather(
        isolated_settings, seasons=(2021, 2021), http=http, source="openmeteo"
    )
    assert counts["weather_observed"] > 0
    with pipeline.lake_for(isolated_settings, read_only=True) as lake:
        assert lake.read("weather_observed").height == counts["weather_observed"]
        assert lake.ingest_log()["source"].to_list()[-1] == "open-meteo+f1-timing+open-meteo"


def test_climatology_from_its_months_matches_the_full_history() -> None:
    slot = _slot()
    days = pl.date_range(date(1960, 1, 1), date(2025, 12, 31), eager=True)
    daily = pl.DataFrame(
        {
            "date": days,
            "precipitation_sum": [float((d.toordinal() * 7) % 5) for d in days],
            "temperature_2m_max": [float(d.toordinal() % 30) for d in days],
        }
    )
    months = weather.climatology_months(slot)
    assert len(months) <= 2 * weather.CLIMATOLOGY_YEARS
    wanted = {f"{y}-{m}" for y, m in months}
    keys = [f"{d.year}-{d.month}" for d in daily["date"]]
    subset = daily.filter(pl.Series([k in wanted for k in keys]))
    assert weather.climatology(subset, slot) == weather.climatology(daily, slot)


def test_build_fetches_climatology_a_month_at_a_time(curated: Curated, tmp_path: Path) -> None:
    spans: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        q = {k: v[0] for k, v in parse_qs(urlsplit(str(request.url)).query).items()}
        if "daily" in q:
            start, end = date.fromisoformat(q["start_date"]), date.fromisoformat(q["end_date"])
            spans.append((end - start).days + 1)
        return _fake_api(request)

    http = _client(tmp_path, handler)
    frames = weather.build(http, curated.tables, seasons=(2021, 2021), today=date(2026, 10, 15))
    assert spans
    assert max(spans) <= 31
    assert frames["weather_forecast"].filter(pl.col("kind") == "climatology").height > 0


def test_build_takes_the_reanalysis_from_era5_when_given(curated: Curated, tmp_path: Path) -> None:
    from f1p.sources import era5

    archive: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if "archive-api" in str(request.url):
            archive.append(str(request.url))
        return _fake_api(request)

    class FakeEra5(era5.Era5):
        def hourly_all(self, lat: float, lon: float, today: date | None = None) -> pl.DataFrame:
            times = pl.datetime_range(
                datetime(1985, 1, 1, tzinfo=UTC),
                datetime(2021, 12, 31, 23, tzinfo=UTC),
                interval="1h",
                eager=True,
            )
            return pl.DataFrame(
                {
                    "time": times,
                    "precipitation": [0.0] * len(times),
                    "temperature_2m": [18.0] * len(times),
                    "wind_speed_10m": [5.0] * len(times),
                }
            )

    frames = weather.build(
        _client(tmp_path, handler),
        curated.tables,
        seasons=(2021, 2021),
        today=date(2026, 10, 15),
        era5=FakeEra5(tmp_path / "era5"),
    )
    assert not archive  # no Open-Meteo archive request at all
    clim = frames["weather_forecast"].filter(pl.col("kind") == "climatology")
    assert clim.height > 0
    assert (clim["wet_prob"] == 0.0).all()
    assert frames["weather_observed"].height > 0
    assert (frames["weather_forecast"]["kind"] == "forecast").any()  # still Open-Meteo


# --------------------------------------------------------------------------- stations + track
def _stations_db(path: Path, circuits: pl.DataFrame) -> bytes:
    """A Meteostat stations.db with one station 5 km north of every circuit."""
    import sqlite3

    db = sqlite3.connect(path)
    db.execute(
        "CREATE TABLE stations (id TEXT PRIMARY KEY, country TEXT, region TEXT, "
        "latitude REAL, longitude REAL, elevation INT, timezone TEXT)"
    )
    db.execute(
        'CREATE TABLE inventory (station TEXT, provider TEXT, parameter TEXT, "start" DATE, '
        '"end" DATE, completeness INT)'
    )
    for i, (lat, lon) in enumerate(circuits.select("latitude", "longitude").iter_rows()):
        if lat is None or lon is None:
            continue
        sid = f"S{i:04d}"
        db.execute(
            "INSERT INTO stations VALUES (?,?,?,?,?,?,?)", (sid, "", "", lat + 0.045, lon, 0, "")
        )
        db.execute(
            "INSERT INTO inventory VALUES (?,?,?,?,?,?)", (sid, "x", "prcp", "1950-01-01", None, 90)
        )
    db.commit()
    db.close()
    return path.read_bytes()


def _gz_csv(rows: list[str]) -> bytes:
    import gzip

    return gzip.compress("\n".join(rows).encode())


def _meteostat_file(url: str, *, hourly_rain: bool = True) -> bytes:
    import calendar

    parts = url.rstrip("/").split("/")
    kind, year = parts[-3], int(parts[-2])
    days = [
        date(year, m, d)
        for m in range(1, 13)
        for d in range(1, calendar.monthrange(year, m)[1] + 1)
    ]
    if kind == "daily":
        rows = ["year,month,day,temp,tmin,tmax,prcp"]
        rows += [
            f"{d.year},{d.month},{d.day},15,10,20,{2.0 if d.day % 4 == 0 else 0.0}" for d in days
        ]
    else:
        rows = ["year,month,day,hour,temp,prcp,wspd"]
        rain = "1.0" if hourly_rain else ""
        rows += [f"{d.year},{d.month},{d.day},{h},17,{rain},12" for d in days for h in range(24)]
    return _gz_csv(rows)


def _fake_all(stations: bytes, *, hourly_rain: bool = True, track: bool = False) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("stations.db"):
            return httpx.Response(200, content=stations)
        if "data.meteostat.net" in url:
            return httpx.Response(200, content=_meteostat_file(url, hourly_rain=hourly_rain))
        if "livetiming" in url:
            return _fake_livetiming(url) if track else httpx.Response(404)
        return _fake_api(request)

    return handler


def _fake_livetiming(url: str) -> httpx.Response:
    import json

    if url.endswith("Index.json"):
        meetings = [
            {
                "Name": "Belgian Grand Prix",
                "Sessions": [
                    {
                        "Name": "Race",
                        "Type": "Race",
                        "Path": "2021/2021-08-29_Belgian_Grand_Prix/2021-08-29_Race/",
                    }
                ],
            }
        ]
        return httpx.Response(200, content=json.dumps({"Meetings": meetings}).encode())
    if url.endswith("SessionStatus.jsonStream"):
        lines = ['00:10:00.000{"Status":"Started"}', '01:10:00.000{"Status":"Finished"}']
    elif url.endswith("WeatherData.jsonStream"):
        # hot and dry before the start (10 min), cool and wet during the race
        lines = [
            f'00:{m:02d}:00.000{{"AirTemp":"{14 if m >= 10 else 30}","Rainfall":"{int(m >= 10)}"}}'
            for m in range(60)
        ]
    else:
        return httpx.Response(404)
    return httpx.Response(200, content="\n".join(lines).encode())


def test_track_summary_reads_the_race_window_only() -> None:
    status = [
        (100.0, {"Status": "Started"}),
        (900.0, {"Status": "Aborted"}),
        (1200.0, {"Status": "Started"}),
        (5000.0, {"Status": "Finished"}),
        (5600.0, {"Status": "Ends"}),
    ]
    feed = [(50.0, {"AirTemp": "40", "Rainfall": "0"})]  # before the start: ignored
    feed += [
        (float(t), {"AirTemp": "20.0", "Rainfall": "1" if t < 1100 else "0"})
        for t in range(100, 5601, 100)
    ]
    out = weather.track_summary(feed, status)
    assert out is not None
    assert out["temp_c"] == 20.0
    assert out["readings"] == 56
    assert 0.17 < out["rain_share"] < 0.19
    assert weather.track_summary([], status) is None


def test_observed_row_lets_the_track_decide_wet() -> None:
    slot = _slot()
    dry_station = {"precip_mm": 0.0, "max_hourly_mm": 0.0, "temp_c": 15.0}
    track = {"rain_share": 0.4, "temp_c": 13.0, "readings": 100}
    row = weather.observed_row(slot, dry_station, track)
    assert row["wet"] is True
    assert row["temp_c"] == 13.0
    assert row["precip_mm"] == 0.0
    proxy = weather.observed_row(slot, None, track)
    assert proxy["wet"] is True
    assert proxy["max_hourly_mm"] == weather.WET_HOURLY_MM
    assert proxy["precip_mm"] > 0
    assert weather.observed_row(slot, dry_station, {**track, "rain_share": 0.0})["wet"] is False


def test_build_from_stations_and_track(curated: Curated, tmp_path: Path) -> None:
    from f1p.sources.meteostat import Meteostat

    stations = _stations_db(tmp_path / "src.db", curated.tables["circuits"])
    seen: list[str] = []
    inner = _fake_all(stations, track=True)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return inner(request)

    http = _client(tmp_path, handler)
    frames = weather.build(
        http,
        curated.tables,
        seasons=(2021, 2021),
        today=date(2026, 10, 15),
        meteostat=Meteostat(http, tmp_path / "meteostat", today=date(2026, 10, 15)),
        raw_dir=tmp_path / "raw",
    )
    assert not [u for u in seen if "archive-api" in u]  # no Open-Meteo archive at all
    clim = frames["weather_forecast"].filter(pl.col("kind") == "climatology")
    n_events = curated.tables["events"].filter(pl.col("season") == 2021).height
    assert clim.height == n_events
    assert (clim["wet_prob"] - 0.25).abs().max() < 0.05  # rain every fourth day  # type: ignore[operator]
    obs = frames["weather_observed"]
    assert obs.height == n_events
    assert obs["wet"].all()  # the stations report 1 mm every hour
    spa = curated.tables["events"].filter(pl.col("race_date") == date(2021, 8, 29))["event_id"][0]
    row = obs.filter(pl.col("event_id") == spa).row(0, named=True)
    assert row["temp_c"] == 14.0  # from the track feed, not the station's 17
    # the station files are cached: a second build makes no Meteostat requests
    seen.clear()
    weather.build(
        http,
        curated.tables,
        seasons=(2021, 2021),
        today=date(2026, 10, 15),
        meteostat=Meteostat(http, tmp_path / "meteostat", today=date(2026, 10, 15)),
    )
    assert not [u for u in seen if "meteostat" in u]


def test_build_falls_back_to_open_meteo_without_station_rain(
    curated: Curated, tmp_path: Path
) -> None:
    from f1p.sources.meteostat import Meteostat

    stations = _stations_db(tmp_path / "src.db", curated.tables["circuits"])
    seen: list[str] = []
    inner = _fake_all(stations, hourly_rain=False)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return inner(request)

    http = _client(tmp_path, handler)
    frames = weather.build(
        http,
        curated.tables,
        seasons=(2021, 2021),
        today=date(2026, 10, 15),
        meteostat=Meteostat(http, tmp_path / "meteostat", today=date(2026, 10, 15)),
        raw_dir=tmp_path / "raw",
    )
    n_events = curated.tables["events"].filter(pl.col("season") == 2021).height
    archive = [u for u in seen if "archive-api" in u]
    assert len(archive) == n_events  # one hourly request per race, no climatology requests
    assert "daily" not in " ".join(archive)
    assert frames["weather_observed"].height == n_events


def test_update_weather_defaults_to_stations(
    isolated_settings: Settings, f1db_fixture: Path, tmp_path: Path
) -> None:
    import zipfile

    archive = tmp_path / "f1db-sqlite.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(f1db_fixture, "f1db.db")
    pipeline.update(isolated_settings, local_file=archive)
    with pipeline.lake_for(isolated_settings, read_only=True) as lake:
        circuits = lake.read("circuits")
    http = _client(tmp_path, _fake_all(_stations_db(tmp_path / "src.db", circuits)))
    counts = pipeline.update_weather(isolated_settings, seasons=(2021, 2021), http=http)
    assert counts["weather_observed"] > 0
    with pipeline.lake_for(isolated_settings, read_only=True) as lake:
        log = lake.ingest_log()
        assert log["source"].to_list()[-1] == "meteostat+f1-timing+open-meteo"
    assert (pipeline.raw_dir(isolated_settings) / "meteostat" / "stations.db").exists()
