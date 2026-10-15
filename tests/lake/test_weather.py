from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import httpx
import polars as pl
import pytest

from f1p.core.config import Settings
from f1p.lake import pipeline, weather
from f1p.lake.curate_f1db import Curated
from f1p.lake.weather import RaceSlot
from f1p.sources.http import HttpClient


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


def _fake_api(request: httpx.Request) -> httpx.Response:
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
    http = HttpClient(cache_dir=tmp_path / "http", transport=httpx.MockTransport(_fake_api))
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
    http = HttpClient(cache_dir=tmp_path / "http", transport=httpx.MockTransport(_fake_api))
    counts = pipeline.update_weather(isolated_settings, seasons=(2021, 2021), http=http)
    assert counts["weather_observed"] > 0
    with pipeline.lake_for(isolated_settings, read_only=True) as lake:
        assert lake.read("weather_observed").height == counts["weather_observed"]
        assert lake.ingest_log()["source"].to_list()[-1] == "open-meteo"
