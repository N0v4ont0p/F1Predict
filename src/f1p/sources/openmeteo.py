"""Open-Meteo weather connector (plan §6.1): reanalysis history and archived forecasts.

* **Archive** (ERA5 reanalysis, 1940 → a few days ago): what the weather actually was.
* **Historical forecast** (archived short-range model output, 2016 →): what forecasters
  expected in the hours before a race, for the pre-race checkpoint.
* **Forecast** (live, 16 days ahead): for upcoming races.

Weather data by Open-Meteo.com, CC BY 4.0. Every response goes through the polite
:class:`~f1p.sources.http.HttpClient` (rate limits, retries, raw cache).
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import polars as pl

from f1p.sources.http import HttpClient

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ATTRIBUTION = "Weather data by Open-Meteo.com (https://open-meteo.com), licensed CC BY 4.0."

HOURLY = ("precipitation", "temperature_2m", "wind_speed_10m")
DAILY = ("precipitation_sum", "temperature_2m_max")
HISTORICAL_FORECAST_FIRST = date(2016, 1, 1)


def _base(lat: float, lon: float, start: date, end: date) -> dict[str, Any]:
    return {
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "timezone": "GMT",
    }


def _hourly_frame(payload: dict[str, Any], variables: tuple[str, ...]) -> pl.DataFrame:
    h = payload.get("hourly") or {}
    times = h.get("time") or []
    data: dict[str, Any] = {"time": times}
    for v in variables:
        data[v] = h.get(v) or [None] * len(times)
    return pl.DataFrame(
        data, schema={"time": pl.String, **dict.fromkeys(variables, pl.Float64)}
    ).with_columns(pl.col("time").str.to_datetime("%Y-%m-%dT%H:%M", time_zone="UTC").alias("time"))


def hourly(
    http: HttpClient,
    lat: float,
    lon: float,
    start: date,
    end: date,
    *,
    source: str = "archive",
    variables: tuple[str, ...] = HOURLY,
) -> pl.DataFrame:
    """Hourly series (UTC) from the ``archive``, ``historical_forecast`` or ``forecast`` API."""
    url = {
        "archive": ARCHIVE_URL,
        "historical_forecast": HISTORICAL_FORECAST_URL,
        "forecast": FORECAST_URL,
    }[source]
    params = _base(lat, lon, start, end) | {"hourly": ",".join(variables)}
    # Past data is immutable; live forecasts change every model run.
    max_age = timedelta(hours=1) if source == "forecast" else None
    return _hourly_frame(http.get_json(url, params, max_age=max_age), variables)


def daily_history(http: HttpClient, lat: float, lon: float, start: date, end: date) -> pl.DataFrame:
    """Daily reanalysis (precipitation sum, max temperature) for climatology."""
    params = _base(lat, lon, start, end) | {"daily": ",".join(DAILY)}
    d = http.get_json(ARCHIVE_URL, params).get("daily") or {}
    times = d.get("time") or []
    return pl.DataFrame(
        {"date": times, **{v: d.get(v) or [None] * len(times) for v in DAILY}},
        schema={"date": pl.String, **dict.fromkeys(DAILY, pl.Float64)},
    ).with_columns(pl.col("date").str.to_date("%Y-%m-%d"))
