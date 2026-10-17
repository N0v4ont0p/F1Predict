"""Race weather: climatology, pre-race forecasts and what actually happened (plan §6.1).

Weather enters the lake as two tables with different knowability:

``weather_forecast``
    * ``climatology`` — how often it rained on these dates at this circuit in the 30 years
      before the season (daily ERA5 reanalysis). Knowable at the start of the season, so it
      is available at every checkpoint, for every race back to 1950.
    * ``forecast`` — the archived short-range forecast for the race window (2016+), or the
      live forecast for an upcoming race. It is stamped at the pre-race checkpoint (C4):
      the archive holds short-lead runs, the forecast a team would read on race morning.

``weather_observed``
    ERA5 reanalysis over the race window — the label "was it a wet race?", public once the
    race has been run.

The race window is the scheduled start (known from 2024) plus two hours; earlier races use
the usual local start (14:00, or the circuit's night-race slot) at the circuit's solar
time offset.
"""

from __future__ import annotations

import calendar
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

import polars as pl

from f1p.core.logging import get_logger
from f1p.lake import schema as S
from f1p.lake import timing
from f1p.sources import openmeteo
from f1p.sources.http import FetchError, HttpClient

log = get_logger("lake.weather")

RACE_WINDOW = timedelta(hours=2)
DEFAULT_LOCAL_START = 14
# (local start hour, day shift) for evening and night races before exact start times exist.
NIGHT_RACES: dict[str, tuple[int, int]] = {
    "marina_bay": (20, 0),
    "jeddah": (20, 0),
    "lusail": (19, 0),
    "yas_marina": (17, 0),
    "las_vegas": (22, -1),  # Saturday night local; F1DB dates it Sunday
}
NIGHT_FROM: dict[str, int] = {"bahrain": 2014}  # Bahrain moved to a night race in 2014
WET_HOURLY_MM = 0.5  # a race is "wet" if any race-window hour reaches this much rain
WET_DAILY_MM = 1.0  # a climatology "rain day"
CLIMATOLOGY_YEARS = 30
CLIMATOLOGY_HALF_WINDOW = 7  # days either side of the race's calendar date


@dataclass(frozen=True, slots=True)
class RaceSlot:
    event_id: str
    season: int
    circuit_id: str
    race_date: date
    race_start_utc: datetime | None
    lat: float
    lon: float

    @property
    def window_start(self) -> datetime:
        if self.race_start_utc is not None:
            return self.race_start_utc
        hour, shift = NIGHT_RACES.get(self.circuit_id, (DEFAULT_LOCAL_START, 0))
        if self.circuit_id in NIGHT_FROM and self.season >= NIGHT_FROM[self.circuit_id]:
            hour = 18
        offset = round(self.lon / 15.0)  # solar time zone
        local = datetime.combine(self.race_date + timedelta(days=shift), time(hour), tzinfo=UTC)
        return local - timedelta(hours=offset)

    @property
    def window_end(self) -> datetime:
        return self.window_start + RACE_WINDOW


def race_slots(tables: dict[str, pl.DataFrame], seasons: tuple[int, int]) -> list[RaceSlot]:
    ev = (
        tables["events"]
        .filter(pl.col("season").is_between(*seasons))
        .join(tables["circuits"].select("circuit_id", "latitude", "longitude"), on="circuit_id")
        .sort("season", "round")
    )
    return [
        RaceSlot(
            r["event_id"],
            r["season"],
            r["circuit_id"],
            r["race_date"],
            r["race_start_utc"],
            r["latitude"],
            r["longitude"],
        )
        for r in ev.iter_rows(named=True)
        if r["latitude"] is not None and r["longitude"] is not None
    ]


# --------------------------------------------------------------------------- pure transforms
def window_summary(hourly: pl.DataFrame, slot: RaceSlot) -> dict[str, Any] | None:
    """Precipitation and temperature over the race window (``None`` if no data)."""
    w = hourly.filter(
        (pl.col("time") >= slot.window_start.replace(minute=0)) & (pl.col("time") < slot.window_end)
    ).drop_nulls("precipitation")
    if w.is_empty():
        return None
    temp = w["temperature_2m"].drop_nulls()
    return {
        "precip_mm": float(w["precipitation"].sum()),
        "max_hourly_mm": float(w["precipitation"].max()),  # type: ignore[arg-type]
        "temp_c": float(temp.mean()) if temp.len() else None,  # type: ignore[arg-type]
    }


def climatology_months(slot: RaceSlot) -> list[tuple[int, int]]:
    """The calendar months (year, month) that :func:`climatology` reads for ``slot``.

    Fetching only these, instead of every day since 1940, keeps each Open-Meteo request
    to about a month: the archive API weighs requests by the days they cover, and whole
    histories ran into its hourly and daily quotas.
    """
    out: set[tuple[int, int]] = set()
    for year in range(slot.season - CLIMATOLOGY_YEARS, slot.season):
        try:
            centre = slot.race_date.replace(year=year)
        except ValueError:  # 29 February
            centre = slot.race_date.replace(year=year, day=28)
        for d in (
            centre - timedelta(days=CLIMATOLOGY_HALF_WINDOW),
            centre + timedelta(days=CLIMATOLOGY_HALF_WINDOW),
        ):
            out.add((d.year, d.month))
    return sorted(out)


_EMPTY_DAILY = pl.DataFrame(schema={"date": pl.Date, **dict.fromkeys(openmeteo.DAILY, pl.Float64)})


def _climatology_daily(
    http: HttpClient,
    slot: RaceSlot,
    archive_last: date,
    months: dict[tuple[str, int, int], pl.DataFrame],
) -> pl.DataFrame:
    """Daily history for the months ``slot``'s climatology needs, one cached request each."""
    frames = []
    for year, month in climatology_months(slot):
        start = date(year, month, 1)
        if start > archive_last:
            continue
        key = (slot.circuit_id, year, month)
        if key not in months:
            end = date(year, month, calendar.monthrange(year, month)[1])
            months[key] = openmeteo.daily_history(
                http, slot.lat, slot.lon, start, min(end, archive_last)
            )
        frames.append(months[key])
    return pl.concat(frames) if frames else _EMPTY_DAILY


def climatology(daily: pl.DataFrame, slot: RaceSlot) -> dict[str, Any] | None:
    """Share of rain days around this calendar date in the years before the season."""
    first = slot.season - CLIMATOLOGY_YEARS
    rows = []
    for year in range(first, slot.season):
        try:
            centre = slot.race_date.replace(year=year)
        except ValueError:  # 29 February
            centre = slot.race_date.replace(year=year, day=28)
        lo = centre - timedelta(days=CLIMATOLOGY_HALF_WINDOW)
        hi = centre + timedelta(days=CLIMATOLOGY_HALF_WINDOW)
        rows.append(daily.filter(pl.col("date").is_between(lo, hi)))
    days = pl.concat(rows).drop_nulls("precipitation_sum") if rows else daily.head(0)
    if days.height < 30:
        return None
    return {
        "wet_prob": float((days["precipitation_sum"] >= WET_DAILY_MM).mean()),  # type: ignore[arg-type]
        "precip_mm": float(days["precipitation_sum"].mean()),  # type: ignore[arg-type]
        "temp_c": float(days["temperature_2m_max"].mean()),  # type: ignore[arg-type]
    }


def season_start(season: int) -> datetime:
    return datetime(season, 1, 1, tzinfo=UTC)


def forecast_row(slot: RaceSlot, summary: dict[str, Any]) -> dict[str, Any]:
    """A forecast for the race window, stamped at the pre-race checkpoint."""
    wet_prob = min(1.0, summary["max_hourly_mm"] / WET_HOURLY_MM) if summary else 0.0
    return {
        "event_id": slot.event_id,
        "kind": "forecast",
        "wet_prob": wet_prob,
        "precip_mm": summary["precip_mm"],
        "temp_c": summary["temp_c"],
        "observed_at": timing.pre_race_as_of(slot.race_date, slot.race_start_utc),
    }


def observed_row(slot: RaceSlot, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_id": slot.event_id,
        "window_start_utc": slot.window_start,
        "precip_mm": summary["precip_mm"],
        "max_hourly_mm": summary["max_hourly_mm"],
        "temp_c": summary["temp_c"],
        "wet": summary["max_hourly_mm"] >= WET_HOURLY_MM,
        "observed_at": slot.window_end + timing.RESULT_PUBLICATION_LAG,
    }


def frames(
    forecasts: Iterable[dict[str, Any]], observed: Iterable[dict[str, Any]]
) -> dict[str, pl.DataFrame]:
    out = {}
    for spec, rows in ((S.WEATHER_FORECAST, list(forecasts)), (S.WEATHER_OBSERVED, list(observed))):
        df = pl.DataFrame(rows, schema=spec.columns) if rows else pl.DataFrame(schema=spec.columns)
        df = S.conform(df, spec).sort(spec.primary_key)
        S.validate(df, spec)
        out[spec.name] = df
    return out


# --------------------------------------------------------------------------- fetching
def build(
    http: HttpClient,
    tables: dict[str, pl.DataFrame],
    *,
    seasons: tuple[int, int],
    today: date | None = None,
) -> dict[str, pl.DataFrame]:
    """Fetch and assemble both weather tables for every race in ``seasons``."""
    today = today or datetime.now(UTC).date()
    archive_last = today - timedelta(days=6)  # ERA5 lags real time by about five days
    slots = race_slots(tables, seasons)
    forecasts: list[dict[str, Any]] = []
    observed: list[dict[str, Any]] = []
    months: dict[tuple[str, int, int], pl.DataFrame] = {}
    for i, slot in enumerate(slots, start=1):
        log.info("weather %d/%d: %s", i, len(slots), slot.event_id)
        try:
            daily = _climatology_daily(http, slot, archive_last, months)
        except FetchError as exc:
            log.warning("climatology for %s unavailable: %s", slot.event_id, exc)
            daily = _EMPTY_DAILY
        clim = climatology(daily, slot)
        if clim is not None:
            forecasts.append(
                {
                    "event_id": slot.event_id,
                    "kind": "climatology",
                    **clim,
                    "observed_at": season_start(slot.season),
                }
            )
        day0 = slot.window_start.date() - timedelta(days=1)
        day1 = slot.window_end.date() + timedelta(days=1)
        try:
            if day1 <= archive_last:
                hourly = openmeteo.hourly(http, slot.lat, slot.lon, day0, day1)
                if (summary := window_summary(hourly, slot)) is not None:
                    observed.append(observed_row(slot, summary))
            if slot.race_date >= openmeteo.HISTORICAL_FORECAST_FIRST:
                upcoming = slot.race_date >= today
                if upcoming and slot.race_date > today + timedelta(days=15):
                    continue  # beyond the forecast horizon
                source = "forecast" if upcoming else "historical_forecast"
                hourly = openmeteo.hourly(http, slot.lat, slot.lon, day0, day1, source=source)
                if (summary := window_summary(hourly, slot)) is not None:
                    forecasts.append(forecast_row(slot, summary))
        except FetchError as exc:  # one missing race must not sink the table
            log.warning("weather for %s unavailable: %s", slot.event_id, exc)
    return frames(forecasts, observed)
