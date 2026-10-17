"""Meteostat weather-station records (https://meteostat.net): free files, no key, no quota.

Meteostat collects the observations of national weather services (NOAA, DWD, Météo-France,
…) and serves them as static gzipped CSVs, one file per station and year:

* ``https://data.meteostat.net/hourly/<year>/<station>.csv.gz`` — hourly, UTC
* ``https://data.meteostat.net/daily/<year>/<station>.csv.gz``  — daily

The station list is a SQLite file (``stations.db``, ~30 MB, refreshed monthly). For a
circuit we try the nearest stations within :data:`MAX_KM` in turn and keep the first that
has precipitation for the period; it is usually an airport 5-30 km from the track.

Files are cached under ``data/raw/meteostat``: past years for good, the current year again
after a day. Data by Meteostat, generally CC BY 4.0 (https://dev.meteostat.net/license).
"""

from __future__ import annotations

import gzip
import io
import math
import sqlite3
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl

from f1p.core.logging import get_logger
from f1p.sources.http import FetchError, HttpClient

log = get_logger("sources.meteostat")

BASE = "https://data.meteostat.net"
STATIONS_URL = f"{BASE}/stations.db"
HOURLY_URL = BASE + "/hourly/{year}/{station}.csv.gz"
DAILY_URL = BASE + "/daily/{year}/{station}.csv.gz"
ATTRIBUTION = "Weather-station data by Meteostat (https://meteostat.net), CC BY 4.0."

MAX_KM = 60.0
CANDIDATES = 4  # nearest stations tried per circuit
STATIONS_MAX_AGE = timedelta(days=30)
CURRENT_MAX_AGE = timedelta(days=1)
MIN_CLIMATE_DAYS = 200  # a station's daily history must have this many rain readings

_HOURLY_SCHEMA = {
    "time": pl.Datetime("us", "UTC"),
    "precipitation": pl.Float64,
    "temperature_2m": pl.Float64,
    "wind_speed_10m": pl.Float64,
}
_DAILY_SCHEMA = {"date": pl.Date, "precipitation_sum": pl.Float64, "temperature_2m_max": pl.Float64}


@dataclass(frozen=True, slots=True)
class Station:
    id: str
    km: float
    start: date | None  # first and last day with precipitation, any provider
    end: date | None

    def covers(self, first: date, last: date) -> bool:
        starts_in_time = self.start is None or self.start <= last
        return starts_in_time and (self.end is None or self.end >= first)


def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(a))


def _float(col: str, frame: pl.DataFrame) -> pl.Expr:
    if col in frame.columns:
        return pl.col(col).cast(pl.Float64, strict=False)
    return pl.lit(None, dtype=pl.Float64)


def parse_hourly(raw: bytes) -> pl.DataFrame:
    """A Meteostat hourly file in the Open-Meteo layout (UTC, mm, °C, km/h)."""
    if not raw:
        return pl.DataFrame(schema=_HOURLY_SCHEMA)
    f = pl.read_csv(io.BytesIO(gzip.decompress(raw)), infer_schema_length=0)
    if f.is_empty():
        return pl.DataFrame(schema=_HOURLY_SCHEMA)
    return f.select(
        pl.datetime(
            pl.col("year").cast(pl.Int32),
            pl.col("month").cast(pl.Int32),
            pl.col("day").cast(pl.Int32),
            pl.col("hour").cast(pl.Int32),
            time_zone="UTC",
        ).alias("time"),
        _float("prcp", f).alias("precipitation"),
        _float("temp", f).alias("temperature_2m"),
        _float("wspd", f).alias("wind_speed_10m"),
    ).cast(_HOURLY_SCHEMA)  # pyright: ignore[reportArgumentType]


def parse_daily(raw: bytes) -> pl.DataFrame:
    """A Meteostat daily file as ``date, precipitation_sum, temperature_2m_max``."""
    if not raw:
        return pl.DataFrame(schema=_DAILY_SCHEMA)
    f = pl.read_csv(io.BytesIO(gzip.decompress(raw)), infer_schema_length=0)
    if f.is_empty():
        return pl.DataFrame(schema=_DAILY_SCHEMA)
    return f.select(
        pl.date(
            pl.col("year").cast(pl.Int32),
            pl.col("month").cast(pl.Int32),
            pl.col("day").cast(pl.Int32),
        ).alias("date"),
        _float("prcp", f).alias("precipitation_sum"),
        _float("tmax", f).alias("temperature_2m_max"),
    ).cast(_DAILY_SCHEMA)  # pyright: ignore[reportArgumentType]


class Meteostat:
    """Nearest-station hourly and daily weather, from cached Meteostat files."""

    def __init__(self, http: HttpClient, cache_dir: Path, today: date | None = None) -> None:
        self.http = http
        self.cache_dir = cache_dir
        self.today = today or datetime.now(UTC).date()
        self._nearby: dict[tuple[float, float], list[Station]] = {}
        self._parsed: dict[tuple[str, str, int], pl.DataFrame] = {}
        self._no_stations = False

    # ------------------------------------------------------------------ stations
    def stations_db(self) -> Path | None:
        """The cached station list, refreshed monthly; ``None`` if it cannot be had."""
        path = self.cache_dir / "stations.db"
        max_age = STATIONS_MAX_AGE.total_seconds()
        if self._no_stations or (path.exists() and time.time() - path.stat().st_mtime < max_age):
            return path if path.exists() else None
        try:
            log.info("Meteostat station list: downloading (about 30 MB, monthly)")
            body = self.http.get(STATIONS_URL, use_cache=False, store=False).body
        except FetchError as exc:
            self._no_stations = True  # do not retry for every race
            if path.exists():
                log.warning("Meteostat station list: refresh failed (%s); using the old copy", exc)
                return path
            log.warning("Meteostat station list unavailable (%s); no station data this run", exc)
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        tmp.write_bytes(body)
        tmp.replace(path)
        return path

    def nearby(self, lat: float, lon: float) -> list[Station]:
        """Stations with precipitation records within :data:`MAX_KM`, nearest first."""
        key = (round(lat, 3), round(lon, 3))
        if key in self._nearby:
            return self._nearby[key]
        dlat = MAX_KM / 111.0
        dlon = MAX_KM / (111.0 * max(math.cos(math.radians(lat)), 0.05))
        db_path = self.stations_db()
        if db_path is None:
            return []
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as db:
            rows = db.execute(
                """
                SELECT s.id, s.latitude, s.longitude, MIN(i.start), MAX(i.end)
                FROM stations s JOIN inventory i ON i.station = s.id AND i.parameter = 'prcp'
                WHERE s.latitude BETWEEN ? AND ? AND s.longitude BETWEEN ? AND ?
                GROUP BY s.id
                """,
                (lat - dlat, lat + dlat, lon - dlon, lon + dlon),
            ).fetchall()
        out = []
        for sid, slat, slon, start, end in rows:
            km = distance_km(lat, lon, slat, slon)
            if km <= MAX_KM:
                out.append(
                    Station(
                        str(sid),
                        km,
                        date.fromisoformat(start) if start else None,
                        date.fromisoformat(end) if end else None,
                    )
                )
        out.sort(key=lambda s: s.km)
        self._nearby[key] = out
        return out

    # ------------------------------------------------------------------ files
    def _file(self, kind: str, station: str, year: int) -> bytes:
        """One station-year file (empty bytes when Meteostat has none), cached."""
        path = self.cache_dir / kind / station / f"{year}.csv.gz"
        if path.exists():
            current = year >= self.today.year
            age = time.time() - path.stat().st_mtime
            if not current or age < CURRENT_MAX_AGE.total_seconds():
                return path.read_bytes()
        if year > self.today.year:
            return b""
        url = (HOURLY_URL if kind == "hourly" else DAILY_URL).format(year=year, station=station)
        try:
            body = self.http.get(url, use_cache=False, store=False).body
        except FetchError as exc:
            if exc.status != 404:
                raise
            body = b""  # no data for that station and year
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        tmp.write_bytes(body)
        tmp.replace(path)
        return body

    # ------------------------------------------------------------------ queries
    def hourly(self, lat: float, lon: float, start: date, end: date) -> pl.DataFrame:
        """Hourly weather (UTC) from the nearest station with rain readings for the period."""
        lo = datetime.combine(start, datetime.min.time(), UTC)
        hi = datetime.combine(end + timedelta(days=1), datetime.min.time(), UTC)
        tried = 0
        for st in self.nearby(lat, lon):
            if tried >= CANDIDATES:
                break
            if not st.covers(start, end):
                continue
            tried += 1
            frame = pl.concat(
                [self._year("hourly", st.id, y) for y in range(start.year, end.year + 1)]
            ).filter((pl.col("time") >= lo) & (pl.col("time") < hi))
            if frame["precipitation"].drop_nulls().len():
                log.info("Meteostat hourly %s,%s: station %s (%.0f km)", lat, lon, st.id, st.km)
                return frame
        return pl.DataFrame(schema=_HOURLY_SCHEMA)

    def daily(self, lat: float, lon: float, years: Iterable[int]) -> pl.DataFrame:
        """Daily rain and maximum temperature for ``years``, from the nearest station with a
        usable history (at least :data:`MIN_CLIMATE_DAYS` rain readings)."""
        years = sorted(set(years))
        if not years:
            return pl.DataFrame(schema=_DAILY_SCHEMA)
        first, last = date(years[0], 1, 1), date(years[-1], 12, 31)
        best = pl.DataFrame(schema=_DAILY_SCHEMA)
        tried = 0
        for st in self.nearby(lat, lon):
            if tried >= CANDIDATES:
                break
            if not st.covers(first, last):
                continue
            tried += 1
            frame = pl.concat([self._year("daily", st.id, y) for y in years])
            rain = frame["precipitation_sum"].drop_nulls().len()
            if rain >= MIN_CLIMATE_DAYS:
                log.info("Meteostat daily %s,%s: station %s (%.0f km)", lat, lon, st.id, st.km)
                return frame
            if rain > best["precipitation_sum"].drop_nulls().len():
                best = frame
        return best

    def _year(self, kind: str, station: str, year: int) -> pl.DataFrame:
        key = (kind, station, year)
        if key not in self._parsed:
            raw = self._file(kind, station, year)
            self._parsed[key] = parse_hourly(raw) if kind == "hourly" else parse_daily(raw)
        return self._parsed[key]
