"""ERA5 reanalysis straight from the Copernicus Climate Data Store (CDS).

Uses the *ERA5 hourly time-series* dataset (``reanalysis-era5-single-levels-timeseries``): one
request returns the hourly history of a single point as CSV, 1940 to about a week ago, with no
per-day quota. It is the same reanalysis Open-Meteo's archive serves, at ERA5's own 0.25° grid.
ERA5 has no forecasts, so pre-race and live forecasts still come from Open-Meteo.

Needs a free CDS account, the ERA5 licence accepted on the dataset page, and the API key in
``~/.cdsapirc`` (``url: https://cds.climate.copernicus.eu/api`` and ``key: <your key>``) or in
``CDSAPI_URL`` / ``CDSAPI_KEY``. Contains modified Copernicus Climate Change Service
information; neither the European Commission nor ECMWF is responsible for its use.

Downloads are cached under ``data/raw/era5``: each circuit's history up to the end of last year
once for good, and the current year again when the cached copy is a few days old.
"""

from __future__ import annotations

import io
import os
import time
import zipfile
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from f1p.core.logging import get_logger

log = get_logger("sources.era5")

DATASET = "reanalysis-era5-single-levels-timeseries"
VARIABLES = (
    "2m_temperature",
    "total_precipitation",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
)
FIRST = date(1940, 1, 1)
CURRENT_MAX_AGE = timedelta(days=3)
ATTRIBUTION = "Contains modified Copernicus Climate Change Service information (ERA5)."


class Era5Error(RuntimeError):
    """A CDS request failed (not configured, licence not accepted, or the request errored)."""


# a retrieve function: (dataset, request, target path) -> None; cdsapi's Client.retrieve
Retrieve = Callable[[str, dict[str, Any], str], Any]


def configured() -> bool:
    """True when ``cdsapi`` is installed and a CDS key is set up."""
    try:
        import cdsapi  # noqa: F401  # pyright: ignore[reportUnusedImport]
    except ImportError:
        return False
    if os.environ.get("CDSAPI_KEY"):
        return True
    rc = Path(os.environ.get("CDSAPI_RC", Path.home() / ".cdsapirc"))
    return rc.exists()


def _default_retrieve() -> Retrieve:
    import cdsapi  # pyright: ignore[reportMissingImports]

    client = cdsapi.Client(quiet=True, progress=False)
    return client.retrieve


def _read_csv(raw: bytes) -> pl.DataFrame:
    """The CDS answer: a CSV, or a zip holding one or more CSVs."""
    if raw[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            parts = [
                pl.read_csv(zf.read(n), infer_schema_length=10_000)
                for n in zf.namelist()
                if n.endswith(".csv")
            ]
        frame = parts[0]
        for p in parts[1:]:  # one file per variable group: join on time
            frame = frame.join(p.select(["valid_time", *p.columns[1:]]), on="valid_time")
        return frame
    return pl.read_csv(raw, infer_schema_length=10_000)


def to_hourly(raw: pl.DataFrame) -> pl.DataFrame:
    """CDS columns (K, m, m/s) to the Open-Meteo layout and units (°C, mm, km/h, UTC)."""
    cols = {c.lower(): c for c in raw.columns}

    def col(*names: str) -> pl.Expr:
        for n in names:
            if n in cols:
                return pl.col(cols[n]).cast(pl.Float64)
        return pl.lit(None, dtype=pl.Float64)

    time_col = cols.get("valid_time") or cols.get("time") or raw.columns[0]
    out = raw.select(
        pl.col(time_col).cast(pl.String).str.slice(0, 16).alias("time"),
        (col("tp", "total_precipitation") * 1000).alias("precipitation"),
        (col("t2m", "2m_temperature") - 273.15).alias("temperature_2m"),
        ((col("u10") ** 2 + col("v10") ** 2).sqrt() * 3.6).alias("wind_speed_10m"),
    )
    return (
        out.with_columns(
            pl.col("time")
            .str.replace("T", " ")
            .str.to_datetime("%Y-%m-%d %H:%M", time_zone="UTC")
            .alias("time")
        )
        .with_columns(pl.col("precipitation").clip(lower_bound=0.0))
        .sort("time")
        .unique("time", keep="last", maintain_order=True)
    )


class Era5:
    """Point histories from the CDS, cached per location and period."""

    def __init__(self, cache_dir: Path, retrieve: Retrieve | None = None) -> None:
        self.cache_dir = cache_dir
        self._retrieve = retrieve
        self._memo: dict[tuple[float, float], pl.DataFrame] = {}

    def _fetch(self, lat: float, lon: float, start: date, end: date, path: Path) -> None:
        retrieve = self._retrieve or _default_retrieve()
        self._retrieve = retrieve
        request = {
            "variable": list(VARIABLES),
            "location": {"latitude": lat, "longitude": lon},
            "date": [f"{start.isoformat()}/{end.isoformat()}"],
            "data_format": "csv",
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        t0 = time.monotonic()
        log.info("ERA5 %s,%s %s to %s: requesting from the CDS", lat, lon, start, end)
        try:
            retrieve(DATASET, request, str(tmp))
        except Exception as exc:  # cdsapi raises plain exceptions (HTTP, licence, queue)
            raise Era5Error(f"CDS request for {lat},{lon} failed: {exc}") from exc
        tmp.replace(path)
        log.info(
            "ERA5 %s,%s: %.0f KB in %.0fs",
            lat,
            lon,
            path.stat().st_size / 1024,
            time.monotonic() - t0,
        )

    def hourly_all(self, lat: float, lon: float, today: date | None = None) -> pl.DataFrame:
        """Hourly history of one point, 1940 to the latest ERA5 day (cached)."""
        lat, lon = round(lat, 2), round(lon, 2)
        if (lat, lon) in self._memo:
            return self._memo[(lat, lon)]
        today = today or datetime.now(UTC).date()
        last = today - timedelta(days=6)  # ERA5 lags real time by about five days
        year_start = date(last.year, 1, 1)
        base = self.cache_dir / f"{lat:+.2f}_{lon:+.2f}"
        history = base.with_name(base.name + f"_{FIRST.year}-{last.year - 1}.csv")
        current = base.with_name(base.name + f"_{last.year}.csv")
        if not history.exists():
            self._fetch(lat, lon, FIRST, year_start - timedelta(days=1), history)
        stale = (
            not current.exists()
            or datetime.now(UTC).timestamp() - current.stat().st_mtime
            > CURRENT_MAX_AGE.total_seconds()
        )
        if stale:
            self._fetch(lat, lon, year_start, last, current)
        frame = pl.concat([to_hourly(_read_csv(p.read_bytes())) for p in (history, current)])
        self._memo[(lat, lon)] = frame
        return frame

    def hourly(self, lat: float, lon: float, start: date, end: date) -> pl.DataFrame:
        frame = self.hourly_all(lat, lon)
        lo = datetime.combine(start, datetime.min.time(), UTC)
        hi = datetime.combine(end + timedelta(days=1), datetime.min.time(), UTC)
        return frame.filter((pl.col("time") >= lo) & (pl.col("time") < hi))

    def daily(self, lat: float, lon: float) -> pl.DataFrame:
        """Daily precipitation sum and maximum temperature (UTC days), like Open-Meteo's."""
        return (
            self.hourly_all(lat, lon)
            .group_by(pl.col("time").dt.date().alias("date"))
            .agg(
                pl.col("precipitation").sum().alias("precipitation_sum"),
                pl.col("temperature_2m").max().alias("temperature_2m_max"),
            )
            .sort("date")
        )
