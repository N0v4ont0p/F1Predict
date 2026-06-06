"""Weather & environment enrichment via the Open-Meteo API (free, no API key).

We attach race-day weather to every event so the model and the Monte-Carlo engine can react
to the conditions that genuinely reshape an F1 race — air temperature (tyre working range),
relative humidity, rainfall (wet-race chaos), and wind.

Design choices
--------------
* **Offline-first.** Every lookup is cached to ``data/cache/weather.parquet`` keyed by
  ``(circuit_id, date)``. After the first fetch the platform runs fully offline.
* **Two data sources, one interface.**
    - *Historical* races (a past ``date``) use the Open-Meteo **archive** API, which returns
      measured daily values from a reanalysis dataset.
    - *Future* races, or any lookup when the network is unavailable, fall back to the
      researched per-circuit **climatology** in :mod:`features.tracks` — so weather features
      are always populated and predictions for 2026+ still get sensible conditions.
* **No key, polite usage.** Open-Meteo is free for non-commercial use and needs no key. We
  fetch one row per race (not per lap), so volume is tiny.

The public entry points are :func:`get_race_weather` (one race) and :func:`enrich_weather`
(annotate a whole master frame, with caching).
"""
from __future__ import annotations

from datetime import date as _date
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..config import Config, load_config
from ..features.tracks import track_climate, track_coords
from ..utils.logging import get_logger

log = get_logger()

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Columns produced for every race.
WEATHER_COLUMNS = ["wx_temp_c", "wx_humidity", "wx_rain_prob", "wx_wind_kmh", "wx_source"]


def _cache_path(cfg: Config) -> Path:
    cache = cfg.path("paths.cache_dir")
    cache.mkdir(parents=True, exist_ok=True)
    return cache / "weather.parquet"


def _load_cache(cfg: Config) -> pd.DataFrame:
    path = _cache_path(cfg)
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:  # noqa: BLE001 — a corrupt cache should never be fatal
            log.warning("[yellow]weather cache unreadable; rebuilding[/yellow]")
    return pd.DataFrame(columns=["circuit_id", "date", *WEATHER_COLUMNS])


def _save_cache(cfg: Config, df: pd.DataFrame) -> None:
    df.drop_duplicates(["circuit_id", "date"], keep="last").to_parquet(
        _cache_path(cfg), index=False
    )


def _parse_date(value) -> _date | None:
    if value in (None, "", "NaT"):
        return None
    try:
        return pd.to_datetime(value).date()
    except (ValueError, TypeError):
        return None


def _fetch_archive(lat: float, lon: float, day: _date, timeout: int) -> dict | None:
    """Fetch measured daily weather for a past date from the Open-Meteo archive."""
    import requests

    params = {
        "latitude": round(lat, 4), "longitude": round(lon, 4),
        "start_date": day.isoformat(), "end_date": day.isoformat(),
        "daily": "temperature_2m_max,temperature_2m_mean,precipitation_sum,"
                 "precipitation_hours,relative_humidity_2m_mean,wind_speed_10m_max",
        "timezone": "auto",
    }
    try:
        r = requests.get(ARCHIVE_URL, params=params, timeout=timeout)
        r.raise_for_status()
        daily = r.json().get("daily", {})
        if not daily.get("time"):
            return None
        temp = _first(daily.get("temperature_2m_mean")) or _first(daily.get("temperature_2m_max"))
        precip = _first(daily.get("precipitation_sum")) or 0.0
        precip_hours = _first(daily.get("precipitation_hours")) or 0.0
        humidity = _first(daily.get("relative_humidity_2m_mean"))
        wind = _first(daily.get("wind_speed_10m_max")) or 0.0
        # Convert a daily rainfall total into a wet-race probability proxy in [0,1].
        rain_prob = min(1.0, 0.15 * precip_hours + 0.05 * float(precip))
        return {
            "wx_temp_c": float(temp) if temp is not None else None,
            "wx_humidity": float(humidity) / 100.0 if humidity is not None else None,
            "wx_rain_prob": float(rain_prob),
            "wx_wind_kmh": float(wind),
            "wx_source": "open-meteo",
        }
    except Exception as exc:  # noqa: BLE001 — network/JSON issues fall back to climatology
        log.warning(f"[yellow]weather fetch failed ({day}): {exc}[/yellow]")
        return None


def _first(seq):
    if isinstance(seq, list) and seq:
        return seq[0]
    return None


def _climatology(circuit_id: str) -> dict:
    clim = track_climate(circuit_id)
    clim["wx_source"] = "climatology"
    return clim


def get_race_weather(
    circuit_id: str,
    date: object,
    cfg: Config | None = None,
    online: bool = True,
) -> dict:
    """Return weather for one race as a dict of :data:`WEATHER_COLUMNS`.

    Uses the cache first, then the Open-Meteo archive for past dates (if ``online``), and
    finally the circuit climatology. Never raises — always returns populated values.
    """
    cfg = cfg or load_config()
    day = _parse_date(date)
    cache = _load_cache(cfg)
    key_date = day.isoformat() if day else ""

    hit = cache[(cache["circuit_id"] == circuit_id) & (cache["date"] == key_date)]
    if len(hit):
        row = hit.iloc[0]
        return {c: row[c] for c in WEATHER_COLUMNS}

    result = None
    is_past = day is not None and day <= datetime.now(timezone.utc).date()
    if online and is_past:
        lat, lon = track_coords(circuit_id)
        if (lat, lon) != (0.0, 0.0):
            timeout = int(cfg.get("data.request_timeout", 25))
            result = _fetch_archive(lat, lon, day, timeout)

    if result is None or result.get("wx_temp_c") is None:
        result = _climatology(circuit_id)

    # Persist (even climatology, so future offline runs are instant and stable).
    new = {"circuit_id": circuit_id, "date": key_date, **result}
    cache = pd.concat([cache, pd.DataFrame([new])], ignore_index=True)
    _save_cache(cfg, cache)
    return {c: result.get(c) for c in WEATHER_COLUMNS}


def enrich_weather(
    df: pd.DataFrame,
    cfg: Config | None = None,
    online: bool = True,
    progress=None,
) -> pd.DataFrame:
    """Annotate a master frame with per-race weather columns (cached, de-duplicated).

    One API call (at most) per unique ``(circuit_id, date)`` pair. Returns a copy of ``df``
    with the :data:`WEATHER_COLUMNS` merged in.
    """
    cfg = cfg or load_config()
    df = df.copy()
    pairs = df[["circuit_id", "date"]].drop_duplicates().reset_index(drop=True)
    records = []
    n = len(pairs)
    for i, row in pairs.iterrows():
        wx = get_race_weather(str(row["circuit_id"]), row["date"], cfg, online=online)
        records.append({"circuit_id": row["circuit_id"], "date": row["date"], **wx})
        if progress is not None:
            progress((i + 1) / max(n, 1), f"weather {i + 1}/{n}")
    wx_df = pd.DataFrame(records)
    merged = df.merge(wx_df, on=["circuit_id", "date"], how="left")
    return merged
