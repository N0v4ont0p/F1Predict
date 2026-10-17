from __future__ import annotations

import gzip
from datetime import UTC, date, datetime
from pathlib import Path

import httpx

from f1p.sources import meteostat as M
from f1p.sources.http import DEFAULT_LIMITS, HttpClient, RateLimit


def _client(tmp_path: Path, handler: object) -> HttpClient:
    fast = {h: RateLimit(per_second=1e6) for h in DEFAULT_LIMITS}
    return HttpClient(
        cache_dir=tmp_path / "http",
        transport=httpx.MockTransport(handler),  # type: ignore[arg-type]
        limits=fast,
    )


def test_parsers_convert_to_the_open_meteo_layout() -> None:
    hourly = gzip.compress(
        b"year,month,day,hour,temp,temp_source,prcp,prcp_source,wspd,wspd_source\n"
        b"2021,8,29,13,14.5,x,2.3,x,18.0,x\n2021,8,29,14,14.0,x,,,,\n"
    )
    h = M.parse_hourly(hourly)
    assert h["time"].to_list()[0] == datetime(2021, 8, 29, 13, tzinfo=UTC)
    assert h["precipitation"].to_list() == [2.3, None]
    assert h["wind_speed_10m"].to_list() == [18.0, None]
    d = M.parse_daily(gzip.compress(b"year,month,day,tmax,prcp\n2021,8,29,19.5,12.0\n"))
    assert d.row(0) == (date(2021, 8, 29), 12.0, 19.5)
    assert M.parse_hourly(b"").is_empty()


def test_missing_station_years_are_cached_as_empty(tmp_path: Path) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(404)

    ms = M.Meteostat(_client(tmp_path, handler), tmp_path / "ms", today=date(2026, 10, 17))
    assert ms._file("daily", "06490", 1999) == b""  # pyright: ignore[reportPrivateUsage]
    assert ms._file("daily", "06490", 1999) == b""  # pyright: ignore[reportPrivateUsage]
    assert len(calls) == 1
    assert ms._file("daily", "06490", 2030) == b""  # pyright: ignore[reportPrivateUsage]
    assert len(calls) == 1  # the future is never requested


def test_distance() -> None:
    assert abs(M.distance_km(50.437, 5.971, 50.437 + 0.045, 5.971) - 5.0) < 0.1
