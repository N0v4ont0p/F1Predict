from __future__ import annotations

import io
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from f1p.sources import era5


def _csv(start: date, end: date) -> str:
    rows = ["valid_time,u10,v10,t2m,tp,latitude,longitude"]
    t = datetime.combine(start, datetime.min.time())
    while t.date() <= end:
        rows.append(f"{t:%Y-%m-%d %H:%M:%S},3.0,4.0,293.15,0.0005,45.5,9.25")
        t += timedelta(hours=1)
    return "\n".join(rows) + "\n"


class FakeCds:
    def __init__(self, zipped: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self.zipped = zipped

    def __call__(self, dataset: str, request: dict[str, Any], target: str) -> None:
        assert dataset == era5.DATASET
        self.calls.append(request)
        a, b = request["date"][0].split("/")
        start, end = date.fromisoformat(a), date.fromisoformat(b)
        start = max(start, end - timedelta(days=40))  # keep the fake small
        body = _csv(start, end)
        if self.zipped:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as zf:
                zf.writestr("data.csv", body)
            Path(target).write_bytes(buf.getvalue())
        else:
            Path(target).write_text(body)


@pytest.mark.parametrize("zipped", [False, True])
def test_units_daily_and_cache(tmp_path: Path, zipped: bool) -> None:
    cds = FakeCds(zipped)
    src = era5.Era5(tmp_path, retrieve=cds)
    hourly = src.hourly_all(45.5, 9.25, today=date(2026, 10, 17))
    assert len(cds.calls) == 2  # history to last year + this year
    assert cds.calls[0]["date"] == ["1940-01-01/2025-12-31"]
    assert cds.calls[1]["date"] == ["2026-01-01/2026-10-11"]
    row = hourly.row(0, named=True)
    assert row["temperature_2m"] == pytest.approx(20.0)
    assert row["precipitation"] == pytest.approx(0.5)
    assert row["wind_speed_10m"] == pytest.approx(18.0)  # 5 m/s
    assert hourly["time"].dtype == pl.Datetime("us", "UTC")

    daily = src.daily(45.5, 9.25)
    full = daily.filter(pl.col("date") == date(2026, 10, 1)).row(0, named=True)
    assert full["precipitation_sum"] == pytest.approx(12.0)
    assert full["temperature_2m_max"] == pytest.approx(20.0)

    again = era5.Era5(tmp_path, retrieve=cds)  # a new run: both files cached and fresh
    again.hourly_all(45.5, 9.25, today=date(2026, 10, 17))
    assert len(cds.calls) == 2


def test_cds_failure_is_an_era5_error(tmp_path: Path) -> None:
    def broken(dataset: str, request: dict[str, Any], target: str) -> None:
        raise RuntimeError("required licences not accepted")

    with pytest.raises(era5.Era5Error, match="licences"):
        era5.Era5(tmp_path, retrieve=broken).hourly_all(1.0, 2.0)
