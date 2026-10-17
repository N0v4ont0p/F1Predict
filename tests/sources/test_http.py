from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import httpx
import pytest

from f1p.sources.http import FetchError, HttpClient, RateLimit, _Limiter, clean_cache


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def _client(tmp_path: Path, handler, clock: FakeClock | None = None) -> HttpClient:  # type: ignore[no-untyped-def]
    clock = clock or FakeClock()
    return HttpClient(
        cache_dir=tmp_path / "http",
        transport=httpx.MockTransport(handler),
        sleep=clock.sleep,
        clock=clock,
    )


def test_caches_successful_responses(tmp_path: Path) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, json={"ok": True})

    with _client(tmp_path, handler) as http:
        first = http.get("https://api.example.com/x", {"b": 2, "a": 1})
        second = http.get("https://api.example.com/x", {"a": 1, "b": 2})
    assert first.json() == {"ok": True}
    assert not first.from_cache
    assert second.from_cache
    assert len(calls) == 1


def test_max_age_forces_refetch(tmp_path: Path) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, json=len(calls))

    with _client(tmp_path, handler) as http:
        http.get("https://api.example.com/x")
        assert http.get("https://api.example.com/x", max_age=timedelta(0)).json() == 2


def test_corrupt_cache_entry_is_refetched(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"fresh")

    with _client(tmp_path, handler) as http:
        http.get("https://api.example.com/x")
        body = next((tmp_path / "http").rglob("*.body"))
        body.write_bytes(b"tampered")
        resp = http.get("https://api.example.com/x")
    assert resp.body == b"fresh"
    assert not resp.from_cache


def test_retries_honour_retry_after(tmp_path: Path) -> None:
    responses = iter(
        [httpx.Response(429, headers={"Retry-After": "7"}), httpx.Response(200, json=1)]
    )
    clock = FakeClock()
    with _client(tmp_path, lambda r: next(responses), clock) as http:
        assert http.get("https://api.example.com/x").json() == 1
    assert 7.0 in clock.slept


def test_server_errors_back_off_then_give_up(tmp_path: Path) -> None:
    clock = FakeClock()
    with _client(tmp_path, lambda r: httpx.Response(503), clock) as http:
        http.max_retries = 2
        with pytest.raises(FetchError, match="gave up after 3 attempts"):
            http.get("https://api.example.com/x")
    assert [s for s in clock.slept if s >= 2] == [2.0, 4.0]


def test_client_errors_are_not_retried(tmp_path: Path) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(404)

    with _client(tmp_path, handler) as http, pytest.raises(FetchError) as exc:
        http.get("https://api.example.com/missing")
    assert exc.value.status == 404
    assert len(calls) == 1


def test_limiter_enforces_per_second_and_per_hour() -> None:
    clock = FakeClock()
    limiter = _Limiter(RateLimit(per_second=2, per_hour=3), clock=clock, sleep=clock.sleep)
    for _ in range(3):
        limiter.acquire()
    assert clock.now == pytest.approx(1.0)  # third call waited for the 1 s window
    limiter.acquire()  # fourth call must wait for the hourly window
    assert clock.now == pytest.approx(3600.0)


def test_download_streams_to_file(tmp_path: Path) -> None:
    with _client(tmp_path, lambda r: httpx.Response(200, content=b"x" * 10_000)) as http:
        out = http.download("https://files.example.com/a.zip", tmp_path / "a.zip")
    assert out.read_bytes() == b"x" * 10_000
    assert not (tmp_path / "a.zip.part").exists()


def test_query_string_in_the_url_is_kept(tmp_path: Path) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json=[1])

    client = HttpClient(cache_dir=tmp_path, transport=httpx.MockTransport(handler))
    client.get_json("https://example.org/v1/location?a=1&date>=2024-01-01")
    assert "a=1" in seen[0]
    assert "date" in seen[0]


def test_429_without_retry_after_waits_out_the_quota_window(tmp_path: Path) -> None:
    responses = iter([httpx.Response(429), httpx.Response(429), httpx.Response(200, json=1)])
    clock = FakeClock()
    with _client(tmp_path, lambda r: next(responses), clock) as http:
        assert http.get("https://api.example.com/x").json() == 1
    assert [s for s in clock.slept if s >= 2] == [15.0, 30.0]


def test_clean_cache_removes_partial_corrupt_and_selected(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        big = request.url.path.endswith("big")
        return httpx.Response(200, content=b"x" * (5000 if big else 10))

    with _client(tmp_path, handler) as http:
        for path in ("a", "b", "c", "big"):
            http.get(f"https://api.example.com/{path}")
        http.get("https://other.example.com/keep")
    cache = tmp_path / "http"
    small = sorted(p for p in cache.rglob("*.body") if p.stat().st_size < 1000 and "api." in str(p))
    bodies = small[:2]
    # a crash mid-write, a body without metadata, a tampered body
    (bodies[0].parent / "deadbeef.tmp").write_bytes(b"half")
    bodies[0].with_suffix(".meta.json").unlink()
    bodies[1].write_bytes(b"tampered")

    dry = clean_cache(cache, dry_run=True)
    assert dry.reasons == {"partial download": 2, "corrupt": 2}  # files: body + metadata
    assert len(list(cache.rglob("*.body"))) == 5  # nothing deleted on a dry run

    report = clean_cache(cache, hosts={"api.example.com"}, larger_than=1000)
    assert report.reasons == {
        "partial download": 2,
        "corrupt": 2,
        "api.example.com (obsolete large file)": 2,
    }
    left = {p.parent.parent.name for p in cache.rglob("*.body")}
    assert left == {"api.example.com", "other.example.com"}
    assert len(list(cache.rglob("*.body"))) == 2

    clean_cache(cache, everything=True)
    assert not list(cache.rglob("*.body"))
    assert not list(cache.rglob("*.meta.json"))


def test_exhausted_quota_skips_further_requests_but_serves_cache(tmp_path: Path) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/cached":
            return httpx.Response(200, json=1)
        return httpx.Response(429)

    clock = FakeClock()
    with _client(tmp_path, handler, clock) as http:
        assert http.get("https://api.example.com/cached").json() == 1
        http.max_retries = 1
        with pytest.raises(FetchError):
            http.get("https://api.example.com/a")
        n = len(calls)
        with pytest.raises(FetchError, match="quota used up"):
            http.get("https://api.example.com/b")
        assert len(calls) == n  # no request, no waiting
        assert http.get("https://api.example.com/cached").json() == 1


def test_a_host_that_keeps_failing_is_skipped_after_a_few_requests(tmp_path: Path) -> None:
    calls: list[str] = []
    ok = {"value": False}

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.host)
        if request.url.host == "good.example.com" or ok["value"]:
            return httpx.Response(200, json=1)
        raise httpx.ConnectTimeout("timed out")

    with _client(tmp_path, handler) as http:
        http.max_retries = 1
        # a success in between resets the count
        for path in ("a", "b"):
            with pytest.raises(FetchError):
                http.get(f"https://down.example.com/{path}")
        ok["value"] = True
        assert http.get("https://down.example.com/c").json() == 1
        ok["value"] = False
        for path in ("d", "e", "f"):
            with pytest.raises(FetchError, match="gave up"):
                http.get(f"https://down.example.com/{path}")
        n = len(calls)
        with pytest.raises(FetchError, match="unreachable earlier"):
            http.get("https://down.example.com/g")
        assert len(calls) == n  # skipped without a request or a wait
        assert http.get("https://good.example.com/x").json() == 1  # other hosts unaffected
