"""Polite HTTP: per-host rate limits, retries that honour ``Retry-After``, and a raw cache.

The raw cache is content-addressed by request (method + URL + sorted params). Each entry is a
pair of files: the body and a JSON sidecar recording the URL, status, fetch time and SHA-256.
Entries for immutable history are kept forever; entries for data that can still change
(e.g. the current season) carry a ``max_age`` after which they are refetched.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlsplit

import httpx

from f1p import __version__
from f1p.core.logging import get_logger

log = get_logger("sources.http")

USER_AGENT = f"f1predict/{__version__} (+https://github.com/N0v4ont0p/F1Predict)"
RETRY_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})


@dataclass(frozen=True, slots=True)
class RateLimit:
    """At most ``per_second`` requests in any 1 s window and ``per_hour`` in any hour."""

    per_second: float = 4.0
    per_hour: int | None = None


# Published or community-reported limits (plan §6.1); unknown hosts get a gentle default.
DEFAULT_LIMITS: dict[str, RateLimit] = {
    "api.jolpi.ca": RateLimit(per_second=4, per_hour=500),
    "api.openf1.org": RateLimit(per_second=3, per_hour=1800),
    "archive-api.open-meteo.com": RateLimit(per_second=5, per_hour=5000),
    "api.open-meteo.com": RateLimit(per_second=5, per_hour=5000),
    "historical-forecast-api.open-meteo.com": RateLimit(per_second=5, per_hour=5000),
    "livetiming.formula1.com": RateLimit(per_second=3),
    "ensemble-api.open-meteo.com": RateLimit(per_second=5, per_hour=5000),
}
FALLBACK_LIMIT = RateLimit(per_second=2)
RATE_LIMIT_WAIT = 15.0  # first wait after an HTTP 429 without Retry-After; grows to 60 s


class _Limiter:
    """Sliding-window limiter for one host (thread-safe)."""

    def __init__(self, limit: RateLimit, clock: Any = time.monotonic, sleep: Any = time.sleep):
        self.limit = limit
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._recent: deque[float] = deque()
        self._hour: deque[float] = deque()

    def acquire(self) -> None:
        with self._lock:
            while True:
                now = self._clock()
                while self._recent and now - self._recent[0] >= 1.0:
                    self._recent.popleft()
                while self._hour and now - self._hour[0] >= 3600.0:
                    self._hour.popleft()
                waits = []
                if len(self._recent) >= max(1, int(self.limit.per_second)):
                    waits.append(1.0 - (now - self._recent[0]))
                if self.limit.per_hour is not None and len(self._hour) >= self.limit.per_hour:
                    waits.append(3600.0 - (now - self._hour[0]))
                if not waits:
                    self._recent.append(now)
                    self._hour.append(now)
                    return
                self._sleep(max(waits))


class FetchError(RuntimeError):
    """A request failed permanently (non-retryable status or retries exhausted)."""

    def __init__(self, url: str, status: int | None, detail: str):
        super().__init__(f"{url}: {detail}")
        self.url = url
        self.status = status


@dataclass(frozen=True, slots=True)
class CachedResponse:
    url: str
    status: int
    fetched_at: datetime
    sha256: str
    body: bytes
    from_cache: bool

    def json(self) -> Any:
        return json.loads(self.body)


def request_key(url: str, params: Mapping[str, Any] | None = None) -> str:
    query = urlencode(sorted((k, str(v)) for k, v in (params or {}).items()))
    return hashlib.sha256(f"GET {url}?{query}".encode()).hexdigest()


@dataclass
class HttpClient:
    """Shared client used by every connector."""

    cache_dir: Path
    limits: dict[str, RateLimit] = field(default_factory=lambda: dict(DEFAULT_LIMITS))
    max_retries: int = 5
    backoff_base: float = 2.0
    timeout: float = 60.0
    transport: httpx.BaseTransport | None = None
    sleep: Any = time.sleep
    clock: Any = time.monotonic
    _limiters: dict[str, _Limiter] = field(default_factory=dict, init=False, repr=False)
    # hosts whose quota ran out (HTTP 429 through every retry): for the rest of this client's
    # life their uncached requests fail at once instead of waiting minutes each
    _exhausted: set[str] = field(default_factory=set, init=False, repr=False)
    _client: httpx.Client | None = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------ lifecycle
    def __enter__(self) -> HttpClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                timeout=self.timeout,
                follow_redirects=True,
                headers={"User-Agent": USER_AGENT},
                transport=self.transport,
            )
        return self._client

    # ------------------------------------------------------------------ cache
    def _paths(self, url: str, key: str) -> tuple[Path, Path]:
        host = urlsplit(url).hostname or "unknown"
        base = self.cache_dir / host / key[:2] / key
        return base.with_suffix(".body"), base.with_suffix(".meta.json")

    def _read_cache(self, url: str, key: str, max_age: timedelta | None) -> CachedResponse | None:
        body_path, meta_path = self._paths(url, key)
        if not (body_path.exists() and meta_path.exists()):
            return None
        meta = json.loads(meta_path.read_text())
        fetched_at = datetime.fromisoformat(meta["fetched_at"])
        if max_age is not None and datetime.now(UTC) - fetched_at > max_age:
            return None
        body = body_path.read_bytes()
        if hashlib.sha256(body).hexdigest() != meta["sha256"]:
            log.warning("cache entry corrupt, refetching: %s", url)
            return None
        return CachedResponse(url, meta["status"], fetched_at, meta["sha256"], body, True)

    def _write_cache(self, url: str, key: str, status: int, body: bytes) -> CachedResponse:
        body_path, meta_path = self._paths(url, key)
        body_path.parent.mkdir(parents=True, exist_ok=True)
        fetched_at = datetime.now(UTC)
        digest = hashlib.sha256(body).hexdigest()
        tmp = body_path.with_suffix(".tmp")
        tmp.write_bytes(body)
        tmp.replace(body_path)
        meta = {
            "url": url,
            "status": status,
            "fetched_at": fetched_at.isoformat(),
            "sha256": digest,
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        return CachedResponse(url, status, fetched_at, digest, body, False)

    # ------------------------------------------------------------------ fetch
    def _limiter(self, url: str) -> _Limiter:
        host = urlsplit(url).hostname or "unknown"
        if host not in self._limiters:
            self._limiters[host] = _Limiter(
                self.limits.get(host, FALLBACK_LIMIT), clock=self.clock, sleep=self.sleep
            )
        return self._limiters[host]

    def get(
        self,
        url: str,
        params: Mapping[str, Any] | None = None,
        *,
        max_age: timedelta | None = None,
        use_cache: bool = True,
        store: bool = True,
    ) -> CachedResponse:
        """GET with caching. ``max_age=None`` keeps a cached entry forever; ``store=False``
        skips writing the body to the cache (for large payloads kept elsewhere)."""
        key = request_key(url, params)
        if use_cache and (hit := self._read_cache(url, key, max_age)) is not None:
            return hit
        host = urlsplit(url).hostname or "unknown"
        if host in self._exhausted:
            raise FetchError(url, 429, "quota used up earlier in this run; skipped")
        last_detail = "no attempt made"
        status: int | None = None
        for attempt in range(self.max_retries + 1):
            self._limiter(url).acquire()
            t0 = time.monotonic()
            try:
                # an empty params dict would replace a query string already in ``url``
                resp = self.client.get(url, params=dict(params) if params else None)
            except httpx.TransportError as exc:
                status, last_detail = None, f"transport error: {exc!r}"
            else:
                status = resp.status_code
                if status == 200:
                    log.info(
                        "fetched %s (%.0f KB, %.1fs)",
                        url,
                        len(resp.content) / 1024,
                        time.monotonic() - t0,
                    )
                    if not store:
                        body = resp.content
                        digest = hashlib.sha256(body).hexdigest()
                        return CachedResponse(url, status, datetime.now(UTC), digest, body, False)
                    return self._write_cache(url, key, status, resp.content)
                last_detail = f"HTTP {status}"
                if status not in RETRY_STATUSES:
                    raise FetchError(url, status, last_detail)
                retry_after = _retry_after_seconds(resp.headers.get("Retry-After"))
                if retry_after is not None and attempt < self.max_retries:
                    log.info("%s → %s; waiting %.0fs (Retry-After)", url, status, retry_after)
                    self.sleep(retry_after)
                    continue
            if attempt < self.max_retries:
                delay = self.backoff_base ** (attempt + 1)
                if status == 429:  # a quota window (e.g. Open-Meteo's per minute): wait it out
                    delay = max(delay, min(RATE_LIMIT_WAIT * (attempt + 1), 60.0))
                log.info("%s failed (%s); retry %d in %.0fs", url, last_detail, attempt + 1, delay)
                self.sleep(delay)
        if status == 429:
            self._exhausted.add(host)
            log.warning(
                "%s: quota used up; skipping its uncached requests for the rest of this run "
                "(run the step again later, e.g. tomorrow, to fill them in)",
                host,
            )
        raise FetchError(
            url, status, f"gave up after {self.max_retries + 1} attempts: {last_detail}"
        )

    def get_json(self, url: str, params: Mapping[str, Any] | None = None, **kw: Any) -> Any:
        return self.get(url, params, **kw).json()

    def download(self, url: str, dest: Path) -> Path:
        """Stream a large file to ``dest`` (no body cache; the file itself is the artefact)."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        self._limiter(url).acquire()
        with self.client.stream("GET", url) as resp:
            if resp.status_code != 200:
                raise FetchError(url, resp.status_code, f"HTTP {resp.status_code}")
            with tmp.open("wb") as fh:
                for chunk in resp.iter_bytes(1 << 20):
                    fh.write(chunk)
        tmp.replace(dest)
        return dest


def _retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        from email.utils import parsedate_to_datetime

        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, (when - datetime.now(UTC)).total_seconds())


# ---------------------------------------------------------------------- cache cleaning
@dataclass
class CleanReport:
    files: int = 0
    bytes: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    def add(self, path: Path, reason: str) -> None:
        self.files += 1
        self.bytes += path.stat().st_size if path.exists() else 0
        self.reasons[reason] = self.reasons.get(reason, 0) + 1


def clean_cache(
    cache_dir: Path,
    *,
    hosts: set[str] | None = None,
    larger_than: int | None = None,
    everything: bool = False,
    dry_run: bool = False,
) -> CleanReport:
    """Delete raw-cache files under ``cache_dir`` (``<host>/<xx>/<key>.body|.meta.json``).

    Always removes partial downloads: ``.tmp`` files, a body without its metadata (or the
    reverse) and bodies whose checksum no longer matches. With ``hosts``, also every entry of
    those hosts, limited to bodies over ``larger_than`` bytes when given; with ``everything``,
    every entry. A deleted entry is simply fetched again when next needed.
    """
    report = CleanReport()
    if not cache_dir.exists():
        return report

    def drop(path: Path, reason: str) -> None:
        if path.exists():
            report.add(path, reason)
            if not dry_run:
                path.unlink()

    for tmp in cache_dir.rglob("*.tmp"):
        drop(tmp, "partial download")
    for body in cache_dir.rglob("*.body"):
        meta = body.with_suffix(".meta.json")
        host = body.relative_to(cache_dir).parts[0]
        if not meta.exists():
            drop(body, "partial download")
            continue
        reason = ""
        if everything:
            reason = "all downloads"
        elif hosts is not None and host in hosts:
            if larger_than is None:
                reason = f"{host}"
            elif body.stat().st_size > larger_than:
                reason = f"{host} (obsolete large file)"
        if not reason:
            try:
                digest = json.loads(meta.read_text())["sha256"]
                if hashlib.sha256(body.read_bytes()).hexdigest() != digest:
                    reason = "corrupt"
            except (ValueError, KeyError, OSError):
                reason = "corrupt"
        if reason:
            drop(body, reason)
            drop(meta, reason)
    for meta in cache_dir.rglob("*.meta.json"):
        if not meta.with_name(meta.name.removesuffix(".meta.json") + ".body").exists():
            drop(meta, "partial download")
    return report
