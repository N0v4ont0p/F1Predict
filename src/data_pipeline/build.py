"""Master dataset builder (v2) — real-data first with per-season Parquet caching.

Strategy:
* Fetch each season once from jolpica and cache it to ``data/cache/season_YYYY.parquet``.
  Subsequent builds reuse the cache (the API is slow & rate-limited).
* The completed seasons are concatenated into the master. A synthetic generator remains as
  an explicit offline fallback, but **real data is the default** so predictions are sane.
* A manifest records coverage, the content hash, and profiling for reproducibility.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from ..config import Config, load_config
from ..utils.logging import get_logger
from ..utils.profiling import profile
from . import synthetic
from .jolpica import JolpicaError, fetch_season
from .schema import KEY_COLUMNS, RAW_COLUMNS

log = get_logger()


def _hash_frame(df: pd.DataFrame) -> str:
    h = hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes())
    return h.hexdigest()[:16]


def _cache_path(cfg: Config, season: int) -> Path:
    cache = cfg.path("paths.cache_dir")
    cache.mkdir(parents=True, exist_ok=True)
    return cache / f"season_{season}.parquet"


def fetch_season_cached(cfg: Config, season: int, force: bool = False) -> pd.DataFrame | None:
    """Return a season's data, using the Parquet cache when available."""
    cache = _cache_path(cfg, season)
    if cache.exists() and not force:
        return pd.read_parquet(cache)
    base = cfg.get("data.jolpica_base_url")
    timeout = int(cfg.get("data.request_timeout"))
    sleep = float(cfg.get("data.rate_limit_sleep"))
    try:
        df = fetch_season(season, base, timeout, sleep)
    except JolpicaError as exc:
        log.warning(f"[yellow]{exc}[/yellow]")
        return None
    # Only cache seasons that actually contain results (skip not-yet-run future seasons).
    if len(df):
        df.to_parquet(cache, index=False)
    return df


def build_master(
    cfg: Config | None = None,
    seasons: list[int] | None = None,
    source: str = "auto",
    force_refresh: bool = False,
    refresh_seasons: list[int] | None = None,
) -> pd.DataFrame:
    """Build & persist the master dataset.

    ``source``: ``"auto"`` (real jolpica per-season with cache; synthetic only if a season
    completely fails), ``"jolpica"`` (strict real), ``"synthetic"`` (offline demo).

    ``force_refresh`` re-fetches **every** season from the API (ignores the Parquet cache).
    ``refresh_seasons`` is the surgical alternative — re-fetch only the listed seasons (e.g.
    the in-progress year after a race) while reusing the cache for immutable past seasons.
    Used by ``f1predict update`` to refresh latest results without hammering the API.
    """
    cfg = cfg or load_config()
    if seasons is None:
        start = int(cfg.get("data.start_season"))
        end = int(cfg.get("data.end_season", 2026))
        seasons = list(range(start, end + 1))
    refresh_set = set(refresh_seasons or [])

    frames: list[pd.DataFrame] = []
    synth_seasons: list[int] = []

    with profile("build_master") as prof:
        if source == "synthetic":
            frames.append(synthetic.generate(seasons, seed=cfg.get("project.seed", 42)))
        else:
            for season in seasons:
                force = force_refresh or (season in refresh_set)
                df = fetch_season_cached(cfg, season, force=force)
                if df is not None and len(df):
                    frames.append(df)
                elif source == "auto" and cfg.get("data.allow_synthetic_fallback", True):
                    synth_seasons.append(season)
            if synth_seasons and source == "auto" and not frames:
                # Only fall back to synthetic if we got *nothing* real (offline).
                frames.append(synthetic.generate(synth_seasons, seed=cfg.get("project.seed", 42)))

    if not frames:
        raise JolpicaError("no data fetched and no synthetic fallback produced")

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=KEY_COLUMNS).sort_values(
        ["season", "round", "position"]
    ).reset_index(drop=True)
    df = df[list(RAW_COLUMNS)]

    out_path = cfg.path("paths.master_dataset")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    manifest = {
        "rows": int(len(df)),
        "seasons": sorted(int(s) for s in df["season"].unique()),
        "synthetic_seasons": sorted(synth_seasons),
        "content_hash": _hash_frame(df),
        "profile": prof.as_dict(),
        "source": source,
        "drivers": int(df["driver_id"].nunique()),
        "races": int(df.groupby(["season", "round"]).ngroups),
    }
    out_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info(
        f"master: [green]{len(df)}[/green] rows · {manifest['races']} races · "
        f"{len(manifest['seasons'])} seasons -> {out_path}"
    )
    return df


_MASTER_CACHE: dict[str, tuple[float, pd.DataFrame]] = {}


def load_master(cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    path = cfg.path("paths.master_dataset")
    if not path.exists():
        log.warning("master dataset missing -> building from real data (first run, may take a minute)")
        return build_master(cfg, source="auto")
    key = str(path)
    mtime = path.stat().st_mtime
    cached = _MASTER_CACHE.get(key)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    df = pd.read_parquet(path)
    _MASTER_CACHE[key] = (mtime, df)
    return df


def clear_caches() -> None:
    """Drop the in-process master-dataset cache (used by the interactive shell `/reload`)."""
    _MASTER_CACHE.clear()


def data_status(cfg: Config | None = None) -> dict:
    cfg = cfg or load_config()
    path = cfg.path("paths.master_dataset")
    manifest_path = Path(str(path).replace(".parquet", ".manifest.json"))
    if not path.exists():
        return {"exists": False}
    info = {"exists": True, "path": str(path)}
    if manifest_path.exists():
        info.update(json.loads(manifest_path.read_text()))
    return info
