"""The data lake: raw → staged → curated tables on DuckLake (plan §6.2).

* :mod:`f1p.lake.store`     — DuckLake catalogue (snapshots, time travel) over Parquet.
* :mod:`f1p.lake.schema`    — the curated entity model as explicit, checked table specs.
* :mod:`f1p.lake.timing`    — ``observed_at`` rules: when each fact became public.
* :mod:`f1p.lake.f1db`      — F1DB SQLite → curated tables.
* :mod:`f1p.lake.quality`   — data-quality gates (fail loudly, never warn silently).
* :mod:`f1p.lake.pipeline`  — orchestrates fetch → curate → gate → commit.
"""
