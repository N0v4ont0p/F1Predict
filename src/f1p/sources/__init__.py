"""Source connectors: one module per external data source (plan §6.1).

Connectors only *fetch* and cache raw payloads; turning them into curated tables is the job of
``f1p.lake``. Every fetch is rate-limited per host, retried on transient failures and recorded
in an immutable raw cache, so re-running a pipeline never re-downloads history.
"""
