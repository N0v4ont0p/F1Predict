"""Models known to ``f1p eval`` and the forecasting commands."""

from __future__ import annotations

from typing import Any


def registered_models() -> dict[str, Any]:
    from f1p.models.world_v1 import WorldV1

    models: list[Any] = [
        WorldV1(),
        # ablations: each switches one ingredient off (or on)
        WorldV1(name="world_v1_plain", smooth_stages=False),
        WorldV1(name="world_v1_noret", retirements=False),
        WorldV1(name="world_v1_q", quali_obs=True),
    ]
    return {m.name: m for m in models}
