"""Models known to ``f1p eval`` and the forecasting commands."""

from __future__ import annotations

from typing import Any


def registered_models() -> dict[str, Any]:
    from f1p.models.world_v1 import WorldV1

    models: list[Any] = [
        WorldV1(),
        WorldV1(name="world_v1_noret", retirements=False),
        WorldV1(
            name="world_v1_pace",
            features={"C1": ("rating",), "C4": ("rating", "grid", "quali", "pace")},
        ),
    ]
    return {m.name: m for m in models}
