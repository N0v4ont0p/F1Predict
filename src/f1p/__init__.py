"""f1predict — probabilistic Formula 1 forecasting.

See ``docs/plan/REBUILD_PLAN.md`` for the architecture this package implements.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("f1predict")
except PackageNotFoundError:  # pragma: no cover - running from a source tree without install
    __version__ = "0.0.0+local"

__all__ = ["__version__"]
