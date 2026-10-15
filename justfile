# f1predict task runner — `just --list` for everything.

default:
    @just --list

# Install the environment (Python 3.13 via uv, all dev tools).
setup:
    uv sync --all-groups

# Format, lint, type-check and enforce module boundaries.
lint:
    uv run ruff format --check .
    uv run ruff check .
    uv run basedpyright
    uv run lint-imports

# Apply formatting and auto-fixable lint.
fmt:
    uv run ruff format .
    uv run ruff check --fix .

# Run the test suite.
test *args:
    uv run pytest {{args}}

# Everything CI runs.
ci: lint test
