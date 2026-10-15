# f1predict task runner — `just --list` for everything.

default:
    @just --list

# Install the environment (Python 3.13 via uv, all dev tools).
setup:
    uv sync --all-groups --all-extras

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

# Benchmark every compute unit on this machine (plan §11a.6); results go to artifacts/bench/.
bench *args:
    uv run f1p silicon bench {{args}}

# Open the web app on localhost (args: --refresh, --prod, --no-open, --port N).
web *args:
    ./f1p-web.sh {{args}}

# Everything CI runs.
ci: lint test
