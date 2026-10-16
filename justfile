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

# Dev mode: the job server (f1p dev serve) plus the web app, opened on /dev/ to run the checks
# and the M6 checklist and follow them live (args: --parallel N for the server).
dev *args:
    ./f1p-dev.sh {{args}}

# After a race weekend: update the data, refit, publish, record the ledger and rebuild the site.
refresh:
    uv run f1p data update
    uv run f1p data pace --from `date +%Y`
    uv run f1p data laps --from `date +%Y`
    uv run f1p data circuits
    uv run f1p model publish
    uv run f1p model ledger
    cd web && pnpm sync-data && pnpm build

# M6 nights (plan §E): the lap posterior of the next race (event id, e.g. 2026-R20), the
# simulator's calibrated knobs, then the forecast publish.
overnight event:
    uv run f1p lap fit {{event}}
    uv run f1p sim calibrate
    uv run f1p model publish

# A resumable hyper-parameter search on tune, then the one-SE choice on select (plan §C).
tune space trials="100":
    uv run f1p tune run {{space}} --trials {{trials}}
    uv run f1p tune select {{space}}

# Train the lap-sequence model for a forecast season (small or full, plan §D.2).
lapseq season size="small":
    uv run f1p lapseq train {{season}} --size {{size}}

# Stack v2 on select (2022-23), or once on the holdout with `just stack --split holdout`.
stack *args:
    uv run f1p eval stack {{args}}

# Everything CI runs.
ci: lint test
