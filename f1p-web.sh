#!/usr/bin/env bash
# f1p-web.sh — open the f1predict web app on localhost with one command.
#
#   ./f1p-web.sh              dev server (hot reload) on the first free port from 3000
#   ./f1p-web.sh --refresh    update data + race pace + republish forecasts first
#   ./f1p-web.sh --prod       build the static site and serve it (what you would deploy)
#   ./f1p-web.sh --no-open    do not open a browser (CI, remote shells)
#   ./f1p-web.sh --port 4000  start from a different port
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEB="$ROOT/web"
MARTS="$ROOT/artifacts/marts"
PORT=3000
MODE=dev
REFRESH=0
OPEN=1
STALE_DAYS=7

say() { printf '\033[1;36m›\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31m✗\033[0m %s\n' "$*" >&2; exit 1; }

usage() { sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case "$1" in
    --refresh) REFRESH=1 ;;
    --prod) MODE=prod ;;
    --dev) MODE=dev ;;
    --no-open) OPEN=0 ;;
    --port) shift; PORT="${1:?--port needs a number}" ;;
    -h | --help) usage; exit 0 ;;
    *) die "unknown option: $1 (try --help)" ;;
  esac
  shift
done

# --- prerequisites -------------------------------------------------------------------------
need() {
  command -v "$1" >/dev/null 2>&1 || die "$1 is not installed. $2"
}
need node "Install Node.js 20+: https://nodejs.org (macOS: brew install node)"
need pnpm "Install pnpm: corepack enable && corepack prepare pnpm@latest --activate"
node_major="$(node -p 'process.versions.node.split(".")[0]')"
[ "$node_major" -ge 20 ] || die "Node.js 20+ is required (found $(node -v))"
if [ "$REFRESH" -eq 1 ]; then
  need uv "Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
fi

# --- dependencies (only when the lockfile changed) -----------------------------------------
lock_hash() {
  if command -v shasum >/dev/null 2>&1; then shasum -a 256 "$WEB/pnpm-lock.yaml" | cut -d' ' -f1
  else sha256sum "$WEB/pnpm-lock.yaml" | cut -d' ' -f1; fi
}
stamp="$WEB/node_modules/.f1p-lock-hash"
if [ ! -d "$WEB/node_modules" ] || [ "$(cat "$stamp" 2>/dev/null)" != "$(lock_hash)" ]; then
  say "installing web dependencies"
  (cd "$WEB" && pnpm install --frozen-lockfile)
  lock_hash >"$stamp"
fi

# --- data -----------------------------------------------------------------------------------
if [ "$REFRESH" -eq 1 ]; then
  say "refreshing data (F1DB, race pace) and republishing forecasts"
  (cd "$ROOT" && uv run f1p data update && uv run f1p data pace --from "$(date +%Y)" \
    && uv run f1p model publish)
fi
if [ -f "$MARTS/index.json" ]; then
  if [ -n "$(find "$MARTS/index.json" -mtime +"$STALE_DAYS" 2>/dev/null)" ]; then
    warn "forecasts are more than $STALE_DAYS days old — run with --refresh to update"
  fi
  (cd "$WEB" && pnpm --silent sync-data "$MARTS" >/dev/null)
elif [ ! -f "$WEB/public/data/index.json" ]; then
  die "no forecasts found — run with --refresh (needs uv) to build them"
else
  say "using the forecasts bundled with the site (run --refresh for fresh ones)"
fi

# --- port -----------------------------------------------------------------------------------
port_busy() {
  if command -v lsof >/dev/null 2>&1; then lsof -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1
  elif command -v ss >/dev/null 2>&1; then ss -ltn "sport = :$1" | grep -q LISTEN
  else (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; fi
}
while port_busy "$PORT"; do PORT=$((PORT + 1)); done
URL="http://localhost:$PORT"

# --- serve ----------------------------------------------------------------------------------
pids=()
cleanup() {
  for p in "${pids[@]:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

if [ "$MODE" = prod ]; then
  say "building the static site"
  (cd "$WEB" && pnpm build >/dev/null)
  need python3 "python3 is used to serve the static build"
  say "serving web/out at $URL"
  python3 -m http.server "$PORT" --bind 127.0.0.1 --directory "$WEB/out" >/dev/null 2>&1 &
else
  say "starting the dev server at $URL"
  (cd "$WEB" && exec pnpm exec next dev --port "$PORT") &
fi
pids+=("$!")

# wait until it answers (up to 60 s), then open the browser
for _ in $(seq 1 120); do
  if curl -fsS -o /dev/null "$URL/" 2>/dev/null; then break; fi
  kill -0 "${pids[0]}" 2>/dev/null || die "the server exited — see the output above"
  sleep 0.5
done
curl -fsS -o /dev/null "$URL/" 2>/dev/null || die "the server did not answer at $URL"
say "ready: $URL  (Ctrl-C to stop)"
if [ "$OPEN" -eq 1 ]; then
  if command -v open >/dev/null 2>&1; then open "$URL"
  elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL" >/dev/null 2>&1 || true
  else warn "open $URL in your browser"; fi
fi
wait "${pids[0]}"
