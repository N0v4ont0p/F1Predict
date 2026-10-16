#!/usr/bin/env bash
# f1p-dev.sh — start dev mode on a fresh Mac: the job server plus the web app on /dev/.
#
#   ./f1p-dev.sh               installs uv if missing, then opens http://localhost:3000/dev/
#   ./f1p-dev.sh --parallel 2  let the job server run two jobs at once
#
# Only uv and Node.js are needed up front; everything else is one click on the page
# ("Install everything").
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

say() { printf '\033[1;36m›\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m✗\033[0m %s\n' "$*" >&2; exit 1; }

if ! command -v uv >/dev/null 2>&1; then
  say "installing uv (Python manager) into ~/.local/bin"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
command -v node >/dev/null 2>&1 \
  || die "Node.js is not installed: download the LTS installer from https://nodejs.org, then run this again"
if ! command -v pnpm >/dev/null 2>&1; then
  say "enabling pnpm (ships with Node.js)"
  corepack enable 2>/dev/null || npm install -g pnpm
fi

# `uv run` installs Python 3.13 and the base packages on first use (a minute or two).
say "starting the job server"
uv run f1p dev serve "$@" &
api=$!
trap 'kill $api 2>/dev/null || true' EXIT INT TERM
./f1p-web.sh --path /dev/
