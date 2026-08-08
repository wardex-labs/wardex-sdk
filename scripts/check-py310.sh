#!/usr/bin/env bash
# Run the whole Python suite on CPython 3.10 — the floor `requires-python`
# declares in sdks/python/pyproject.toml.
#
# Why this exists: the development venv is 3.14, so a 3.11+-only runtime API
# (`asyncio.create_task(..., context=...)` is the one that actually happened)
# passes every local run and only turns red in CI's 3.10 job — after the push.
# This is that job, before the push.
#
# Usage: scripts/check-py310.sh [pytest args...]
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# The 3.10 interpreter gets its own environment. Syncing 3.10 into `.venv`
# would swap the development interpreter out from under the next `uv run`,
# and the next `uv sync` would swap it back — a check nobody runs twice.
#
# Assigned, not defaulted: `UV_PROJECT_ENVIRONMENT` is a general uv knob that
# people export from a shell profile or direnv, so honouring an inherited value
# would aim this sync at whatever venv they named — the one outcome the
# paragraph above says cannot happen.
export UV_PROJECT_ENVIRONMENT=.venv-py310

# `--reinstall-package` rebuilds the native module from the current Rust
# source; without it uv treats the installed wheel as current and the run
# measures whatever was built last.
uv sync --python 3.10 --reinstall-package wardex-sdk --no-default-groups --group test

# `--no-sync` runs exactly what the line above installed — plain `uv run`
# re-syncs with the default groups, which would reinstate the 3.14 resolution
# of the dev tooling in this environment.
uv run --python 3.10 --no-sync pytest sdks/python/tests "$@"
