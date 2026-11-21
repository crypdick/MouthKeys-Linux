#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
uv sync --project linux --python "${MOUTHKEYS_PYTHON:-3.13}" --extra gui --locked
uv run --project linux --locked --no-sync prek run --all-files
uv build linux
