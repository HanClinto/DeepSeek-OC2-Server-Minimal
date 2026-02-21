#!/usr/bin/env bash
# start.sh – Start the DeepSeek-OCR2 server.
set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

echo "Starting DeepSeek-OCR2 server on http://${HOST}:${PORT}"
uv run uvicorn server:app --host "$HOST" --port "$PORT"
