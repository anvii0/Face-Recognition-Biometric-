#!/usr/bin/env bash
# Start the app on http://localhost:8000 with a per-machine BENCH_TOKEN (stored in data/, gitignored).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
[ -s data/.bench_token ] || python3 -c "import secrets;print(secrets.token_urlsafe(16))" > data/.bench_token
export BENCH_TOKEN=$(cat data/.bench_token)
echo "Kiosk: http://localhost:8000/?token=$BENCH_TOKEN"
echo "Bench: http://localhost:8000/bench?token=$BENCH_TOKEN"
exec .venv/bin/uvicorn backend.main:asgi --host 127.0.0.1 --port "${PORT:-8000}"
