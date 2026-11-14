#!/usr/bin/env bash
# Create the runtime venv (local or EC2). Usage: scripts/install.sh [python3.12]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${1:-python3.12}
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt
.venv/bin/pip install -q --no-deps -r requirements-nodeps.txt   # mediapipe without opencv-contrib
.venv/bin/python -c "import cv2, onnxruntime, mediapipe; print('runtime ok: opencv', cv2.__version__)"
