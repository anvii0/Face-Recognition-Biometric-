# HID Hackathon — "Can You Spot a Spoof?" face liveness + identity prototype

## Goal
Webcam app returning ALLOW / REQUEST_VERIFICATION / DENY in <=2 s (end of capture -> decision on
screen), verifying liveness (photo, phone screen, video replay, paper mask) AND identity against an
enrolled face. Low-confidence -> step-up challenge. Report MEASURED per-spoof detection rate, FAR, FRR,
APCER/BPCER with n + 95% Wilson CIs, latency p50/p95 — never invented numbers.

## Architecture (as built)
- `backend/vision.py` YuNet detection (+5 landmarks), quality metrics, MiniFASNet x2 ONNX ensemble
  (upstream crop logic; YuNet box squared to mimic RetinaFace), FFT/moire cue, landmark motion cue.
- `backend/decision.py` pure decide(raw, cfg, variant); mirrored in `frontend/decision.js`
  (parity test). Variants: A MiniFASNet, B +moire, C +motion. Every decision has a human reason.
- `backend/challenge.py` blink/smile (MediaPipe Face Landmarker blendshapes) + head turn (YuNet yaw), server-side.
- `backend/pipeline.py` one pipeline: analyze() per frame; verify() (identity runs in parallel with
  liveness); LiveSession (WS); ChallengeSession (shared by gated enrollment + verify step-up).
- `backend/matcher.py` Rekognition CompareFaces (1:1), SearchFacesByImage (1:N, duplicate check),
  IndexFaces/DeleteFaces; keep-warm loop (first real CompareFaces ~2 s cold).
- `backend/people.py` people metadata SQLite + S3 refs (SSE AES256, 3-day lifecycle) + Rekognition collection.
- `backend/store.py` attempt log SQLite (data/bench.db) + eval_data/<label>/<id>/ frame saving.
- `backend/main.py` FastAPI; BENCH_TOKEN shared-token auth (pure ASGI, covers WS; ?token= -> HttpOnly cookie).
  uvicorn target is `backend.main:asgi` (NOT `app`).
- Frontend (no build): `/` kiosk (index.html, kiosk.js, kiosk.css — hand CSS, no CDN JS), `/bench`
  (bench.html/js/css), shared `camera.js`, `decision.js`.
- AWS data: bucket `hid-liveness-<acct>`, collection `hid-liveness-faces` via `scripts/aws_resources.py`.

## Environment
- Region us-east-1 ONLY (IAM does NOT block other regions — discipline is on us). Temporary creds from env;
  never print/write/commit. ExpiredToken -> ask the user to re-export, then restart Claude from that shell.
- Account destroyed ~2026-11-16 02:40 IST: everything reproducible from repo.
- Python 3.12 venv `.venv` (Homebrew here is Intel/broken; ffmpeg in /usr/local is Intel too). AWS CLI v1 in venv.
- mediapipe installed `--no-deps` (requirements-nodeps.txt) to avoid opencv-contrib clash; needs matplotlib.
- OpenCV pinned <5 (5.x removed the Caffe loader; we only need 4.x APIs). uvicorn needs `websockets` for WS.
- No face images, eval data, models, data/, logs in git (see .gitignore).

## How to run
- Setup: `scripts/install.sh python3.12` ; models: `.venv/bin/pip install -r requirements-dev.txt && .venv/bin/python scripts/fetch_models.py`
  (+ `curl -L -o models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`)
- AWS data resources: `.venv/bin/python scripts/aws_resources.py` (`--destroy` to remove)
- Run: `scripts/run_local.sh` (prints kiosk/bench URLs with token)
- Tests: `.venv/bin/python -m pytest -q tests` (e2e test uses real AWS; skipped without creds)
- Rekognition real-pair check: `.venv/bin/python scripts/compare_faces.py a.jpg b.jpg [other.jpg]`
- Preflight: `.venv/bin/python scripts/preflight.py`

## Current phase
Phase 1 (+1b bench, +1c kiosk) built and verified with a public-domain sample + Chrome fake camera.
WAITING ON USER: real webcam tests (enroll self, printed-photo & phone-screen enrollment attempts, verify
self ALLOW, verify spoof DENY), real face pair for CompareFaces, confirm turn-left/right sign on real camera.
Next: Phase 2 hardening (step-up already exists), Phase 3 eval harness + report.py, Phase 4 deploy.

## Measured so far (sample image, laptop, localhost) — not eval numbers
- Server verify total 360-650 ms warm (Rekognition 320-600 ms dominates; runs parallel to liveness ~15 ms/frame).
- Browser capture-end -> decision shown: 0.46-0.69 s (bench), 0.5 s (kiosk). Live loop 80 ms p50 / 100-105 p95, ~12 fps.
- Cold first CompareFaces ~2.07 s -> mitigated by keep-warm on the enrolled reference.

## Open issues / risks
- Thresholds/moire range are UNTUNED placeholders (Phase 3 tunes on dev split only).
- A clean digital photo scores high on MiniFASNet (0.82-0.96): passive model targets recapture artifacts.
  Camera-injection of a digital image (virtual camera) would pass passive liveness; the active challenge
  is what stops a static image (verified: enrollment + step-up reject it). Real printed/screen attacks untested yet.
- Head-turn sign assumed (user's left => +yaw on un-mirrored frames); `challenge.left_yaw_sign` flips it.
- Verify upload is 5 x ~45 KB JPEG; on slow Wi-Fi this may threaten the 2 s budget (measure in Phase 4).
- Duplicate-check admin detail is UI separation only (same token), not an access-control boundary.
- Phase 5 TODO: eval/report.py (Wilson CIs, md+png), docs/architecture.md (Mermaid, latency budget, scaling,
  privacy/retention), one-command demo reset script, pre-recorded fallback demo video, /demo low-bandwidth page.
