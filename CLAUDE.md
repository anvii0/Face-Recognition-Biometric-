# HID Hackathon — "Can You Spot a Spoof?" face liveness + identity prototype

## Goal
Webcam app returning ALLOW / REQUEST_VERIFICATION / DENY in <=2 s (end of capture -> decision on
screen), verifying liveness (photo, phone screen, video replay, paper mask) AND identity against an
enrolled face. Low-confidence -> step-up challenge (blink / head turn; OTP mock optional).
Report measured per-spoof detection rate, FAR, FRR, latency p50/p95 — never invented numbers.

## Architecture (planned)
- frontend/: static HTML/JS (no build), getUserMedia, enroll + verify screens, 3-5 frames/0.5 s, eval-mode labels.
- backend/: FastAPI + ONNX Runtime CPU. Quality gates -> MiniFASNet liveness (+ optional FFT/moire)
  -> Rekognition CompareFaces (fallback: local ArcFace) -> decision layer (thresholds in config.yaml)
  -> step-up endpoint. JSON-lines log per request with per-stage latency.
- eval/: replay eval_data/<label>/* through the same pipeline; dev/test split; metrics + sweep plot; recorder.
- deploy/: EC2 + systemd, CloudFront for HTTPS, S3 for enrolled faces/logs; deploy.sh / teardown.sh.

## Environment
- Region us-east-1 ONLY. Temporary sandbox creds come from env vars; never print/write/commit them.
  ExpiredToken -> ask the user to re-export.
- Account destroyed ~2026-11-16 02:40 IST: everything must be reproducible from the repo.
- Python: **3.12** venv at `.venv` (python@3.11 unavailable: Homebrew here is an Intel install on an
  arm64 Mac and is broken). Ubuntu 24.04 on EC2 also ships 3.12. AWS CLI v1 is installed in the venv: `.venv/bin/aws`.
- No face images, eval data, model binaries, or logs in git (see .gitignore).

## How to run
- Preflight: `.venv/bin/python scripts/preflight.py`

## Current phase
Phase 0 DONE (2026-11-14). Next: Phase 1 local baseline.

## Open issues / findings
- Preflight: STS, Rekognition (ListCollections/DetectFaces/CompareFaces), S3 create/put, EC2 run-instances
  dry-run (c6i.large, t3.large, Ubuntu 24.04 AMI via SSM), SG, IAM role + instance profile, CloudFront
  write, Lambda/ECR/APIGW/EB all WORK. Role: assumed-role/WSParticipantRole. Plan A architecture stands.
- Other regions are NOT blocked by IAM -> region discipline is on us; always pass us-east-1.
- Account has a `WSLambdaCurtailerRole` + 1 Lambda (sandbox-managed; may stop/clean resources) — watch for
  instances being stopped unexpectedly.
- python.org Python lacks system CA certs: use certifi for urllib/HTTPS (requests/boto already fine).
- Outbound internet OK (GitHub, HF, PyPI). MiniFASNet .pth weights downloadable -> convert to ONNX locally.
- Rekognition CompareFaces preflight uses a synthetic face-less image (InvalidParameter = reachable);
  real face match is verified in Phase 1.
