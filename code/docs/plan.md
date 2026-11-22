# Execution plan

HID Hackathon "Can You Spot a Spoof?" — face liveness + identity prototype on the AWS sandbox
(us-east-1 only; account destroyed ~2026-11-16 02:40 IST, so everything must be reproducible from this repo).

Goal: webcam app that returns ALLOW / REQUEST_VERIFICATION / DENY within 2 s of the end of capture,
checks liveness (printed photo, phone screen, video replay, paper mask) AND identity, steps up on low
confidence, and reports **measured** per-attack detection rate, FAR, FRR, APCER/BPCER (n + 95% Wilson CIs)
and decision time p50/p95. No invented numbers; never tune on the held-out test set.

## Phases and status

| Phase | Scope | Status |
|---|---|---|
| 0 Preflight | STS, Rekognition, S3, EC2 dry-run, IAM, CloudFront, Bedrock checks (`scripts/preflight.py`) | Done |
| 1 Local baseline | MiniFASNet ONNX, YuNet, quality gates, Rekognition 1:1/1:N, decision layer, kiosk, test bench, liveness-gated enrollment, step-up challenges | Done (sample image + fake camera; real webcam tests pending) |
| A hid-gate merge | Port teammate's cloud signals (Bedrock VLM, DetectLabels, DetectFaces extras, finger gesture); variants D (cloud-only) and E (fused, fail closed); bench compares A-E | Done |
| B Kiosk / Admin / Lab split | Public kiosk (verify only), staff `/admin` (enrollment, people, access log, reset) and `/lab` (bench) behind an `ADMIN_PIN`; separate `access_log` and `lab_runs` stores; automatic identity warm-up | In progress |
| 2 Hardening | Decision boundary tests (done), challenge calibration on real camera (turn sign) | Partly done |
| 3 Evaluation | Labelled capture in the lab (dev vs held-out sessions), `eval/` replay harness, threshold tuning on dev only, `eval/report.py` (markdown + PNG, n + Wilson CIs) | Next |
| 4 Deployment | EC2 + systemd, CloudFront HTTPS, S3, least-privilege role, `deploy.sh` (generates ADMIN_PIN) / `teardown.sh` (our resources only), smoke test from a second device | Planned |
| 5 Polish | README, architecture/privacy docs, demo script, one-command demo reset, fallback demo video, low-bandwidth `/demo` page, slide-ready charts | Planned |

## Part B plan (current)

1. **Stores:** split the attempt log into `access_log` (real kiosk attempts) and `lab_runs` (labelled lab runs,
   with an optional session name for dev vs held-out). One-time migration: labelled attempts go to `lab_runs`,
   the rest to `access_log`; the old DB is kept as a backup.
2. **Auth:** public routes are only `/`, `GET /api/config` (no thresholds), `POST /api/verify`, `/api/stepup/*`,
   `/ws/live`, `/healthz`, `/static/*`. Public verify ignores label, save_frames, variant and ground-truth
   fields. Every staff route needs `X-Admin-Pin`, compared with `secrets.compare_digest` against `ADMIN_PIN`;
   503 if `ADMIN_PIN` is unset (fail closed); lockout after 5 wrong PINs. `BENCH_TOKEN` is removed.
3. **Pages:** `/` kiosk (verify only, tiny Staff link), `/admin` (enroll, people, access log, clear log,
   demo reset), `/lab` (current bench plus session name, session filter, JSON export; metrics from lab runs only).
   Shared `common.css` / `common.js`.
4. **Warm-up:** after each successful enrollment and at server start, run a throwaway identity check so the
   first real verify is not cold (measured ~2.3 s cold vs ~0.35-0.6 s warm).
5. **Tests and docs:** auth (401/503/lockout), store separation, public-verify field stripping, migration;
   README, CLAUDE.md, docs/architecture.md.

## Ground rules

- Never commit credentials, face images, eval data, model binaries, databases or logs.
- Never touch the teammate's Rekognition collection `hid-gate`; reset/teardown only remove our resources.
- Report weaknesses honestly (3D masks, unfamiliar cameras, lighting, VLM evidence can be wrong).
