"""FastAPI app: kiosk (/), test bench (/bench), REST API (/api), live WebSocket (/ws/live).

Run: BENCH_TOKEN=... .venv/bin/uvicorn backend.main:asgi --port 8000
If BENCH_TOKEN is unset, auth is disabled (localhost dev only; deploy.sh always sets it).
"""
from __future__ import annotations

import asyncio
import base64
import csv
import hmac
import io
import json
import logging
import os
import random
import re
import threading
import time
import uuid
from http.cookies import SimpleCookie
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode

import boto3
import yaml
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .decision import ALLOW, DENY, VARIANTS
from .people import ID_RE, People
from .pipeline import ChallengeSession, Identify, LiveSession, Pipeline, face_thumb, stepup_decision
from .store import DATA, EVAL, LABELS, ROOT, Store

CONFIG_PATH = Path(os.environ.get("HID_CONFIG", ROOT / "config.yaml"))
FRONTEND = ROOT / "frontend"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
log = logging.getLogger("hid")

CFG: dict[str, Any] = yaml.safe_load(CONFIG_PATH.read_text())
DATA_DIR = Path(os.environ.get("HID_DATA_DIR", DATA))  # overridable for isolated tests
if os.environ.get("HID_COLLECTION"):
    CFG["match"]["collection"] = os.environ["HID_COLLECTION"]
PIPE = Pipeline(CFG)
STORE = Store(DATA_DIR)


def _bucket() -> str | None:
    b = os.environ.get("HID_BUCKET") or CFG["storage"]["bucket"]
    if b != "auto":
        return b or None
    try:
        acct = boto3.client("sts", region_name="us-east-1").get_caller_identity()["Account"]
        return f"hid-liveness-{acct}"
    except Exception as e:  # noqa: BLE001 - run without S3 rather than crash
        log.warning("no S3 bucket (%s); reference images will not be persisted", e)
        return None


PEOPLE = People(DATA_DIR / "people.db", PIPE.matcher, _bucket())
PIPE.reference_for_warmup = lambda: (PEOPLE.reference(PEOPLE.ids()[-1]) if PEOPLE.ids() else None)
if PIPE.warm_kick is not None:
    PIPE.warm_kick.set()

SESSIONS: dict[str, ChallengeSession] = {}
_sess_lock = threading.Lock()
FRIENDLY_FAIL = "We couldn't confirm a live person. Please try again."


# ---------------- shared-token auth (pure ASGI so it also covers WebSockets) ----------------
class TokenAuth:
    COOKIE = "bench_token"

    def __init__(self, app: Any, token: str | None) -> None:
        self.app, self.token = app, token

    def _ok(self, candidate: str | None) -> bool:
        return bool(candidate) and hmac.compare_digest(candidate.encode(), self.token.encode())  # type: ignore[union-attr]

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] not in ("http", "websocket") or not self.token or scope["path"] == "/healthz":
            return await self.app(scope, receive, send)
        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
        query = parse_qs(scope.get("query_string", b"").decode())
        cookie = SimpleCookie(headers.get("cookie", ""))
        from_cookie = cookie[self.COOKIE].value if self.COOKIE in cookie else None
        from_query = (query.get("token") or [None])[0]
        if self._ok(from_query) and scope["type"] == "http":
            # Exchange ?token= for an HttpOnly cookie, then redirect to strip the token from the URL.
            rest = urlencode({k: v[0] for k, v in query.items() if k != "token"})
            secure = "https" in (headers.get("x-forwarded-proto", "") + headers.get("cloudfront-forwarded-proto", ""))
            c = f"{self.COOKIE}={self.token}; HttpOnly; SameSite=Strict; Path=/; Max-Age=172800" + ("; Secure" if secure else "")
            await send({"type": "http.response.start", "status": 303, "headers": [
                (b"location", (scope["path"] + ("?" + rest if rest else "")).encode()), (b"set-cookie", c.encode())]})
            return await send({"type": "http.response.body", "body": b""})
        if self._ok(from_cookie) or self._ok(headers.get("x-bench-token")) or self._ok(from_query):
            return await self.app(scope, receive, send)
        if scope["type"] == "websocket":
            return await send({"type": "websocket.close", "code": 4401})
        await send({"type": "http.response.start", "status": 401, "headers": [(b"content-type", b"text/plain")]})
        await send({"type": "http.response.body", "body": b"401: open this page once with ?token=<BENCH_TOKEN>"})


app = FastAPI(title="HID liveness")
app.mount("/static", StaticFiles(directory=FRONTEND), name="static")
_token = os.environ.get("BENCH_TOKEN") or None
if not _token:
    log.warning("BENCH_TOKEN not set: auth DISABLED (local dev only)")
asgi = TokenAuth(app, _token)  # uvicorn target: backend.main:asgi


def _jsonl(event: dict[str, Any]) -> None:
    with open(LOG_DIR / "requests.jsonl", "a") as f:
        f.write(json.dumps(event, default=str) + "\n")


def _log_attempt(**kw: Any) -> str:
    """Single sink for every decision / enrollment attempt: SQLite + JSON-lines (with per-stage ms)."""
    aid = STORE.add_attempt(**kw)
    _jsonl({"ts": time.time(), "id": aid, **{k: v for k, v in kw.items() if k not in ("thumb", "frames", "per_frame")}})
    return aid


def _label(v: Any) -> str:
    return v if v in LABELS else "unlabeled"


# ---------------- identity resolution ----------------
def _identify(mode: str, person_id: str | None) -> tuple[Identify | None, str]:
    """Returns (identify fn, reason when none). claim = 1:1 CompareFaces; open = 1:N SearchFacesByImage.
    identify(probe) -> (similarity, error, person_id)."""
    m = PIPE.matcher
    if m is None or mode == "none":
        return None, "identity check disabled"
    if mode == "open":
        if not PEOPLE.ids():
            return None, "nobody enrolled"

        def search(probe: bytes) -> tuple[float | None, str | None, str | None]:
            hits, err, _ = m.search(probe, max_faces=1)
            if err:
                return None, err, None
            return (hits[0][1], None, hits[0][0]) if hits else (0.0, None, None)
        return search, ""
    pid = person_id or (PEOPLE.ids()[-1] if PEOPLE.ids() else None)
    if pid is None:
        return None, "nobody enrolled"
    if not PEOPLE.get(pid):
        raise HTTPException(404, f"ID '{pid}' is not enrolled")
    ref = PEOPLE.reference(pid)
    if ref is None:
        return None, "reference image unavailable"

    def compare(probe: bytes) -> tuple[float | None, str | None, str | None]:
        sim, err, _ = m.compare(ref, probe)
        return sim, err, pid
    return compare, ""


# ---------------- pages ----------------
@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def kiosk() -> FileResponse:
    return FileResponse(FRONTEND / "index.html")


@app.get("/bench")
def bench() -> FileResponse:
    return FileResponse(FRONTEND / "bench.html")


# ---------------- config ----------------
@app.get("/api/config")
def get_config() -> dict[str, Any]:
    return {**CFG, "labels": list(LABELS), "variants": list(VARIANTS), "users": PEOPLE.ids()}


@app.post("/api/config")
def set_config(body: dict[str, Any]) -> dict[str, Any]:
    th, cl = body.get("thresholds", {}), body.get("cloud", {})
    for k in ("t_lo", "t_hi", "t_match", "t_match_lo"):
        if k in th:
            CFG["thresholds"][k] = float(th[k])
    for k in ("vlm_live_min", "vlm_spoof_max"):
        if k in cl:
            CFG["cloud"][k] = float(cl[k])
    if body.get("save"):
        text = CONFIG_PATH.read_text()
        items = list(CFG["thresholds"].items()) + [(k, CFG["cloud"][k]) for k in ("vlm_live_min", "vlm_spoof_max")]
        for k, v in items:  # targeted rewrite keeps the file's comments
            text = re.sub(rf"(?m)^(\s+{k}:\s*)[0-9.]+", rf"\g<1>{v:g}", text, count=1)
        CONFIG_PATH.write_text(text)
    return {"thresholds": CFG["thresholds"], "cloud": {k: CFG["cloud"][k] for k in ("vlm_live_min", "vlm_spoof_max")},
            "saved": bool(body.get("save"))}


# ---------------- people ----------------
@app.get("/api/users")
def users() -> list[str]:
    return PEOPLE.ids()


@app.get("/api/people")
def people() -> list[dict[str, Any]]:
    return PEOPLE.list()


@app.get("/api/people/{pid}/photo.jpg")
def person_photo(pid: str) -> Response:
    b = PEOPLE.photo(pid)
    if not b:
        raise HTTPException(404)
    return Response(b, media_type="image/jpeg", headers={"Cache-Control": "private, no-store"})


@app.delete("/api/people/{pid}")
def delete_person(pid: str) -> dict[str, bool]:
    t0 = time.perf_counter()
    if not PEOPLE.delete(pid):
        raise HTTPException(404)
    _jsonl({"ts": time.time(), "event": "person_deleted", "person_id": pid,
            "ms": round((time.perf_counter() - t0) * 1e3, 1)})
    return {"deleted": True}


def _store_person(pid: str, name: str, role: str, refs: list[bytes], infos: list[Any], gated: bool,
                  consent_ts: float | None) -> dict[str, Any]:
    photo = face_thumb(infos[0].img, infos[0].face, 192)
    tm = PEOPLE.add(pid=pid, name=name, role=role, jpegs=refs, ref_index=0, gated=gated,
                    consent_ts=consent_ts, photo=photo)
    if PIPE.warm_kick is not None:
        PIPE.warm_kick.set()  # warm CompareFaces with the new reference before the first verify
    return {"timings": tm, "photo_b64": base64.b64encode(photo).decode()}


@app.post("/api/enroll")
def quick_enroll(name: str = Form(...), files: list[UploadFile] = File(...)) -> dict[str, Any]:
    """Bench-only quick enrollment (quality-gated, NOT liveness-gated). The kiosk uses /api/enroll/start."""
    if not ID_RE.match(name):
        raise HTTPException(400, "name must be 1-32 chars of A-Z a-z 0-9 _ -")
    jpegs = [f.file.read() for f in files]
    if not 1 <= len(jpegs) <= 5:
        raise HTTPException(400, "send 1-5 captures")
    t0 = time.perf_counter()
    ref, problems, infos = PIPE.check_enroll(jpegs)
    if problems:
        raise HTTPException(422, "Enrollment rejected: " + " | ".join(problems))
    dup = PEOPLE.duplicate_of(jpegs[ref], name, CFG["enrollment"]["duplicate_similarity"])
    if dup:
        raise HTTPException(409, f"Face already enrolled as '{dup[0]}' (similarity {dup[1]:.1f})")
    order = [ref] + [i for i in range(len(jpegs)) if i != ref]
    out = _store_person(name, name, "", [jpegs[i] for i in order], [infos[i] for i in order], False, None)
    return {"user": name, "captures": len(jpegs), "reference_index": ref,
            "ms": round((time.perf_counter() - t0) * 1e3, 1), **out}


# ---------------- liveness-gated enrollment (kiosk) ----------------
def _new_session(sess: ChallengeSession) -> str:
    sid = uuid.uuid4().hex
    with _sess_lock:
        for k in [k for k, s in SESSIONS.items() if time.time() - s.created > 300]:
            SESSIONS.pop(k, None)
        SESSIONS[sid] = sess
    return sid


def _session(sid: str, purpose: str) -> ChallengeSession:
    s = SESSIONS.get(sid)
    if s is None or s.purpose != purpose:
        raise HTTPException(404, "session not found or expired")
    return s


async def _frame_body(request: Request) -> bytes:
    data = await request.body()
    if not data or len(data) > 2_000_000:
        raise HTTPException(400, "send one JPEG frame as the request body")
    return data


@app.post("/api/enroll/start")
def enroll_start(body: dict[str, Any]) -> dict[str, Any]:
    pid, name = str(body.get("person_id", "")).strip(), str(body.get("name", "")).strip()
    if not ID_RE.match(pid):
        raise HTTPException(400, "ID must be 1-32 characters: letters, digits, _ or -")
    if not 1 <= len(name) <= 64:
        raise HTTPException(400, "name is required")
    if not body.get("consent"):
        raise HTTPException(400, "consent is required to store face data")
    if PEOPLE.get(pid) and not body.get("replace"):
        raise HTTPException(409, f"ID '{pid}' is already enrolled — use Re-enroll on the People tab")
    e = CFG["enrollment"]
    sess = ChallengeSession(PIPE, kinds=list(e["challenges"]), center_frames=int(e["center_frames"]),
                            min_liveness=float(e["min_liveness"]), purpose="enroll",
                            meta={"person_id": pid, "name": name, "role": str(body.get("role", ""))[:64],
                                  "consent_ts": time.time(), "label": _label(body.get("label")), "logged": False})
    return {"sid": _new_session(sess), "center_frames": e["center_frames"], "challenges": len(e["challenges"])}


def _log_enroll(sess: ChallengeSession, decision: str, reason: str, extra_ms: dict[str, float] | None = None,
                raw_extra: dict[str, Any] | None = None) -> None:
    """Enrollment outcomes are logged like decisions. Images are NEVER stored for enrollment attempts here."""
    if sess.meta.get("logged"):
        return
    sess.meta["logged"] = True
    st = sess.state()
    _log_attempt(label=sess.meta["label"], variant="A", user=sess.meta["person_id"], source="enroll",
                 decision=decision, reason=reason,
                 raw={"liveness_avg": st["liveness_avg"], "challenge": st.get("challenge"), "minifas": None,
                      "quality": None, **(raw_extra or {})},
                 timings={**sess.summary_ms(), **(extra_ms or {})}, per_frame=[], thumb=None, frames=None)


@app.post("/api/enroll/frame")
async def enroll_frame(sid: str, request: Request) -> dict[str, Any]:
    sess = _session(sid, "enroll")
    st = await asyncio.to_thread(sess.frame, await _frame_body(request))
    if st["phase"] == "failed":
        _log_enroll(sess, DENY, f"Enrollment rejected (spoof-at-enrollment suspected): {sess.reason}")
        st["message"] = "That took too long — please try again." if "timed out" in sess.reason else FRIENDLY_FAIL
    return st


@app.post("/api/enroll/complete")
def enroll_complete(body: dict[str, Any], admin: bool = False) -> dict[str, Any]:
    sid = str(body.get("sid", ""))
    sess = _session(sid, "enroll")
    if sess.phase != "passed":
        raise HTTPException(409, "enrollment not ready: liveness and challenges must pass first")
    t0 = time.perf_counter()
    meta = sess.meta
    best = sess.best_references(int(CFG["enrollment"]["references"]))
    refs = [b for b, _ in best]
    td = time.perf_counter()
    dup = PEOPLE.duplicate_of(refs[0], meta["person_id"], CFG["enrollment"]["duplicate_similarity"])
    dup_ms = round((time.perf_counter() - td) * 1e3, 1)
    if dup:
        sess.good.clear()  # drop images of the blocked enrollment
        _log_enroll(sess, DENY, f"Enrollment blocked: face already enrolled as {dup[0]} (similarity {dup[1]:.1f})",
                    {"search_faces": dup_ms}, {"duplicate_of": dup[0], "similarity": dup[1]})
        SESSIONS.pop(sid, None)
        detail: dict[str, Any] = {"message": "This face is already enrolled under another ID. Please contact an administrator."}
        if admin:  # bench only — UI separation, not an access-control boundary (same token)
            detail["admin_detail"] = f"matches '{dup[0]}' ({PEOPLE.name(dup[0])}) at similarity {dup[1]:.1f}"
        raise HTTPException(409, detail)
    out = _store_person(meta["person_id"], meta["name"], meta["role"], refs, [f for _, f in best], True,
                        meta["consent_ts"])
    tm = {"search_faces": dup_ms, **out["timings"], "complete_total": round((time.perf_counter() - t0) * 1e3, 1)}
    done = " + ".join(sess.runner.done) if sess.runner else ""
    _log_enroll(sess, ALLOW, f"Enrolled: liveness {sess.state()['liveness_avg']:.2f}, challenges {done} passed", tm)
    SESSIONS.pop(sid, None)
    return {"person": PEOPLE.get(meta["person_id"]), "photo_b64": out["photo_b64"],
            "timings": {**sess.summary_ms(), **tm}}


@app.post("/api/enroll/cancel")
def enroll_cancel(body: dict[str, Any]) -> dict[str, bool]:
    SESSIONS.pop(str(body.get("sid", "")), None)
    return {"cancelled": True}


# ---------------- verify + step-up ----------------
@app.post("/api/verify")
def verify(files: list[UploadFile] = File(...), user: str | None = Form(None), person_id: str | None = Form(None),
           mode: str = Form("claim"), variant: str = Form("A"), label: str = Form("unlabeled"),
           save_frames: bool = Form(False), source: str = Form("verify")) -> dict[str, Any]:
    if variant not in VARIANTS or label not in LABELS or mode not in ("claim", "open"):
        raise HTTPException(400, f"variant in {VARIANTS}, label in {LABELS}, mode in claim|open")
    jpegs = [f.file.read() for f in files]
    if not 1 <= len(jpegs) <= 8:
        raise HTTPException(400, "send 1-8 frames")
    if source == "kiosk":  # the public kiosk always uses the configured variant
        variant = CFG["kiosk"]["variant"]
    identify, why = _identify(mode, person_id or user)
    # Bench/lab attempts always collect cloud signals so all variants are comparable on identical attempts;
    # the kiosk only pays for them when its variant needs them.
    cloud = source != "kiosk" or variant in ("D", "E")
    r = PIPE.verify(jpegs, identify=identify, variant=variant, no_identity_reason=why, name_of=PEOPLE.name,
                    cloud=cloud)
    r["raw"]["mode"] = mode
    thumb = r.pop("thumb")
    aid = _log_attempt(label=label, variant=variant, user=r["raw"].get("person_id"), source=source,
                       decision=r["decision"], reason=r["reason"], raw=r["raw"], timings=r["timings"],
                       per_frame=r["per_frame"], thumb=thumb, frames=jpegs if save_frames else None)
    return {"id": aid, **r}


@app.post("/api/stepup/start")
def stepup_start(body: dict[str, Any]) -> dict[str, Any]:
    s = CFG["stepup"]
    kinds = random.sample(list(s["pool"]), int(s["challenges"]))
    sess = ChallengeSession(PIPE, kinds=kinds, center_frames=0, min_liveness=0.0, purpose="stepup", pool=list(s["pool"]),
                            meta={"attempt_id": body.get("attempt_id"), "mode": body.get("mode", "claim"),
                                  "person_id": body.get("person_id"), "label": _label(body.get("label")),
                                  "t0": time.perf_counter()})
    return {"sid": _new_session(sess), "challenges": kinds, "timeout_s": CFG["challenge"]["timeout_s"]}


@app.post("/api/stepup/frame")
async def stepup_frame(sid: str, request: Request) -> dict[str, Any]:
    sess = _session(sid, "stepup")
    st = await asyncio.to_thread(sess.frame, await _frame_body(request))
    if st["phase"] in ("passed", "failed"):
        m = sess.meta
        identify = _identify(m["mode"], m["person_id"])[0] if st["phase"] == "passed" else None
        final = await asyncio.to_thread(stepup_decision, sess, identify, CFG)
        if final.get("user"):
            final["person_id"], final["user"] = final["user"], PEOPLE.name(final["user"])
        final["id"] = _log_attempt(
            label=m["label"], variant="A", user=final.get("person_id"), source="stepup", decision=final["decision"],
            reason=final["reason"],
            raw={"stepup_of": m["attempt_id"], "liveness": final.get("liveness"), "similarity": final.get("similarity"),
                 "challenge": sess.runner.state() if sess.runner else None, "minifas": None, "quality": None},
            timings={**sess.summary_ms(), "match": final.get("match_ms"),
                     "total": round((time.perf_counter() - m["t0"]) * 1e3, 1)}, per_frame=[], thumb=None, frames=None)
        st["final"] = final
        SESSIONS.pop(sid, None)
    return st


# ---------------- attempts / export / misc ----------------
@app.get("/api/attempts")
def attempts() -> list[dict[str, Any]]:
    return STORE.attempts()


@app.get("/api/attempts/{aid}/thumb.jpg")
def thumb(aid: str) -> Response:
    b = STORE.thumb(aid)
    if not b:
        raise HTTPException(404)
    return Response(b, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})


@app.patch("/api/attempts/{aid}")
def patch_attempt(aid: str, body: dict[str, Any]) -> dict[str, bool]:
    if "client_e2e_ms" not in body or not STORE.set_client_ms(aid, body["client_e2e_ms"]):
        raise HTTPException(404)
    _jsonl({"ts": time.time(), "id": aid, "client_e2e_ms": body["client_e2e_ms"]})
    return {"ok": True}


@app.delete("/api/attempts/{aid}")
def delete_attempt(aid: str) -> dict[str, bool]:
    if not STORE.delete_attempt(aid):
        raise HTTPException(404)
    return {"deleted": True}


@app.get("/api/export.csv")
def export_csv() -> Response:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "ts", "label", "variant", "user", "source", "decision", "reason", "minifas", "moire", "motion",
                "similarity", "match_error", "n_faces", "blur", "brightness", "face_frac", "yaw", "roll_deg",
                "ms_client_e2e", "ms_total", "ms_detect", "ms_quality", "ms_liveness", "ms_cues", "ms_match",
                "ms_decision", "saved_dir"])
    for a in STORE.attempts():
        r, t = a["raw"], a["timings"]
        q = r.get("quality") or {}
        w.writerow([a["id"], a["ts"], a["label"], a["variant"], a["user"], a["source"], a["decision"], a["reason"],
                    r.get("minifas"), r.get("moire"), r.get("motion"), r.get("similarity"), r.get("match_error"),
                    r.get("n_faces"), q.get("blur"), q.get("brightness"), q.get("face_frac"), q.get("yaw"),
                    q.get("roll_deg"), t.get("client_e2e"), t.get("total"), t.get("detect"), t.get("quality"),
                    t.get("liveness"), t.get("cues"), t.get("match"), t.get("decision"), a["saved_dir"]])
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=attempts.csv"})


@app.post("/api/clip")
def clip(label: str = Form(...), file: UploadFile = File(...)) -> dict[str, str]:
    if label not in LABELS:
        raise HTTPException(400, "bad label")
    d = EVAL / label / "clips"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{time.strftime('%Y%m%d-%H%M%S')}.webm"
    p.write_bytes(file.file.read())
    return {"saved": str(p.relative_to(ROOT))}


@app.post("/api/reset")
def reset(body: dict[str, Any] | None = None) -> dict[str, bool]:
    """Demo safety: wipe people (DB + Rekognition faces + S3 images) and attempts; eval_data only if asked."""
    PEOPLE.reset()
    STORE.reset(include_eval=bool((body or {}).get("include_eval")))
    return {"reset": True}


@app.websocket("/ws/live")
async def ws_live(ws: WebSocket) -> None:
    await ws.accept()
    q = ws.query_params
    variant = q.get("variant", "A")
    try:
        identify, _ = _identify(q.get("mode", "claim"), q.get("user") or q.get("person_id") or None)
    except HTTPException:
        identify = None
    sess = LiveSession(PIPE, identify, variant if variant in VARIANTS else "A")
    try:
        while True:
            msg = await ws.receive()
            if msg.get("bytes"):
                out = await asyncio.to_thread(sess.frame, msg["bytes"])
                if sess.who:
                    out["user"] = PEOPLE.name(sess.who)
                await ws.send_json(out)
            elif msg.get("text"):  # control: {"variant": "B"}
                ctl = json.loads(msg["text"])
                if ctl.get("variant") in VARIANTS:
                    sess.variant = ctl["variant"]
            elif msg["type"] == "websocket.disconnect":
                break
    except (WebSocketDisconnect, RuntimeError):
        pass


@app.exception_handler(ValueError)
def bad_image(_: Any, exc: ValueError) -> JSONResponse:
    return JSONResponse({"detail": str(exc)}, status_code=400)
