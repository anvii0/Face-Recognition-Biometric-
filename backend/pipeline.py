"""Verify / live / enrollment / step-up orchestration with per-stage timing (time.perf_counter).
All flows share Pipeline.analyze() and decision.decide(): there is one pipeline, not one per screen."""
from __future__ import annotations

import base64
import random
import statistics
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

import cv2
import numpy as np

from .challenge import ChallengeRunner, Landmarker
from .decision import ALLOW, DENY, decide, quality_failures
from .matcher import RekognitionMatcher
from .vision import Face, Vision, decode_jpeg, motion_score

_pool = ThreadPoolExecutor(max_workers=4)
# identify(probe_jpeg) -> (similarity | None, error | None, person_id | None)
Identify = Callable[[bytes], tuple[float | None, str | None, str | None]]


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1e3, 1)


def png_b64(img: np.ndarray, size: int | None = None) -> str:
    if size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
    return base64.b64encode(cv2.imencode(".png", img)[1].tobytes()).decode()


def jpeg(img: np.ndarray, q: int = 90) -> bytes:
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()


def median_quality(qs: list[dict[str, float]]) -> dict[str, float] | None:
    return {k: float(statistics.median(q[k] for q in qs)) for k in qs[0]} if qs else None


def face_thumb(img: np.ndarray, face: Face, size: int = 96, pad: float = 0.15) -> bytes:
    x, y, w, h = face.box
    H, W = img.shape[:2]
    p = int(max(w, h) * pad)
    crop = img[max(y - p, 0):min(y + h + p, H), max(x - p, 0):min(x + w + p, W)]
    return jpeg(cv2.resize(crop, (size, size)), 85)


COACH = {  # quality failure prefix -> friendly kiosk microcopy
    "too dark": "Too dark — find more light", "overexposed": "Too bright — step out of direct light",
    "blurry": "Hold still for a moment", "face too small": "Move a little closer",
    "head turned": "Look straight at the camera", "head tilted": "Keep your head level",
}


def coach(fails: list[str], n_faces: int) -> str:
    if n_faces == 0:
        return "Step into the frame"
    if n_faces > 1:
        return "Only one person at a time, please"
    for f in fails:
        for k, msg in COACH.items():
            if f.startswith(k):
                return msg
    return "Looking good — hold still"


@dataclass
class FrameInfo:
    img: np.ndarray
    faces: list[Face]
    quality: dict[str, float] | None = None
    fails: list[str] = field(default_factory=list)
    liveness: float | None = None
    moire: float | None = None
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def face(self) -> Face | None:
        return self.faces[0] if self.faces else None

    def box_norm(self) -> list[float] | None:
        if not self.face:
            return None
        x, y, w, h = self.face.box
        H, W = self.img.shape[:2]
        return [x / W, y / H, w / W, h / H]


class Pipeline:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        d = cfg["detector"]
        self.vision = Vision(d["score_threshold"], d["max_side"])
        m = cfg["match"]
        self.matcher = RekognitionMatcher(m["region"], m["collection"]) if m["provider"] == "rekognition" else None
        self._landmarker: Landmarker | None = None
        self.reference_for_warmup: Callable[[], bytes | None] = lambda: None  # set by the app
        self.warm_kick = None
        if self.matcher is not None:
            self.warm_kick = self.matcher.keep_warm(lambda: self.reference_for_warmup(), m.get("keepalive_s", 120))

    @property
    def landmarker(self) -> Landmarker:
        if self._landmarker is None:
            self._landmarker = Landmarker()
        return self._landmarker

    def match_async(self, identify: Identify | None, probe: bytes) -> Future | None:
        return _pool.submit(identify, probe) if identify is not None else None

    # ---------- shared per-frame analysis ----------
    def analyze(self, data: bytes, *, liveness: bool = True, moire: bool = True,
                crops: dict[str, np.ndarray] | None = None) -> FrameInfo:
        tm: dict[str, float] = {}
        t0 = time.perf_counter()
        img = self.vision.downscale(decode_jpeg(data))
        tm["decode"] = _ms(t0)
        t0 = time.perf_counter()
        fi = FrameInfo(img, self.vision.detect(img), timings=tm)
        tm["detect"] = _ms(t0)
        if fi.face:
            t0 = time.perf_counter()
            fi.quality = self.vision.quality(img, fi.face)
            fi.fails = quality_failures(fi.quality, self.cfg["quality"])
            tm["quality"] = _ms(t0)
            if liveness:
                t0 = time.perf_counter()
                fi.liveness = self.vision.liveness(img, fi.face, crops)
                tm["liveness"] = _ms(t0)
            if moire:
                t0 = time.perf_counter()
                fi.moire = self.vision.moire(img, fi.face, crops)
                tm["cues"] = _ms(t0)
        return fi

    # ---------- full verify (kiosk + bench) ----------
    def verify(self, jpegs: list[bytes], *, identify: Identify | None, variant: str,
               no_identity_reason: str = "no enrolled reference",
               name_of: Callable[[str], str] = lambda pid: pid) -> dict[str, Any]:
        t_all = time.perf_counter()
        tm: dict[str, float] = {"decode": 0.0, "detect": 0.0, "quality": 0.0, "liveness": 0.0, "cues": 0.0}

        # Stage 1: decode + detect + quality for every frame; the sharpest becomes the identity probe.
        frames = [self.analyze(b, liveness=False, moire=False) for b in jpegs]
        for f in frames:
            for k in ("decode", "detect", "quality"):
                tm[k] += f.timings.get(k, 0.0)
        with_face = [i for i, f in enumerate(frames) if f.face]
        best = max(with_face, key=lambda i: frames[i].quality["blur"]) if with_face else None  # type: ignore[index]

        # Stage 2: identity (network-bound) runs in parallel with liveness.
        t_match = time.perf_counter()
        fut = self.match_async(identify, jpeg(frames[best].img)) if best is not None else None

        crops: dict[str, np.ndarray] = {}
        per_frame: list[dict[str, Any]] = []
        lms = []
        for i, f in enumerate(frames):
            pf: dict[str, Any] = {"n_faces": len(f.faces), "quality": f.quality}
            if f.face:
                t0 = time.perf_counter()
                pf["liveness"] = self.vision.liveness(f.img, f.face, crops if i == best else None)
                tm["liveness"] += _ms(t0)
                t0 = time.perf_counter()
                pf["moire"] = self.vision.moire(f.img, f.face, crops if i == best else None)
                tm["cues"] += _ms(t0)
                pf["box"] = f.box_norm()
                lms.append(f.face.landmarks)
            per_frame.append(pf)
        t0 = time.perf_counter()
        motion = motion_score(lms)
        tm["cues"] += _ms(t0)

        sim, err, who = None, (no_identity_reason if identify is None else None), None
        if fut is not None:
            sim, err, who = fut.result()
        tm["match"] = _ms(t_match)  # wall time from launch; overlaps liveness + cues

        t0 = time.perf_counter()
        lives = [p["liveness"] for p in per_frame if "liveness" in p]
        moires = [p["moire"] for p in per_frame if "moire" in p]
        raw = {"n_faces": max((len(f.faces) for f in frames), default=0),
               "quality": median_quality([f.quality for f in frames if f.quality]),
               "minifas": float(np.mean(lives)) if lives else None,
               "moire": float(np.mean(moires)) if moires else None,
               "motion": motion, "similarity": sim, "match_error": err,
               "user": name_of(who) if who else None, "person_id": who,
               "n_frames": len(jpegs), "frames_with_face": len(with_face)}
        result = decide(raw, self.cfg, variant)
        tm["decision"] = _ms(t0)
        tm = {k: round(v, 1) for k, v in tm.items()}
        tm["total"] = _ms(t_all)

        debug, thumb = {}, None
        if best is not None:
            debug = {k: png_b64(v, 160 if k.startswith("minifas") else None) for k, v in crops.items()}
            thumb = face_thumb(frames[best].img, frames[best].face)  # type: ignore[arg-type]
        return {**result, "raw": raw, "timings": tm, "per_frame": per_frame, "debug": debug, "thumb": thumb}

    # ---------- quick enrollment check (bench) ----------
    def check_enroll(self, jpegs: list[bytes]) -> tuple[int, list[str], list[FrameInfo]]:
        problems, blurs, infos = [], [], []
        for i, b in enumerate(jpegs):
            f = self.analyze(b, moire=False)
            infos.append(f)
            if len(f.faces) != 1:
                problems.append(f"capture {i + 1}: {len(f.faces)} faces")
                blurs.append(-1.0)
                continue
            if f.fails:
                problems.append(f"capture {i + 1}: " + "; ".join(f.fails))
            blurs.append(f.quality["blur"])  # type: ignore[index]
        return int(np.argmax(blurs)), problems, infos


class LiveSession:
    """Per-WebSocket rolling state. The client waits for each reply before sending the next frame, so the
    backend never queues and always works on the latest frame."""

    def __init__(self, pipe: Pipeline, identify: Identify | None, variant: str) -> None:
        self.p, self.identify, self.variant = pipe, identify, variant
        self.lives: deque[float] = deque(maxlen=8)
        self.moires: deque[float] = deque(maxlen=8)
        self.lms: deque[np.ndarray] = deque(maxlen=5)
        self.quals: deque[dict[str, float]] = deque(maxlen=5)
        self.sim: float | None = None
        self.who: str | None = None
        self.match_err: str | None = "pending" if identify else "no identity check in live mode"
        self.match_fut: Future | None = None
        self.last_match = 0.0

    def frame(self, data: bytes) -> dict[str, Any]:
        t0 = time.perf_counter()
        f = self.p.analyze(data)
        out: dict[str, Any] = {"n_faces": len(f.faces), "w": f.img.shape[1], "h": f.img.shape[0],
                               "coach": coach(f.fails, len(f.faces))}
        if self.match_fut is not None and self.match_fut.done():
            self.sim, self.match_err, self.who = self.match_fut.result()
            self.match_fut = None
        if f.face:
            self.lives.append(f.liveness)  # type: ignore[arg-type]
            self.moires.append(f.moire)  # type: ignore[arg-type]
            self.lms.append(f.face.landmarks)
            self.quals.append(f.quality)  # type: ignore[arg-type]
            out.update(box=f.box_norm(), quality=f.quality, liveness=f.liveness, moire=f.moire, fails=f.fails)
            if (self.identify and self.match_fut is None
                    and time.time() - self.last_match >= self.p.cfg["match"]["live_interval_s"]):
                self.match_fut = self.p.match_async(self.identify, jpeg(f.img))
                self.last_match = time.time()
        else:
            self.lives.clear(); self.lms.clear(); self.quals.clear(); self.moires.clear()  # noqa: E702
        raw = {"n_faces": len(f.faces), "quality": median_quality(list(self.quals)) if f.face else None,
               "minifas": float(np.mean(self.lives)) if self.lives else None,
               "moire": float(np.mean(self.moires)) if self.moires else None,
               "motion": motion_score(list(self.lms)), "similarity": self.sim,
               "match_error": self.match_err, "user": self.who}
        out["rolling"] = decide(raw, self.p.cfg, self.variant)
        out["similarity"] = self.sim
        out["server_ms"] = _ms(t0)
        return out


class ChallengeSession:
    """Shared engine for liveness-gated enrollment and verify step-up.

    Phases: 'center' (collect N good frontal frames; passive liveness must average >= min_liveness)
            -> 'challenge' (randomised active challenges; liveness must stay >= t_lo)
            -> 'passed' | 'failed'. Frames live only in memory and are dropped on failure."""

    def __init__(self, pipe: Pipeline, *, kinds: list[str], center_frames: int, min_liveness: float,
                 purpose: str, meta: dict[str, Any] | None = None, rng: random.Random | None = None) -> None:
        self.p, self.purpose, self.meta = pipe, purpose, meta or {}
        self.cc = pipe.cfg["challenge"]
        self.kinds, self.need, self.min_live, self.rng = kinds, center_frames, min_liveness, rng
        self.phase = "center" if center_frames > 0 else "challenge"
        self.runner = ChallengeRunner(kinds, self.cc, rng) if self.phase == "challenge" else None
        self.good: list[tuple[bytes, FrameInfo]] = []  # center-phase frames (candidate references)
        self.live_scores: list[float] = []
        self.challenge_live: list[float] = []
        self.reason = ""
        self.created = time.time()
        self.stage_ms: dict[str, list[float]] = {"analyze": [], "landmarks": []}
        self.challenge_jpeg: bytes | None = None

    def frame(self, data: bytes) -> dict[str, Any]:
        if self.phase in ("passed", "failed"):
            return self.state()
        if time.time() - self.created > self.cc["session_timeout_s"]:
            return self._fail("session timed out")
        t0 = time.perf_counter()
        f = self.p.analyze(data, moire=False)
        self.stage_ms["analyze"].append(_ms(t0))
        tip = coach(f.fails, len(f.faces))
        frontal = f.quality is not None and abs(f.quality["yaw"]) <= self.cc["frontal_yaw"]
        if self.phase == "center":
            ok = len(f.faces) == 1 and not f.fails and frontal
            if ok:
                self.good.append((data, f))
                self.live_scores.append(f.liveness)  # type: ignore[arg-type]
            if len(self.good) >= self.need:
                mean = float(np.mean(self.live_scores))
                if mean < self.min_live:
                    return self._fail(f"passive liveness {mean:.2f} < {self.min_live:.2f}")
                self.phase = "challenge"
                self.runner = ChallengeRunner(self.kinds, self.cc, self.rng)
            return self.state("Hold still…" if ok else tip, f)
        # challenge phase
        sig = None
        if len(f.faces) > 1:
            return self._fail("more than one face during challenge")
        if len(f.faces) == 1:
            t0 = time.perf_counter()
            sig = self.p.landmarker.signals(f.img)
            self.stage_ms["landmarks"].append(_ms(t0))
            self.challenge_live.append(f.liveness)  # type: ignore[arg-type]
            if self.challenge_jpeg is None and not f.fails and frontal:
                self.challenge_jpeg = data  # a frontal frame from the challenge, for the identity check
        st = self.runner.update(sig, f.quality["yaw"] if f.quality else None)  # type: ignore[union-attr]
        recent = self.challenge_live[-10:]
        if len(recent) >= 5 and float(np.mean(recent)) < self.p.cfg["thresholds"]["t_lo"]:
            return self._fail(f"liveness dropped during challenge ({np.mean(recent):.2f})")
        if st["status"] == "failed":
            return self._fail(st["message"])
        if st["status"] == "passed":
            self.phase = "passed"
        return self.state(st["prompt"] or tip, f, signals=sig)

    def _fail(self, reason: str) -> dict[str, Any]:
        self.phase, self.reason = "failed", reason
        self.good.clear()  # never keep images from a failed attempt
        self.challenge_jpeg = None
        return self.state()

    def best_references(self, k: int = 4) -> list[tuple[bytes, FrameInfo]]:
        """Sharpest, most frontal center-phase frames."""
        return sorted(self.good, key=lambda g: g[1].quality["blur"] * (1 - min(abs(g[1].quality["yaw"]), 0.9)),  # type: ignore[index]
                      reverse=True)[:k]

    def summary_ms(self) -> dict[str, float]:
        return {f"{k}_p50": round(float(np.median(v)), 1) for k, v in self.stage_ms.items() if v}

    def state(self, prompt: str = "", f: FrameInfo | None = None,
              signals: dict[str, float] | None = None) -> dict[str, Any]:
        out: dict[str, Any] = {"phase": self.phase, "prompt": prompt, "reason": self.reason,
                               "progress": min(1.0, len(self.good) / self.need) if self.need else 1.0,
                               "liveness_avg": round(float(np.mean(self.live_scores)), 3) if self.live_scores else None}
        if self.runner:
            ch = self.runner.state()
            out["challenge"] = ch
            out["progress"] = len(ch["done"]) / max(len(ch["done"]) + ch["remaining"], 1)
        if f is not None:
            out.update(box=f.box_norm(), quality=f.quality, liveness=f.liveness, n_faces=len(f.faces))
        if signals:
            out["signals"] = {k: round(v, 2) for k, v in signals.items()}
        return out


def stepup_decision(sess: ChallengeSession, identify: Identify | None, cfg: dict[str, Any]) -> dict[str, Any]:
    """Final decision after a step-up challenge: challenge passed, liveness held, identity matches."""
    t = cfg["thresholds"]
    if sess.phase != "passed":
        return {"decision": DENY, "confidence": 1.0, "reason": f"Denied: step-up failed — {sess.reason or 'not completed'}"}
    live = float(np.mean(sess.challenge_live)) if sess.challenge_live else 0.0
    if sess.challenge_jpeg is None or identify is None:
        return {"decision": DENY, "confidence": 0.5, "reason": "Denied: step-up passed but no usable frontal frame for identity"}
    t0 = time.perf_counter()
    sim, err, who = identify(sess.challenge_jpeg)
    ms = _ms(t0)
    base = {"match_ms": ms, "liveness": round(live, 4), "similarity": sim, "user": who}
    if sim is None:
        return {**base, "decision": DENY, "confidence": 0.5, "reason": f"Denied: identity check failed after step-up ({err})"}
    if sim < t["t_match"]:
        return {**base, "decision": DENY, "confidence": round(1 - sim / 100, 4),
                "reason": f"Denied: step-up passed but face does not match (similarity {sim:.1f} < {t['t_match']:.0f})"}
    done = " + ".join(sess.runner.done) if sess.runner else ""
    return {**base, "decision": ALLOW, "confidence": round(min(live, sim / 100), 4),
            "reason": f"Allowed after step-up: {done} verified, liveness {live:.2f}, similarity {sim:.1f}"}
