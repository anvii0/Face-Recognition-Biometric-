"""Active liveness challenges (blink / turn left / turn right / smile / fingers), verified SERVER-side.

Signals: blink + smile from MediaPipe Face Landmarker blendshapes; finger count from the Bedrock vision model
(ported from hid-gate) plus CompareFaces against the session reference; head turn from the YuNet yaw ratio
(nose offset from eye midpoint / inter-eye distance) on UN-mirrored frames, where the user turning to
THEIR left moves the nose towards image-right (positive yaw). `left_yaw_sign` in config flips this if a
camera delivers mirrored frames.
"""
from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

MODEL = Path(__file__).resolve().parents[1] / "models" / "face_landmarker.task"
PROMPTS = {"blink": "Blink slowly", "turn_left": "Turn your head to your left",
           "turn_right": "Turn your head to your right", "smile": "Give us a smile",
           "fingers": "Hold up {n} finger(s) beside your face"}


class Landmarker:
    """MediaPipe Face Landmarker (IMAGE mode) — ~15 ms/frame on CPU."""

    def __init__(self) -> None:
        from mediapipe.tasks.python import BaseOptions, vision  # heavy import; only when used
        import mediapipe as mp
        self._mp = mp
        self._lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL)), output_face_blendshapes=True,
            num_faces=2, running_mode=vision.RunningMode.IMAGE))
        self._lock = threading.Lock()

    def signals(self, bgr: np.ndarray) -> dict[str, float] | None:
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        with self._lock:
            r = self._lm.detect(img)
        if len(r.face_blendshapes) != 1:
            return None
        b = {c.category_name: float(c.score) for c in r.face_blendshapes[0]}
        return {"blink_l": b["eyeBlinkLeft"], "blink_r": b["eyeBlinkRight"],
                "smile": (b["mouthSmileLeft"] + b["mouthSmileRight"]) / 2}


@dataclass
class _Tracker:
    kind: str
    cfg: dict[str, float]
    stage: int = 0          # 0 = need neutral/open/frontal, 1 = need action, 2 = (blink) need re-open
    hits: int = 0
    started: float = field(default_factory=time.time)
    target: int = 0         # fingers: how many to hold up (1-5, random)
    last_seen: int | None = None

    @property
    def timeout(self) -> float:
        return float(self.cfg["gesture_timeout_s"] if self.kind == "fingers" else self.cfg["timeout_s"])

    def prompt(self) -> str:
        return PROMPTS[self.kind].format(n=self.target)

    def update(self, sig: dict[str, Any] | None, yaw: float | None) -> bool:
        c = self.cfg
        if self.kind == "fingers":
            g = (sig or {}).get("gesture")
            if not g:
                return False
            self.last_seen = g.get("fingers")
            return (g.get("fingers") == self.target and g.get("live", 0.0) > c["vlm_spoof_max"]
                    and g.get("same", 0.0) >= c["gesture_same_min"])
        if self.kind == "blink" and sig and "blink_l" in sig:
            closed = min(sig["blink_l"], sig["blink_r"]) >= c["blink_closed"]
            opened = max(sig["blink_l"], sig["blink_r"]) <= c["blink_open"]
            if self.stage == 0 and opened: self.stage = 1  # noqa: E701
            elif self.stage == 1 and closed: self.stage = 2  # noqa: E701
            elif self.stage == 2 and opened: return True  # noqa: E701
        elif self.kind == "smile" and sig and "smile" in sig:
            if self.stage == 0 and sig["smile"] <= c["smile_neutral"]: self.stage = 1  # noqa: E701
            elif self.stage == 1 and sig["smile"] >= c["smile_on"]:
                self.hits += 1
                return self.hits >= 2
        elif self.kind in ("turn_left", "turn_right") and yaw is not None:
            signed = yaw * c["left_yaw_sign"] * (1 if self.kind == "turn_left" else -1)
            if self.stage == 0 and abs(yaw) <= c["frontal_yaw"]: self.stage = 1  # noqa: E701
            elif self.stage == 1 and signed >= c["turn_yaw"]:
                self.hits += 1
                return self.hits >= 2  # sustained for 2 frames
            elif self.stage == 1:
                self.hits = 0
        return False


class ChallengeRunner:
    """Runs a randomised sequence of challenges with a per-challenge timeout and limited retries."""

    def __init__(self, kinds: list[str], cfg: dict[str, Any], rng: random.Random | None = None,
                 pool: list[str] | None = None) -> None:
        self.rng = rng or random.Random()
        self.cfg = cfg
        self.pending = list(kinds)
        self.rng.shuffle(self.pending)
        self.done: list[str] = []
        self.pool = list(pool or kinds)  # retries draw replacements from the configured pool only
        self.retries = int(cfg["retries"])
        self.status = "running"
        self.message = ""
        self.cur = self._tracker(self.pending.pop(0))

    def _tracker(self, kind: str) -> _Tracker:
        return _Tracker(kind, self.cfg, target=self.rng.randint(1, 5) if kind == "fingers" else 0)

    def update(self, sig: dict[str, Any] | None, yaw: float | None, now: float | None = None) -> dict[str, Any]:
        now = now or time.time()
        if self.status == "running":
            if self.cur.update(sig, yaw):
                self.done.append(self.cur.kind)
                if self.pending:
                    self.cur = self._tracker(self.pending.pop(0))
                else:
                    self.status = "passed"
            elif now - self.cur.started > self.cur.timeout:
                if self.retries > 0:
                    self.retries -= 1
                    self.message = "Didn't catch that — let's try again"
                    choices = [k for k in self.pool if k != self.cur.kind and k not in self.done
                               and k not in self.pending] or [self.cur.kind]
                    self.cur = self._tracker(self.rng.choice(choices))
                else:
                    self.status = "failed"
                    seen = f" (saw {self.cur.last_seen})" if self.cur.kind == "fingers" and self.cur.last_seen is not None else ""
                    self.message = f"challenge '{self.cur.kind}' not completed within {self.cur.timeout:.0f} s{seen}"
        return self.state(now)

    def state(self, now: float | None = None) -> dict[str, Any]:
        now = now or time.time()
        return {"status": self.status, "current": None if self.status != "running" else self.cur.kind,
                "prompt": self.cur.prompt() if self.status == "running" else "",
                "target": self.cur.target if self.status == "running" and self.cur.kind == "fingers" else None,
                "done": list(self.done), "remaining": len(self.pending) + (self.status == "running"),
                "time_left_s": max(0.0, round(self.cur.timeout - (now - self.cur.started), 1)),
                "message": self.message}
