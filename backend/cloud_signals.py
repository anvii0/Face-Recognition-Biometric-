"""Cloud liveness signals ported from the hid-gate prototype (teammate's repo, read-only source):

- Rekognition DetectLabels: devices WITH boxes (phone/tablet/monitor/TV) -> "face centre inside a device
  box" rule; scene-level weak hints (paper, mask, ID card, poster) without boxes.
- Bedrock Converse vision liveness: {"live": 0..1, "attack": ..., "evidence": "<=10 words"}, temperature 0.
  Parsing is STRICT: one JSON object (optionally inside a ``` fence) with exactly-typed fields; anything
  else is a parse failure, which the decision layer treats like a timeout (fail closed).
- Rekognition DetectFaces extras: FACE_OCCLUDED, brightness, sharpness, yaw (second quality opinion).
- Finger-count gesture (enrollment challenge): the vision model counts raised fingers.

All checks run in parallel under one time budget; whatever has not finished is reported as "timeout".
Images stay in memory; nothing is written to disk.
"""
from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import Any, Callable

import boto3
from botocore.config import Config

# Rekognition label names whose instances carry a bounding box: a face inside one is a replay attack.
DEVICE_LABELS = {"Mobile Phone", "Phone", "Cell Phone", "Iphone", "Electronics", "Tablet Computer", "Computer",
                 "Laptop", "Pc", "Monitor", "Screen", "Display", "Television", "TV", "Computer Hardware", "Hardware"}
# Scene-level hints with no box: weak evidence only.
WEAK_LABELS = {"Paper", "Poster", "Mask", "Id Cards", "Document", "Passport", "Driving License", "Photo Frame", "Page"}
ATTACKS = {"none", "printed_photo", "screen", "mask"}

LIVENESS_PROMPT = (
    "You are a presentation-attack detector for a face-recognition entry kiosk. "
    "Decide whether the face in this webcam frame is a live person physically present, "
    "or a spoof: a printed photo, a phone/tablet/monitor screen showing a face, or a mask.\n"
    "Spoof cues: device bezel or screen edges, fingers holding a phone or sheet, paper edges, "
    "curl or creases, glare on a flat surface, moire or pixel grid, a face framed inside a "
    "rectangle, flat depthless lighting, rigid mask surface or eye cut-outs.\n"
    "Live cues: natural depth and shading, face continuous with neck, shoulders and room.\n"
    'Reply with JSON only: {"live": <0.0-1.0 probability the person is live>, '
    '"attack": "none|printed_photo|screen|mask", "evidence": "<max 10 words>"}'
)
GESTURE_PROMPT = (
    "A person at an enrolment kiosk was asked to hold up some fingers of one hand beside "
    "their face. Count the fingers clearly held up (0 if no raised hand is visible; a thumb "
    "counts as a finger). Also judge whether this is a live person physically present, not a "
    "photo, screen or mask.\n"
    'Reply with JSON only: {"fingers": <0-5>, "live": <0.0-1.0>}'
)


class ParseError(ValueError):
    """The model reply was not exactly the JSON we asked for."""


_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.S)


def parse_json_reply(text: str, schema: dict[str, type | tuple[type, ...]]) -> dict[str, Any]:
    """Strict: the whole reply must be one JSON object (a ```json fence is tolerated, nothing else),
    containing every schema key with the right type. Extra prose = ParseError."""
    m = _FENCE.match(text)
    body = (m.group(1) if m else text).strip()
    try:
        d = json.loads(body)
    except json.JSONDecodeError as e:
        raise ParseError(f"not JSON: {body[:60]!r}") from e
    if not isinstance(d, dict):
        raise ParseError("not a JSON object")
    for k, t in schema.items():
        if k not in d or isinstance(d[k], bool) or not isinstance(d[k], t):
            raise ParseError(f"missing/invalid field {k!r}")
    return d


def parse_liveness(text: str) -> dict[str, Any]:
    d = parse_json_reply(text, {"live": (int, float), "attack": str, "evidence": str})
    live = float(d["live"])
    if not 0.0 <= live <= 1.0:
        raise ParseError(f"live out of range: {live}")
    attack = d["attack"].strip().lower()
    if attack not in ATTACKS:
        raise ParseError(f"unknown attack {attack!r}")
    words = d["evidence"].split()
    return {"live": live, "attack": attack, "evidence": " ".join(words[:10])[:120]}


def parse_gesture(text: str) -> dict[str, Any]:
    d = parse_json_reply(text, {"fingers": int, "live": (int, float)})
    if not 0 <= d["fingers"] <= 5 or not 0.0 <= float(d["live"]) <= 1.0:
        raise ParseError("value out of range")
    return {"fingers": int(d["fingers"]), "live": float(d["live"])}


def _timed(fn: Callable[[], Any]) -> tuple[Any, float, str | None]:
    t0 = time.perf_counter()
    try:
        return fn(), round((time.perf_counter() - t0) * 1e3, 1), None
    except ParseError:
        return None, round((time.perf_counter() - t0) * 1e3, 1), "parse_error"
    except Exception as e:  # noqa: BLE001 - a failed check is reported, never crashes the decision
        return None, round((time.perf_counter() - t0) * 1e3, 1), type(e).__name__


class CloudSignals:
    def __init__(self, cfg: dict[str, Any], rekognition_client: Any | None = None) -> None:
        self.cfg = cfg
        bc = Config(retries={"max_attempts": 1}, connect_timeout=2, read_timeout=4, max_pool_connections=16)
        self.rk = rekognition_client or boto3.client("rekognition", region_name="us-east-1", config=bc)
        self.br = boto3.client("bedrock-runtime", region_name="us-east-1", config=bc)
        self.model = cfg["model"]
        self.pool = ThreadPoolExecutor(max_workers=12, thread_name_prefix="cloud")
        self.last_run = 0.0

    # ---------- individual checks ----------
    def labels(self, jpeg: bytes) -> dict[str, Any]:
        r = self.rk.detect_labels(Image={"Bytes": jpeg}, MaxLabels=25, MinConfidence=70)
        devices, weak = [], []
        for lab in r["Labels"]:
            if lab["Name"] in DEVICE_LABELS:
                devices += [{"name": lab["Name"], "conf": round(i["Confidence"], 1), "box": i["BoundingBox"]}
                            for i in lab.get("Instances", [])]
            elif lab["Name"] in WEAK_LABELS and lab["Confidence"] >= 80:
                weak.append(lab["Name"])
        return {"devices": devices, "weak": weak}

    def faces(self, jpeg: bytes) -> list[dict[str, Any]]:
        r = self.rk.detect_faces(Image={"Bytes": jpeg}, Attributes=["DEFAULT", "FACE_OCCLUDED"])
        return [{"box": f["BoundingBox"], "yaw": round(f["Pose"]["Yaw"], 1),
                 "brightness": round(f["Quality"]["Brightness"], 1), "sharpness": round(f["Quality"]["Sharpness"], 1),
                 "occluded": bool(f.get("FaceOccluded", {}).get("Value", False))} for f in r["FaceDetails"]]

    def _converse(self, jpeg: bytes, prompt: str, max_tokens: int) -> str:
        r = self.br.converse(modelId=self.model, messages=[{"role": "user", "content": [
            {"image": {"format": "jpeg", "source": {"bytes": jpeg}}}, {"text": prompt}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0})
        return r["output"]["message"]["content"][0]["text"]

    def vlm(self, jpeg: bytes) -> dict[str, Any]:
        return parse_liveness(self._converse(jpeg, LIVENESS_PROMPT, 100))

    def gesture(self, jpeg: bytes) -> dict[str, Any]:
        return parse_gesture(self._converse(jpeg, GESTURE_PROMPT, 60))

    # ---------- parallel run under a budget ----------
    def start(self, jpeg: bytes) -> dict[str, Future]:
        """Launch all checks now; collect later with finish() so they overlap local work."""
        self.last_run = time.time()
        return {k: self.pool.submit(_timed, lambda f=f: f(jpeg))
                for k, f in (("vlm", self.vlm), ("labels", self.labels), ("faces", self.faces))}

    def finish(self, futs: dict[str, Future], deadline: float) -> dict[str, Any]:
        """Wait until `deadline` (perf_counter seconds); unfinished checks are 'timeout' (fail closed)."""
        wait(futs.values(), timeout=max(0.0, deadline - time.perf_counter()))
        out: dict[str, Any] = {"vlm": None, "devices": None, "weak": None, "faces": None, "errors": {}, "ms": {}}
        for k, f in futs.items():
            val, ms, err = f.result() if f.done() else (None, None, "timeout")
            out["ms"][k] = ms
            if err:
                out["errors"][k] = err
            elif k == "labels":
                out["devices"], out["weak"] = val["devices"], val["weak"]
            else:
                out[k] = val
        return out

    # ---------- keep-warm ----------
    def keep_warm(self, interval_s: float) -> None:
        """Measured: cold image Converse ~2.6 s; after a text-only warm-up ~1.4 s; warm ~1.0 s.
        Keep the Bedrock connection open while idle (Rekognition is warmed by the matcher)."""
        def loop() -> None:
            while True:
                if time.time() - self.last_run > 3:
                    try:
                        self.br.converse(modelId=self.model, messages=[{"role": "user", "content": [{"text": "ok"}]}],
                                         inferenceConfig={"maxTokens": 1})
                    except Exception:  # noqa: BLE001
                        pass
                time.sleep(interval_s)
        threading.Thread(target=loop, daemon=True, name="bedrock-warm").start()

