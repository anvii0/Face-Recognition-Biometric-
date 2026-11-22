"""Decision layer: pure function of RAW scores + config. Mirrored 1:1 in frontend/decision.js
(the lab re-decides stored attempts when sliders move); tests/test_decision_parity.py keeps them equal.

Variants: A MiniFASNet | B A+moire | C B+motion | D cloud-only (hid-gate rules) | E fused local+cloud.
D and E read raw["cloud"] = {vlm, devices, weak, faces, errors, ms} and fail closed on missing signals."""
from __future__ import annotations

from typing import Any

ALLOW, STEP_UP, DENY = "ALLOW", "REQUEST_VERIFICATION", "DENY"


def _clip(v: float) -> float:
    return max(0.0, min(1.0, v))


def fuse(raw: dict[str, Any], fusion: dict[str, float], variant: str) -> dict[str, float]:
    """Returns fused liveness and the penalty contributed by each cue."""
    live = float(raw["minifas"])
    moire_pen = motion_pen = 0.0
    if variant in ("B", "C") and raw.get("moire") is not None:
        span = (fusion["moire_hi"] - fusion["moire_lo"]) or 1e-9
        moire_pen = fusion["w_moire"] * _clip((raw["moire"] - fusion["moire_lo"]) / span)
    if variant == "C" and raw.get("motion") is not None:
        motion_pen = fusion["w_motion"] * (1 - _clip(raw["motion"] / fusion["motion_ref"]))
    return {"liveness": _clip(live - moire_pen - motion_pen), "moire_pen": moire_pen, "motion_pen": motion_pen}


def quality_failures(q: dict[str, float], qc: dict[str, float]) -> list[str]:
    fails = []
    if q["brightness"] < qc["min_brightness"]:
        fails.append(f"too dark (brightness {q['brightness']:.0f} < {qc['min_brightness']:.0f})")
    if q["brightness"] > qc["max_brightness"]:
        fails.append(f"overexposed (brightness {q['brightness']:.0f} > {qc['max_brightness']:.0f})")
    if q["blur"] < qc["min_blur"]:
        fails.append(f"blurry (sharpness {q['blur']:.0f} < {qc['min_blur']:.0f})")
    if q["face_frac"] < qc["min_face_frac"]:
        fails.append(f"face too small ({q['face_frac']:.2f} < {qc['min_face_frac']:.2f} of frame)")
    if abs(q["yaw"]) > qc["max_yaw"]:
        fails.append(f"head turned (yaw {q['yaw']:.2f}, max {qc['max_yaw']:.2f})")
    if abs(q["roll_deg"]) > qc["max_roll_deg"]:
        fails.append(f"head tilted ({q['roll_deg']:.0f} deg, max {qc['max_roll_deg']:.0f})")
    return fails


def decide_local(raw: dict[str, Any], cfg: dict[str, Any], variant: str) -> dict[str, Any]:
    """Variants A/B/C. raw keys: n_faces (max faces in any frame), quality (median metrics) | None, minifas, moire,
    motion, similarity (0-100 | None), match_error (str | None), user (str | None)."""
    t = cfg["thresholds"]

    def out(decision: str, confidence: float, reason: str, live: float | None = None) -> dict[str, Any]:
        return {"decision": decision, "confidence": round(confidence, 4), "reason": reason,
                "liveness": None if live is None else round(live, 4), "variant": variant}

    if raw.get("n_faces", 0) == 0 or raw.get("quality") is None:
        return out(STEP_UP, 0.0, "Step-up: no face detected")
    if raw["n_faces"] > 1:
        return out(STEP_UP, 0.0, f"Step-up: {raw['n_faces']} faces in view, expected 1")
    fails = quality_failures(raw["quality"], cfg["quality"])
    f = fuse(raw, cfg["fusion"], variant)
    live = f["liveness"]
    if fails:
        return out(STEP_UP, 0.0, "Step-up: low quality — " + "; ".join(fails), live)
    if live < t["t_lo"]:
        if f["moire_pen"] >= max(f["motion_pen"], 0.1):
            why = f"screen replay suspected, moire score {raw['moire']:.2f}"
        elif f["motion_pen"] >= 0.1:
            why = f"no natural face motion (motion {raw['motion']:.3f})"
        else:
            why = "presentation attack suspected"
        return out(DENY, 1 - live, f"Denied: {why} (liveness {live:.2f} < {t['t_lo']:.2f})", live)
    sim = raw.get("similarity")
    if sim is None:
        err = raw.get("match_error") or "no face found by matcher"
        return out(STEP_UP, 0.0, f"Step-up: identity could not be checked ({err})", live)
    who = raw.get("user") or "enrolled user"
    if sim < t["t_match_lo"]:
        return out(DENY, 1 - sim / 100,
                   f"Denied: face does not match {who} (similarity {sim:.1f} < {t['t_match_lo']:.0f})", live)
    trust = min(live, sim / 100)
    if live >= t["t_hi"] and sim >= t["t_match"]:
        return out(ALLOW, trust, f"Allowed: live face (liveness {live:.2f}) matched {who} (similarity {sim:.1f})", live)
    parts = []
    if live < t["t_hi"]:
        parts.append(f"liveness {live:.2f} below {t['t_hi']:.2f}")
    if sim < t["t_match"]:
        parts.append(f"similarity {sim:.1f} below {t['t_match']:.0f}")
    return out(STEP_UP, trust, "Step-up: borderline — " + "; ".join(parts), live)


# ======================= cloud signals (variants D and E) =======================
VARIANTS = ("A", "B", "C", "D", "E")


def decide(raw: dict[str, Any], cfg: dict[str, Any], variant: str) -> dict[str, Any]:
    if variant == "D":
        return decide_cloud(raw, cfg)
    if variant == "E":
        return decide_fused(raw, cfg)
    return decide_local(raw, cfg, variant)


def _result(decision: str, confidence: float, reason: str, live: float | None, variant: str) -> dict[str, Any]:
    return {"decision": decision, "confidence": round(confidence, 4), "reason": reason,
            "liveness": None if live is None else round(live, 4), "variant": variant}


def face_centre_in_box(face_box: dict[str, float], box: dict[str, float]) -> bool:
    cx = face_box["Left"] + face_box["Width"] / 2
    cy = face_box["Top"] + face_box["Height"] / 2
    return box["Left"] <= cx <= box["Left"] + box["Width"] and box["Top"] <= cy <= box["Top"] + box["Height"]


def device_over_face(raw: dict[str, Any]) -> dict[str, Any] | None:
    """First detected device whose box contains the face centre. Face box: Rekognition's when it saw exactly
    one face, else our detector's (both normalised to the same probe frame)."""
    c = raw.get("cloud") or {}
    faces = c.get("faces") or []
    face = faces[0]["box"] if len(faces) == 1 else raw.get("face_box")
    if not face:
        return None
    return next((d for d in (c.get("devices") or []) if face_centre_in_box(face, d["box"])), None)


def _missing(c: dict[str, Any], key: str) -> str | None:
    """Why a cloud signal is unusable (timeout / parse_error / exception name), or None if present."""
    err = (c.get("errors") or {}).get(key)
    if err:
        return err
    present = c.get("vlm") if key == "vlm" else c.get("devices") if key == "labels" else c.get("faces")
    return None if present is not None else "missing"


def _rk_quality_tips(face: dict[str, Any], cc: dict[str, float]) -> list[str]:
    tips = []
    if face["brightness"] < cc["rk_min_brightness"]:
        tips.append(f"too dark (Rekognition brightness {face['brightness']:.0f})")
    if face["sharpness"] < cc["rk_min_sharpness"]:
        tips.append(f"blurry (Rekognition sharpness {face['sharpness']:.0f})")
    if face["box"]["Width"] < cc["rk_min_face_width"]:
        tips.append("face too far away")
    if face.get("occluded"):
        tips.append("face partly covered")
    return tips


def decide_cloud(raw: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """Variant D: hid-gate's rules as ported (VLM + devices + Rekognition faces + our identity score)."""
    cc, v = cfg["cloud"], "D"
    c = raw.get("cloud")
    if not c:
        return _result(STEP_UP, 0.0, "Step-up: cloud signals were not collected", None, v)
    vlm = c.get("vlm")
    p = vlm["live"] if vlm else None
    if _missing(c, "faces"):
        return _result(STEP_UP, 0.0, f"Step-up: face check unavailable ({_missing(c, 'faces')})", p, v)
    faces = c["faces"]
    if not faces:
        return _result(STEP_UP, 0.0, "Step-up: no face found by Rekognition", p, v)
    if len(faces) > 1:
        return _result(STEP_UP, 0.0, f"Step-up: {len(faces)} faces in view, expected 1", p, v)
    dev = device_over_face(raw)
    if p is not None and p <= cc["vlm_spoof_max"]:
        attack = vlm["attack"] if vlm["attack"] != "none" else "spoof"
        extra = f"; face is inside a detected {dev['name']}" if dev else ""
        return _result(DENY, 1 - p, f'Denied: {attack} suspected — "{vlm["evidence"]}" (VLM live {p:.2f}){extra}', p, v)
    if dev and (p is None or p < cc["vlm_live_min"]):
        return _result(DENY, dev["conf"] / 100, f"Denied: screen replay suspected — face is inside a detected {dev['name']}", p, v)
    tips = _rk_quality_tips(faces[0], cc)
    if tips:
        return _result(STEP_UP, 0.0, "Step-up: low quality — " + "; ".join(tips), p, v)
    sim = raw.get("similarity")
    if sim is None:
        return _result(STEP_UP, 0.0, f"Step-up: identity could not be checked ({raw.get('match_error') or 'no result'})", p, v)
    who = raw.get("user") or "enrolled user"
    if sim < cc["d_sim_min"]:
        return _result(DENY, 1 - sim / 100, f"Denied: face does not match {who} (similarity {sim:.1f} < {cc['d_sim_min']:.0f})", p, v)
    doubts = []
    if p is None:
        doubts.append(f"liveness model unavailable ({_missing(c, 'vlm')})")
    elif p < cc["vlm_live_min"]:
        doubts.append(f'liveness uncertain (VLM {p:.2f}: "{vlm["evidence"]}")')
    if dev:
        doubts.append(f"a {dev['name']} overlaps the face")
    if c.get("weak"):
        doubts.append("possible spoof material in view: " + ", ".join(c["weak"]))
    if _missing(c, "labels"):
        doubts.append(f"object check unavailable ({_missing(c, 'labels')})")
    if sim < cc["d_sim_strong"]:
        doubts.append(f"identity match borderline ({sim:.1f})")
    if doubts:
        return _result(STEP_UP, min(p or 0.0, sim / 100), "Step-up: " + "; ".join(doubts), p, v)
    return _result(ALLOW, min(p, sim / 100), f'Allowed: live person (VLM {p:.2f}: "{vlm["evidence"]}") matched {who} (similarity {sim:.1f})', p, v)  # type: ignore[arg-type]


def decide_fused(raw: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """Variant E: local (MiniFASNet + moire, i.e. variant-B fusion) AND cloud (VLM + devices), fail closed.
    DENY on any strong spoof evidence or identity failure; ALLOW only when every signal agrees; else step-up."""
    t, cc, v = cfg["thresholds"], cfg["cloud"], "E"
    if raw.get("n_faces", 0) == 0 or raw.get("quality") is None:
        return _result(STEP_UP, 0.0, "Step-up: no face detected", None, v)
    if raw["n_faces"] > 1:
        return _result(STEP_UP, 0.0, f"Step-up: {raw['n_faces']} faces in view, expected 1", None, v)
    c = raw.get("cloud") or {}
    vlm = c.get("vlm")
    p = vlm["live"] if vlm else None
    local = fuse(raw, cfg["fusion"], "B")["liveness"]
    # 1. strong spoof evidence -> DENY (even if image quality is poor: a dark photo is still a photo)
    if p is not None and p <= cc["vlm_spoof_max"]:
        attack = vlm["attack"] if vlm["attack"] != "none" else "spoof"
        return _result(DENY, 1 - p, f'Denied: {attack} suspected — "{vlm["evidence"]}" (VLM live {p:.2f}, local {local:.2f})', local, v)
    dev = device_over_face(raw)
    if dev:
        return _result(DENY, dev["conf"] / 100, f"Denied: screen replay suspected — face is inside a detected {dev['name']} (local {local:.2f})", local, v)
    # 2. image quality (ours + Rekognition's second opinion) -> step-up
    fails = quality_failures(raw["quality"], cfg["quality"])
    faces = c.get("faces") or []
    if len(faces) == 1:
        fails += _rk_quality_tips(faces[0], cc)
    if fails:
        return _result(STEP_UP, 0.0, "Step-up: low quality — " + "; ".join(fails), local, v)
    # 3. local liveness
    if local < t["t_lo"]:
        return _result(DENY, 1 - local, f"Denied: presentation attack suspected (local liveness {local:.2f} < {t['t_lo']:.2f})", local, v)
    # 4. identity
    sim = raw.get("similarity")
    who = raw.get("user") or "enrolled user"
    if sim is None:
        return _result(STEP_UP, 0.0, f"Step-up: identity could not be checked ({raw.get('match_error') or 'no result'})", local, v)
    if sim < t["t_match_lo"]:
        return _result(DENY, 1 - sim / 100, f"Denied: face does not match {who} (similarity {sim:.1f} < {t['t_match_lo']:.0f})", local, v)
    # 5. fail closed: a missing cloud signal can never lead to ALLOW
    missing = [f"{k} {_missing(c, k)}" for k in ("vlm", "labels") if _missing(c, k)]
    if missing:
        return _result(STEP_UP, 0.0, "Step-up: cloud check unavailable (" + ", ".join(missing) + ")", local, v)
    trust = min(local, p, sim / 100)  # type: ignore[type-var]
    if local >= t["t_hi"] and p >= cc["vlm_live_min"] and sim >= t["t_match"]:  # type: ignore[operator]
        return _result(ALLOW, trust, f'Allowed: live face (local {local:.2f}, VLM {p:.2f}: "{vlm["evidence"]}") matched {who} (similarity {sim:.1f})', local, v)
    parts = []
    if local < t["t_hi"]:
        parts.append(f"local liveness {local:.2f} below {t['t_hi']:.2f}")
    if p < cc["vlm_live_min"]:  # type: ignore[operator]
        parts.append(f'VLM live {p:.2f} below {cc["vlm_live_min"]:.2f} ("{vlm["evidence"]}")')
    if sim < t["t_match"]:
        parts.append(f"similarity {sim:.1f} below {t['t_match']:.0f}")
    return _result(STEP_UP, trust, "Step-up: borderline — " + "; ".join(parts), local, v)
