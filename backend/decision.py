"""Decision layer: pure function of RAW scores + config. Mirrored 1:1 in frontend/decision.js
(the bench re-decides stored attempts when sliders move); tests/test_decision_parity.py keeps them equal."""
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


def decide(raw: dict[str, Any], cfg: dict[str, Any], variant: str) -> dict[str, Any]:
    """raw keys: n_faces (max faces in any frame), quality (median metrics) | None, minifas, moire,
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
