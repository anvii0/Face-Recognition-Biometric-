// Mirror of backend/decision.py — keep in sync (tests/test_decision_parity.py checks it).
// Variants: A MiniFASNet | B +moire | C +motion | D cloud-only (hid-gate rules) | E fused local+cloud.
export const ALLOW = "ALLOW", STEP_UP = "REQUEST_VERIFICATION", DENY = "DENY";
const clip = (v) => Math.max(0, Math.min(1, v));
const f0 = (v) => v.toFixed(0), f1 = (v) => v.toFixed(1), f2 = (v) => v.toFixed(2), f3 = (v) => v.toFixed(3);
const r4 = (v) => Math.round(v * 1e4) / 1e4;

export function fuse(raw, fusion, variant) {
  const live = Number(raw.minifas);
  let moirePen = 0, motionPen = 0;
  if ((variant === "B" || variant === "C") && raw.moire != null) {
    const span = (fusion.moire_hi - fusion.moire_lo) || 1e-9;
    moirePen = fusion.w_moire * clip((raw.moire - fusion.moire_lo) / span);
  }
  if (variant === "C" && raw.motion != null) {
    motionPen = fusion.w_motion * (1 - clip(raw.motion / fusion.motion_ref));
  }
  return { liveness: clip(live - moirePen - motionPen), moire_pen: moirePen, motion_pen: motionPen };
}

export function qualityFailures(q, qc) {
  const f = [];
  if (q.brightness < qc.min_brightness) f.push(`too dark (brightness ${f0(q.brightness)} < ${f0(qc.min_brightness)})`);
  if (q.brightness > qc.max_brightness) f.push(`overexposed (brightness ${f0(q.brightness)} > ${f0(qc.max_brightness)})`);
  if (q.blur < qc.min_blur) f.push(`blurry (sharpness ${f0(q.blur)} < ${f0(qc.min_blur)})`);
  if (q.face_frac < qc.min_face_frac) f.push(`face too small (${f2(q.face_frac)} < ${f2(qc.min_face_frac)} of frame)`);
  if (Math.abs(q.yaw) > qc.max_yaw) f.push(`head turned (yaw ${f2(q.yaw)}, max ${f2(qc.max_yaw)})`);
  if (Math.abs(q.roll_deg) > qc.max_roll_deg) f.push(`head tilted (${f0(q.roll_deg)} deg, max ${f0(qc.max_roll_deg)})`);
  return f;
}

export function decideLocal(raw, cfg, variant) {
  const t = cfg.thresholds;
  const out = (decision, confidence, reason, live = null) =>
    ({ decision, confidence: r4(confidence), reason, liveness: live == null ? null : r4(live), variant });

  if (!raw.n_faces || raw.quality == null) return out(STEP_UP, 0, "Step-up: no face detected");
  if (raw.n_faces > 1) return out(STEP_UP, 0, `Step-up: ${raw.n_faces} faces in view, expected 1`);
  const fails = qualityFailures(raw.quality, cfg.quality);
  const f = fuse(raw, cfg.fusion, variant);
  const live = f.liveness;
  if (fails.length) return out(STEP_UP, 0, "Step-up: low quality — " + fails.join("; "), live);
  if (live < t.t_lo) {
    let why;
    if (f.moire_pen >= Math.max(f.motion_pen, 0.1)) why = `screen replay suspected, moire score ${f2(raw.moire)}`;
    else if (f.motion_pen >= 0.1) why = `no natural face motion (motion ${f3(raw.motion)})`;
    else why = "presentation attack suspected";
    return out(DENY, 1 - live, `Denied: ${why} (liveness ${f2(live)} < ${f2(t.t_lo)})`, live);
  }
  const sim = raw.similarity;
  if (sim == null) {
    const err = raw.match_error || "no face found by matcher";
    return out(STEP_UP, 0, `Step-up: identity could not be checked (${err})`, live);
  }
  const who = raw.user || "enrolled user";
  if (sim < t.t_match_lo)
    return out(DENY, 1 - sim / 100, `Denied: face does not match ${who} (similarity ${f1(sim)} < ${f0(t.t_match_lo)})`, live);
  const trust = Math.min(live, sim / 100);
  if (live >= t.t_hi && sim >= t.t_match)
    return out(ALLOW, trust, `Allowed: live face (liveness ${f2(live)}) matched ${who} (similarity ${f1(sim)})`, live);
  const parts = [];
  if (live < t.t_hi) parts.push(`liveness ${f2(live)} below ${f2(t.t_hi)}`);
  if (sim < t.t_match) parts.push(`similarity ${f1(sim)} below ${f0(t.t_match)}`);
  return out(STEP_UP, trust, "Step-up: borderline — " + parts.join("; "), live);
}

// ======================= cloud signals (variants D and E) =======================
export const VARIANTS = ["A", "B", "C", "D", "E"];

export function decide(raw, cfg, variant) {
  if (variant === "D") return decideCloud(raw, cfg);
  if (variant === "E") return decideFused(raw, cfg);
  return decideLocal(raw, cfg, variant);
}

const result = (decision, confidence, reason, live, variant) =>
  ({ decision, confidence: r4(confidence), reason, liveness: live == null ? null : r4(live), variant });

export function faceCentreInBox(f, b) {
  const cx = f.Left + f.Width / 2, cy = f.Top + f.Height / 2;
  return b.Left <= cx && cx <= b.Left + b.Width && b.Top <= cy && cy <= b.Top + b.Height;
}

export function deviceOverFace(raw) {
  const c = raw.cloud || {};
  const faces = c.faces || [];
  const face = faces.length === 1 ? faces[0].box : raw.face_box;
  if (!face) return null;
  return (c.devices || []).find((d) => faceCentreInBox(face, d.box)) || null;
}

function missing(c, key) {
  const err = (c.errors || {})[key];
  if (err) return err;
  const present = key === "vlm" ? c.vlm : key === "labels" ? c.devices : c.faces;
  return present != null ? null : "missing";
}

function rkQualityTips(face, cc) {
  const t = [];
  if (face.brightness < cc.rk_min_brightness) t.push(`too dark (Rekognition brightness ${f0(face.brightness)})`);
  if (face.sharpness < cc.rk_min_sharpness) t.push(`blurry (Rekognition sharpness ${f0(face.sharpness)})`);
  if (face.box.Width < cc.rk_min_face_width) t.push("face too far away");
  if (face.occluded) t.push("face partly covered");
  return t;
}

export function decideCloud(raw, cfg) {
  const cc = cfg.cloud, v = "D", c = raw.cloud;
  if (!c) return result(STEP_UP, 0, "Step-up: cloud signals were not collected", null, v);
  const vlm = c.vlm, p = vlm ? vlm.live : null;
  if (missing(c, "faces")) return result(STEP_UP, 0, `Step-up: face check unavailable (${missing(c, "faces")})`, p, v);
  const faces = c.faces;
  if (!faces.length) return result(STEP_UP, 0, "Step-up: no face found by Rekognition", p, v);
  if (faces.length > 1) return result(STEP_UP, 0, `Step-up: ${faces.length} faces in view, expected 1`, p, v);
  const dev = deviceOverFace(raw);
  if (p != null && p <= cc.vlm_spoof_max) {
    const attack = vlm.attack !== "none" ? vlm.attack : "spoof";
    const extra = dev ? `; face is inside a detected ${dev.name}` : "";
    return result(DENY, 1 - p, `Denied: ${attack} suspected — "${vlm.evidence}" (VLM live ${f2(p)})${extra}`, p, v);
  }
  if (dev && (p == null || p < cc.vlm_live_min))
    return result(DENY, dev.conf / 100, `Denied: screen replay suspected — face is inside a detected ${dev.name}`, p, v);
  const tips = rkQualityTips(faces[0], cc);
  if (tips.length) return result(STEP_UP, 0, "Step-up: low quality — " + tips.join("; "), p, v);
  const sim = raw.similarity;
  if (sim == null) return result(STEP_UP, 0, `Step-up: identity could not be checked (${raw.match_error || "no result"})`, p, v);
  const who = raw.user || "enrolled user";
  if (sim < cc.d_sim_min) return result(DENY, 1 - sim / 100, `Denied: face does not match ${who} (similarity ${f1(sim)} < ${f0(cc.d_sim_min)})`, p, v);
  const doubts = [];
  if (p == null) doubts.push(`liveness model unavailable (${missing(c, "vlm")})`);
  else if (p < cc.vlm_live_min) doubts.push(`liveness uncertain (VLM ${f2(p)}: "${vlm.evidence}")`);
  if (dev) doubts.push(`a ${dev.name} overlaps the face`);
  if ((c.weak || []).length) doubts.push("possible spoof material in view: " + c.weak.join(", "));
  if (missing(c, "labels")) doubts.push(`object check unavailable (${missing(c, "labels")})`);
  if (sim < cc.d_sim_strong) doubts.push(`identity match borderline (${f1(sim)})`);
  if (doubts.length) return result(STEP_UP, Math.min(p || 0, sim / 100), "Step-up: " + doubts.join("; "), p, v);
  return result(ALLOW, Math.min(p, sim / 100), `Allowed: live person (VLM ${f2(p)}: "${vlm.evidence}") matched ${who} (similarity ${f1(sim)})`, p, v);
}

export function decideFused(raw, cfg) {
  const t = cfg.thresholds, cc = cfg.cloud, v = "E";
  if (!raw.n_faces || raw.quality == null) return result(STEP_UP, 0, "Step-up: no face detected", null, v);
  if (raw.n_faces > 1) return result(STEP_UP, 0, `Step-up: ${raw.n_faces} faces in view, expected 1`, null, v);
  const c = raw.cloud || {};
  const vlm = c.vlm, p = vlm ? vlm.live : null;
  const local = fuse(raw, cfg.fusion, "B").liveness;
  if (p != null && p <= cc.vlm_spoof_max) {
    const attack = vlm.attack !== "none" ? vlm.attack : "spoof";
    return result(DENY, 1 - p, `Denied: ${attack} suspected — "${vlm.evidence}" (VLM live ${f2(p)}, local ${f2(local)})`, local, v);
  }
  const dev = deviceOverFace(raw);
  if (dev) return result(DENY, dev.conf / 100, `Denied: screen replay suspected — face is inside a detected ${dev.name} (local ${f2(local)})`, local, v);
  const fails = qualityFailures(raw.quality, cfg.quality);
  const faces = c.faces || [];
  if (faces.length === 1) fails.push(...rkQualityTips(faces[0], cc));
  if (fails.length) return result(STEP_UP, 0, "Step-up: low quality — " + fails.join("; "), local, v);
  if (local < t.t_lo) return result(DENY, 1 - local, `Denied: presentation attack suspected (local liveness ${f2(local)} < ${f2(t.t_lo)})`, local, v);
  const sim = raw.similarity, who = raw.user || "enrolled user";
  if (sim == null) return result(STEP_UP, 0, `Step-up: identity could not be checked (${raw.match_error || "no result"})`, local, v);
  if (sim < t.t_match_lo) return result(DENY, 1 - sim / 100, `Denied: face does not match ${who} (similarity ${f1(sim)} < ${f0(t.t_match_lo)})`, local, v);
  const miss = ["vlm", "labels"].filter((k) => missing(c, k)).map((k) => `${k} ${missing(c, k)}`);
  if (miss.length) return result(STEP_UP, 0, "Step-up: cloud check unavailable (" + miss.join(", ") + ")", local, v);
  const trust = Math.min(local, p, sim / 100);
  if (local >= t.t_hi && p >= cc.vlm_live_min && sim >= t.t_match)
    return result(ALLOW, trust, `Allowed: live face (local ${f2(local)}, VLM ${f2(p)}: "${vlm.evidence}") matched ${who} (similarity ${f1(sim)})`, local, v);
  const parts = [];
  if (local < t.t_hi) parts.push(`local liveness ${f2(local)} below ${f2(t.t_hi)}`);
  if (p < cc.vlm_live_min) parts.push(`VLM live ${f2(p)} below ${f2(cc.vlm_live_min)} ("${vlm.evidence}")`);
  if (sim < t.t_match) parts.push(`similarity ${f1(sim)} below ${f0(t.t_match)}`);
  return result(STEP_UP, trust, "Step-up: borderline — " + parts.join("; "), local, v);
}
