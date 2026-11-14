// Mirror of backend/decision.py — keep in sync (tests/test_decision_parity.py checks it).
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

export function decide(raw, cfg, variant) {
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
