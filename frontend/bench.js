// Test bench: camera, live analysis, capture/verify, re-decision with sliders, metrics.
import { decide, ALLOW, DENY, STEP_UP } from "./decision.js";
import { afterPaint, captureFrames, grabJpeg, listCameras, nextFrame, openCamera, rememberedCamera } from "./camera.js";

const $ = (id) => document.getElementById(id);
const video = $("video"), overlay = $("overlay"), grab = $("grab");
const ATTACKS = ["photo", "screen", "video", "mask"];
const STAGES = [ // [timing key, label, colour]
  ["decode", "decode", "#718096"], ["detect", "detect", "#2b6cb0"], ["quality", "quality", "#319795"],
  ["liveness", "liveness", "#805ad5"], ["cues", "cues", "#d53f8c"], ["match_wait", "Rekognition (wait)", "#dd6b20"],
  ["decision", "decision", "#38a169"], ["network", "network+render", "#a0aec0"],
];

const S = {
  cfg: null, serverThresholds: null, attempts: [], busy: false, auto: false,
  live: false, ws: null, wsResolve: null, ring: [], spark: [], liveLat: [], lastLive: null,
};

// ---------------- helpers ----------------
const fmt = (v, d = 2) => (v == null || Number.isNaN(v) ? "—" : Number(v).toFixed(d));
const status = (t) => { $("status").textContent = t; };
const pct = (v) => (v == null ? "—" : (100 * v).toFixed(1) + "%");
async function api(path, opts = {}) {
  const r = await fetch(path, { credentials: "same-origin", ...opts });
  if (r.status === 401) throw new Error("401: open /bench?token=<BENCH_TOKEN> once to authenticate");
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.headers.get("content-type")?.includes("json") ? r.json() : r.text();
}
function quantile(xs, q) {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b), i = (s.length - 1) * q, lo = Math.floor(i);
  return s[lo] + (s[Math.ceil(i)] - s[lo]) * (i - lo);
}
function wilson(k, n, z = 1.96) { // 95% Wilson score interval
  if (!n) return [null, null];
  const p = k / n, d = 1 + z * z / n, c = p + z * z / (2 * n), m = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n));
  return [(c - m) / d, (c + m) / d];
}
function currentCfg() {
  const th = {};
  for (const k of ["t_lo", "t_hi", "t_match", "t_match_lo"]) th[k] = Number($("s-" + k).value);
  return { ...S.cfg, thresholds: th };
}
const evalVariant = (a) => $("eval-variant").value || a.variant;
// Only verify-type attempts carry the raw scores decide() needs; enroll/step-up rows keep their stored outcome.
const isVerify = (a) => !["enroll", "stepup"].includes(a.source);
const redecide = (a, cfg = currentCfg()) => isVerify(a) ? decide({ ...a.raw }, cfg, evalVariant(a))
  : { decision: a.decision, liveness: a.raw.liveness ?? a.raw.liveness_avg ?? null, reason: a.reason };
const latencyOf = (a) => a.timings.client_e2e ?? a.timings.total;

// ---------------- camera ----------------
async function startCamera(deviceId) {
  $("cam-error").classList.add("hidden");
  try {
    await openCamera(video, deviceId);
    const cams = await listCameras(); // labels only appear after permission is granted
    const cur = video.srcObject.getVideoTracks()[0].getSettings().deviceId;
    $("camera").innerHTML = cams.map((c, i) => `<option value="${c.deviceId}" ${c.deviceId === cur ? "selected" : ""}>${c.label || "camera " + (i + 1)}</option>`).join("");
  } catch (e) {
    $("cam-error").textContent = e.message;
    $("cam-error").classList.remove("hidden");
  }
}
function fpsMeter() {
  let n = 0, t0 = performance.now();
  const tick = () => {
    n++;
    const dt = performance.now() - t0;
    if (dt > 1000) {
      $("camstats").textContent = `${video.videoWidth}×${video.videoHeight} @ ${(n * 1000 / dt).toFixed(1)} fps`;
      n = 0; t0 = performance.now();
    }
    video.requestVideoFrameCallback ? video.requestVideoFrameCallback(tick) : setTimeout(tick, 33);
  };
  tick();
}

// ---------------- overlay & chips ----------------
function videoRect() { // where the video is drawn inside the stage (object-fit: contain)
  const W = overlay.clientWidth, H = overlay.clientHeight, ar = video.videoWidth / video.videoHeight || 16 / 9;
  const w = Math.min(W, H * ar), h = w / ar;
  return { x: (W - w) / 2, y: (H - h) / 2, w, h };
}
function drawOverlay(box, colour) {
  const dpr = devicePixelRatio || 1;
  overlay.width = overlay.clientWidth * dpr; overlay.height = overlay.clientHeight * dpr;
  const g = overlay.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, overlay.width, overlay.height);
  if (!box) return;
  const r = videoRect(), [x, y, w, h] = box;
  g.strokeStyle = colour; g.lineWidth = 3;
  g.strokeRect(r.x + (1 - x - w) * r.w, r.y + y * r.h, w * r.w, h * r.h); // mirror x to match preview
}
function chipLevel(name, v, qc) {
  switch (name) {
    case "blur": return v < qc.min_blur ? "red" : v < qc.min_blur * 1.25 ? "amber" : "green";
    case "brightness": return v < qc.min_brightness || v > qc.max_brightness ? "red"
      : v < qc.min_brightness * 1.25 || v > qc.max_brightness * 0.95 ? "amber" : "green";
    case "face_frac": return v < qc.min_face_frac ? "red" : v < qc.min_face_frac * 1.25 ? "amber" : "green";
    case "yaw": return Math.abs(v) > qc.max_yaw ? "red" : Math.abs(v) > qc.max_yaw * 0.75 ? "amber" : "green";
    case "roll_deg": return Math.abs(v) > qc.max_roll_deg ? "red" : Math.abs(v) > qc.max_roll_deg * 0.75 ? "amber" : "green";
  }
}
function renderChips(q) {
  if (!q) { $("chips").innerHTML = `<span class="chip red">no face</span>`; return; }
  const qc = S.cfg.quality;
  const items = [["blur", "sharpness", q.blur, 0], ["brightness", "brightness", q.brightness, 0],
    ["face_frac", "face size", q.face_frac, 2], ["yaw", "yaw", q.yaw, 2], ["roll_deg", "roll°", q.roll_deg, 0]];
  $("chips").innerHTML = items.map(([k, l, v, d]) => `<span class="chip ${chipLevel(k, v, qc)}">${l} ${fmt(v, d)}</span>`).join("");
}
function drawSparkline() {
  const c = $("sparkline"), dpr = devicePixelRatio || 1;
  c.width = c.clientWidth * dpr; c.height = 40 * dpr;
  const g = c.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0);
  const W = c.clientWidth, H = 40, now = performance.now(), th = currentCfg().thresholds;
  g.clearRect(0, 0, W, H);
  for (const [t, col] of [[th.t_lo, "#c53030"], [th.t_hi, "#12805c"]]) {
    g.strokeStyle = col; g.setLineDash([3, 3]); g.beginPath(); g.moveTo(0, H - t * H); g.lineTo(W, H - t * H); g.stroke();
  }
  g.setLineDash([]); g.strokeStyle = "#2b6cb0"; g.lineWidth = 2; g.beginPath();
  S.spark = S.spark.filter((p) => now - p.t < 5000);
  S.spark.forEach((p, i) => { const x = W - (now - p.t) / 5000 * W, y = H - p.v * H; i ? g.lineTo(x, y) : g.moveTo(x, y); });
  g.stroke();
}

// ---------------- live mode ----------------
function wsUrl() {
  const p = new URLSearchParams({ variant: $("variant").value });
  if ($("user").value) p.set("user", $("user").value);
  return `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/live?${p}`;
}
function connectLive() {
  return new Promise((res, rej) => {
    const ws = new WebSocket(wsUrl());
    ws.binaryType = "arraybuffer";
    ws.onopen = () => res(ws);
    ws.onerror = () => rej(new Error("WebSocket failed (auth? server down?)"));
    ws.onclose = (e) => { if (S.live) { status(`live socket closed (${e.code})`); stopLive(); } };
    ws.onmessage = (m) => { const r = S.wsResolve; S.wsResolve = null; r?.(JSON.parse(m.data)); };
  });
}
async function liveLoop() {
  while (S.live) {
    const t0 = performance.now();
    await nextFrame(video);
    const small = await grabJpeg(video, grab, 480, 0.7);
    const full = await grabJpeg(video, grab, 640, 0.9); // ring buffer for instant "Capture decision"
    S.ring.push({ t: performance.now(), blob: full }); if (S.ring.length > 5) S.ring.shift();
    const reply = new Promise((r) => { S.wsResolve = r; });
    S.ws.send(await small.arrayBuffer());
    const msg = await reply; // stop-and-wait: never queue frames, always send the latest
    if (!S.live) break;
    S.lastLive = msg;
    const d = msg.rolling.decision;
    const col = d === ALLOW ? "#12805c" : d === DENY ? "#c53030" : "#b7791f";
    drawOverlay(msg.box, col);
    renderChips(msg.quality);
    const chip = $("live-chip");
    chip.className = `chip ${d === ALLOW ? "green" : d === DENY ? "red" : "amber"}`;
    chip.textContent = `${d === STEP_UP ? "STEP-UP" : d} · live ${fmt(msg.liveness)} · sim ${fmt(msg.similarity, 0)}`;
    if (msg.liveness != null) S.spark.push({ t: performance.now(), v: msg.liveness });
    drawSparkline();
    await afterPaint();
    S.liveLat.push(performance.now() - t0); if (S.liveLat.length > 30) S.liveLat.shift();
    $("live-stats").textContent = `loop ${fmt(quantile(S.liveLat, 0.5), 0)} ms p50 / ${fmt(quantile(S.liveLat, 0.95), 0)} p95 · server ${fmt(msg.server_ms, 0)} ms · ${(1000 / quantile(S.liveLat, 0.5)).toFixed(1)} fps`;
    const wait = 100 - (performance.now() - t0); // cap at ~10 fps
    if (wait > 0) await new Promise((r) => setTimeout(r, wait));
  }
}
async function startLive() {
  try {
    S.ws = await connectLive();
    S.live = true; $("live-chip").classList.remove("hidden");
    liveLoop().catch((e) => { status("live error: " + e.message); stopLive(); });
  } catch (e) { status(e.message); $("live-toggle").checked = false; }
}
function stopLive() {
  S.live = false; S.ring = [];
  S.ws?.close(); S.ws = null; S.wsResolve?.({ rolling: {} });
  $("live-chip").classList.add("hidden"); $("live-toggle").checked = false;
  drawOverlay(null);
}

// ---------------- enroll / verify ----------------
async function enroll() {
  const name = $("enroll-name").value.trim();
  if (!name) return status("enter a name first");
  const fd = new FormData(); fd.append("name", name);
  for (let i = 0; i < 3; i++) {
    status(`enroll capture ${i + 1}/3 — look at the camera`);
    await new Promise((r) => setTimeout(r, 700));
    fd.append("files", await grabJpeg(video, grab, 960, 0.92), `e${i}.jpg`);
  }
  try {
    const r = await api("/api/enroll", { method: "POST", body: fd });
    status(`enrolled ${r.user} (${r.captures} captures)`);
    await loadUsers(r.user);
  } catch (e) { status(e.message); }
}

async function verify() {
  if (S.busy) return;
  S.busy = true; $("verify-btn").disabled = true;
  try {
    let frames;
    const recent = S.ring.filter((f) => performance.now() - f.t < 700);
    if (S.live && recent.length >= 3) frames = recent.map((f) => f.blob);         // freeze last frames
    else { status("capturing…"); frames = await captureFrames(video, grab, 5, 500); }
    const tEnd = performance.now(); // end of capture
    const fd = new FormData();
    frames.forEach((b, i) => fd.append("files", b, `f${i}.jpg`));
    fd.append("variant", $("variant").value); fd.append("label", $("label").value);
    if ($("user").value) fd.append("user", $("user").value);
    fd.append("save_frames", $("save-frames").checked ? "true" : "false");
    status("deciding…");
    const r = await api("/api/verify", { method: "POST", body: fd });
    renderResult(r, null);
    await afterPaint();
    const e2e = performance.now() - tEnd; // capture end -> decision on screen
    renderResult(r, e2e);
    api(`/api/attempts/${r.id}`, { method: "PATCH", headers: { "content-type": "application/json" },
      body: JSON.stringify({ client_e2e_ms: e2e }) }).catch(() => {});
    r.timings.client_e2e = Math.round(e2e * 10) / 10;
    S.attempts.push({ id: r.id, ts: Date.now() / 1000, label: $("label").value, variant: r.variant, user: r.raw.user,
      decision: r.decision, reason: r.reason, raw: r.raw, timings: r.timings, per_frame: r.per_frame });
    renderAll();
    status(`done in ${e2e.toFixed(0)} ms (${frames.length} frames${S.live ? ", from live buffer" : ""})`);
    if (r.decision === STEP_UP && !S.auto) {
      $("challenge").innerHTML = `<button id="stepup-btn" class="primary">Run step-up challenge</button>`;
      $("stepup-btn").onclick = () => runStepUp(r).catch((e) => status("step-up: " + e.message));
    }
  } catch (e) { status(e.message); } finally { S.busy = false; $("verify-btn").disabled = false; }
}

// ---------------- step-up (same backend engine as the kiosk) ----------------
async function runStepUp(prev) {
  const box = $("challenge");
  const st = await api("/api/stepup/start", { method: "POST", headers: { "content-type": "application/json" },
    body: JSON.stringify({ attempt_id: prev.id, mode: "claim", person_id: $("user").value || null, label: $("label").value }) });
  box.className = "challenge";
  for (;;) {
    const blob = await grabJpeg(video, grab, 640, 0.8);
    const s = await api(`/api/stepup/frame?sid=${st.sid}`, { method: "POST", headers: { "content-type": "image/jpeg" }, body: blob });
    const ch = s.challenge || {};
    box.innerHTML = `<b>${ch.prompt || s.prompt || "…"}</b><div class="mono small">${ch.time_left_s ?? ""} s left · done: ${(ch.done || []).join(", ") || "—"}
      ${s.signals ? ` · blinkL ${s.signals.blink_l} blinkR ${s.signals.blink_r} smile ${s.signals.smile}` : ""} · yaw ${fmt(s.quality?.yaw)}</div>`;
    if (s.final) {
      const f = s.final;
      box.innerHTML = `<span class="dec-${f.decision}">${f.decision}</span> — ${f.reason}`;
      $("badge").className = "badge " + f.decision; $("badge").textContent = f.decision + " (step-up)";
      $("reason").textContent = f.reason;
      loadAttempts();
      return;
    }
    await new Promise((r) => setTimeout(r, 90));
  }
}

function renderResult(r, e2e) {
  $("badge").className = "badge " + r.decision;
  $("badge").textContent = r.decision.replace("_", " ");
  $("reason").textContent = r.reason;
  const raw = r.raw;
  $("scores").innerHTML = [
    ["confidence", fmt(r.confidence)], ["fused liveness", fmt(r.liveness)], ["MiniFASNet", fmt(raw.minifas)],
    ["moire", fmt(raw.moire, 3)], ["motion", fmt(raw.motion, 4)], ["similarity", fmt(raw.similarity, 1)],
    ["match error", raw.match_error || "—"], ["frames w/ face", `${raw.frames_with_face}/${raw.n_frames}`],
    ["variant", r.variant], ["decision time", e2e == null ? "…" : `${e2e.toFixed(0)} ms (capture end → shown)`],
  ].map(([k, v]) => `<span class="muted">${k}</span><span>${v}</span>`).join("");
  renderLatency(r.timings, e2e);
  renderChips(raw.quality);
  $("debug").innerHTML = Object.entries(r.debug || {}).map(([k, b64]) =>
    `<figure><img src="data:image/png;base64,${b64}"><figcaption>${k === "fft" ? "FFT log-magnitude" : k.replace("minifas_", "MiniFASNet crop ×")}</figcaption></figure>`).join("");
  $("perframe").innerHTML = "per-frame liveness: " + r.per_frame.map((p, i) =>
    `#${i} ${p.n_faces ? fmt(p.liveness) + " / moire " + fmt(p.moire, 3) : "no face"}`).join(" · ");
}

function renderLatency(t, e2e) {
  const seg = { ...t, match_wait: Math.max(0, (t.match || 0) - (t.liveness || 0) - (t.cues || 0)),
    network: e2e == null ? 0 : Math.max(0, e2e - t.total) };
  const total = e2e ?? t.total, budget = S.cfg.budget_ms, scale = Math.max(budget * 1.1, total);
  $("latency").innerHTML = `
    <div class="latbar ${total > budget ? "over" : ""}">
      ${STAGES.map(([k, l, c]) => `<span title="${l} ${fmt(seg[k], 0)} ms" style="width:${(100 * (seg[k] || 0) / scale).toFixed(2)}%;background:${c}">${seg[k] > scale * 0.06 ? l : ""}</span>`).join("")}
      <div class="budget" style="left:${(100 * budget / scale).toFixed(2)}%" title="${budget} ms budget"></div>
    </div>
    <div class="legend">${STAGES.map(([k, l, c]) => `<span><i style="background:${c}"></i>${l} ${fmt(seg[k], 0)}</span>`).join("")}
      <span>· server total ${fmt(t.total, 0)} ms · Rekognition ${fmt(t.match, 0)} ms (parallel)</span></div>`;
}

// ---------------- thresholds ----------------
function buildSliders() {
  const th = S.cfg.thresholds;
  const spec = [["t_lo", 0, 1, 0.01], ["t_hi", 0, 1, 0.01], ["t_match", 0, 100, 0.5], ["t_match_lo", 0, 100, 0.5]];
  $("sliders").innerHTML = spec.map(([k, lo, hi, st]) =>
    `<div class="slider"><label for="s-${k}">${k.toUpperCase()}</label><input type="range" id="s-${k}" min="${lo}" max="${hi}" step="${st}" value="${th[k]}"><span id="v-${k}" class="mono">${th[k]}</span></div>`).join("");
  for (const [k] of spec) $("s-" + k).oninput = () => { $("v-" + k).textContent = $("s-" + k).value; renderAll(); };
}
function setSliders(th) { for (const k in th) { $("s-" + k).value = th[k]; $("v-" + k).textContent = th[k]; } renderAll(); }

// ---------------- attempts & metrics ----------------
function expected(label) { return label === "genuine" ? ALLOW : label === "unlabeled" ? null : DENY; }
function renderAll() { renderTable(); renderMetrics(); }

function renderTable() {
  const f = $("filter").value, cfg = currentCfg();
  const rows = S.attempts.filter((a) => !f || a.label === f).slice().reverse();
  $("attempts").querySelector("tbody").innerHTML = rows.map((a) => {
    const d = redecide(a, cfg), exp = expected(a.label);
    const wrong = exp && ((exp === ALLOW && d.decision !== ALLOW) || (exp === DENY && d.decision === ALLOW));
    const cls = wrong ? (d.decision === STEP_UP ? "stepup" : "wrong") : "";
    return `<tr class="${cls}" title="${a.reason.replace(/"/g, "&quot;")}">
      <td>${a.has_thumb === false ? `<span class="muted small">${a.source}</span>` : `<img loading="lazy" src="/api/attempts/${a.id}/thumb.jpg" alt="">`}</td>
      <td>${new Date(a.ts * 1000).toLocaleTimeString()}</td><td>${a.label}</td><td>${evalVariant(a)}</td>
      <td class="num">${fmt(a.raw.minifas)}</td><td class="num">${fmt(a.raw.moire, 3)}</td><td class="num">${fmt(a.raw.motion, 4)}</td>
      <td class="num">${fmt(a.raw.similarity, 1)}</td><td class="num">${fmt(d.liveness)}</td>
      <td class="dec-${d.decision}">${d.decision === STEP_UP ? "STEP-UP" : d.decision}</td><td>${exp ? (exp === ALLOW ? "ALLOW" : "not ALLOW") : "—"}</td>
      <td class="num">${fmt(latencyOf(a), 0)}</td><td><button data-del="${a.id}" title="delete">✕</button></td></tr>`;
  }).join("");
}

function rate(k, n) {
  const [lo, hi] = wilson(k, n);
  return n ? `${pct(k / n)} <span class="muted small">[${pct(lo)}–${pct(hi)}] ${k}/${n}</span>` : `<span class="muted">n=0</span>`;
}
function renderMetrics() {
  const cfg = currentCfg();
  const labelled = S.attempts.filter((a) => a.label !== "unlabeled" && isVerify(a));
  const dec = labelled.map((a) => ({ a, d: redecide(a, cfg).decision }));
  const by = (l) => dec.filter((x) => x.a.label === l);
  const gen = by("genuine"), imp = by("impostor"), atk = dec.filter((x) => ATTACKS.includes(x.a.label));
  const cnt = (xs, p) => xs.filter(p).length;
  const lat = S.attempts.filter(isVerify).map(latencyOf).filter((v) => v != null);
  let html = `<table><tr><th>metric</th><th>value [95% Wilson] k/n</th></tr>
    <tr><td>FRR (genuine not ALLOWed)</td><td>${rate(cnt(gen, (x) => x.d !== ALLOW), gen.length)}</td></tr>
    <tr><td>FRR hard (genuine DENYed)</td><td>${rate(cnt(gen, (x) => x.d === DENY), gen.length)}</td></tr>
    <tr><td>FAR all attacks + impostors</td><td>${rate(cnt([...atk, ...imp], (x) => x.d === ALLOW), atk.length + imp.length)}</td></tr>
    <tr><td>Impostor FAR (identity)</td><td>${rate(cnt(imp, (x) => x.d === ALLOW), imp.length)}</td></tr>`;
  for (const l of ATTACKS) {
    const xs = by(l);
    html += `<tr><td>${l}: detection rate (not ALLOWed)</td><td>${rate(cnt(xs, (x) => x.d !== ALLOW), xs.length)}</td></tr>`;
  }
  html += `<tr><td>decision time p50 / p95</td><td>${fmt(quantile(lat, 0.5), 0)} / ${fmt(quantile(lat, 0.95), 0)} ms <span class="muted small">n=${lat.length}</span></td></tr></table>`;
  $("metric-table").innerHTML = html;
  $("metrics-note").textContent = `${labelled.length} labelled attempts · thresholds as on sliders · variant ${$("eval-variant").value || "as captured"} · bench data is NOT the held-out test set`;

  const labels = ["genuine", "impostor", ...ATTACKS].filter((l) => by(l).length);
  $("confusion").innerHTML = `<h3>Confusion (rows = label)</h3><table><tr><th></th><th>ALLOW</th><th>STEP-UP</th><th>DENY</th><th>n</th></tr>${
    labels.map((l) => { const xs = by(l); return `<tr><td>${l}</td>${[ALLOW, STEP_UP, DENY].map((d) => `<td class="num">${cnt(xs, (x) => x.d === d)}</td>`).join("")}<td class="num">${xs.length}</td></tr>`; }).join("")}</table>`;

  const live = (xs) => xs.map((x) => redecide(x.a, cfg).liveness).filter((v) => v != null);
  drawHist($("hist"), live(gen), live(atk), cfg.thresholds);
  drawSweep($("sweep"), gen.map((x) => x.a), atk.map((x) => x.a), cfg);
}

function canvasCtx(c) {
  const dpr = devicePixelRatio || 1, W = c.clientWidth || 360, H = 160;
  c.width = W * dpr; c.height = H * dpr;
  const g = c.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
  g.font = "10px system-ui"; g.fillStyle = getComputedStyle(document.body).color;
  return [g, W, H];
}
function drawHist(c, gen, atk, th) {
  const [g, W, H] = canvasCtx(c), B = 20, pad = 16;
  const bin = (xs) => { const h = Array(B).fill(0); xs.forEach((v) => h[Math.min(B - 1, Math.floor(v * B))]++); return h; };
  const hg = bin(gen), ha = bin(atk), mx = Math.max(1, ...hg, ...ha), bw = (W - pad) / B;
  [[hg, "rgba(18,128,92,.6)", 0], [ha, "rgba(197,48,48,.6)", bw / 2]].forEach(([h, col, off]) => {
    g.fillStyle = col; h.forEach((v, i) => g.fillRect(pad + i * bw + off, H - pad - v / mx * (H - 2 * pad), bw / 2 - 1, v / mx * (H - 2 * pad)));
  });
  for (const [t, col] of [[th.t_lo, "#c53030"], [th.t_hi, "#12805c"]]) { g.fillStyle = col; g.fillRect(pad + t * (W - pad), 4, 1.5, H - pad - 4); }
  g.fillStyle = getComputedStyle(document.body).color;
  g.fillText(`0`, pad, H - 4); g.fillText(`1`, W - 8, H - 4);
  g.fillText(`genuine n=${gen.length} (green) · attacks n=${atk.length} (red)`, pad + 4, 12);
}
function drawSweep(c, gen, atk, cfg) {
  const [g, W, H] = canvasCtx(c), pad = 16, pts = [];
  for (let t = 0; t <= 1.0001; t += 0.02) {
    const cc = { ...cfg, thresholds: { ...cfg.thresholds, t_hi: t, t_lo: Math.min(cfg.thresholds.t_lo, t) } };
    const frr = gen.length ? gen.filter((a) => redecide(a, cc).decision !== ALLOW).length / gen.length : null;
    const acc = atk.length ? atk.filter((a) => redecide(a, cc).decision === ALLOW).length / atk.length : null;
    pts.push([t, frr, acc]);
  }
  const X = (t) => pad + t * (W - 2 * pad), Y = (v) => H - pad - v * (H - 2 * pad);
  [[1, "#c53030", "FRR"], [2, "#2b6cb0", "attack accept (APCER)"]].forEach(([i, col, name], k) => {
    g.strokeStyle = col; g.lineWidth = 2; g.beginPath();
    pts.filter((p) => p[i] != null).forEach((p, j) => (j ? g.lineTo(X(p[0]), Y(p[i])) : g.moveTo(X(p[0]), Y(p[i]))));
    g.stroke(); g.fillStyle = col; g.fillText(name, pad + 4 + k * 50, 12);
  });
  g.fillStyle = "#12805c"; g.fillRect(X(cfg.thresholds.t_hi), 16, 1.5, H - 2 * pad);
  g.fillStyle = getComputedStyle(document.body).color; g.fillText("T_hi → 0", pad, H - 4); g.fillText("1", W - pad, H - 4);
}

// ---------------- misc ----------------
async function loadUsers(select) {
  const users = await api("/api/users");
  $("user").innerHTML = users.length ? users.map((u) => `<option ${u === select ? "selected" : ""}>${u}</option>`).join("") : `<option value="">(none enrolled)</option>`;
  if (!select && users.length) $("user").value = users[users.length - 1];
}
async function loadAttempts() { S.attempts = await api("/api/attempts"); renderAll(); }

async function autoRun() {
  if (S.auto) { S.auto = false; return; }
  S.auto = true; $("auto-btn").textContent = "Stop";
  const n = Number($("auto-n").value) || 1;
  for (let i = 0; i < n && S.auto; i++) {
    for (let c = 3; c > 0 && S.auto; c--) { status(`attempt ${i + 1}/${n} in ${c}… (${$("label").value})`); await new Promise((r) => setTimeout(r, 1000)); }
    if (S.auto) await verify();
  }
  S.auto = false; $("auto-btn").textContent = "Start";
}

async function recordClip() {
  if (!video.srcObject) return;
  const rec = new MediaRecorder(video.srcObject, { mimeType: "video/webm" }), parts = [];
  rec.ondataavailable = (e) => parts.push(e.data);
  const done = new Promise((r) => { rec.onstop = r; });
  rec.start(); status("recording 3 s clip…");
  await new Promise((r) => setTimeout(r, 3000)); rec.stop(); await done;
  const fd = new FormData(); fd.append("label", $("label").value); fd.append("file", new Blob(parts, { type: "video/webm" }), "clip.webm");
  if (!$("save-frames").checked) return status("clip recorded but not saved (tick 'save frames' to store it)");
  try { status("saved " + (await api("/api/clip", { method: "POST", body: fd })).saved); } catch (e) { status(e.message); }
}

async function init() {
  try {
    S.cfg = await api("/api/config");
  } catch (e) { $("cam-error").textContent = e.message; $("cam-error").classList.remove("hidden"); return; }
  S.serverThresholds = { ...S.cfg.thresholds };
  $("conn").textContent = `server ok · budget ${S.cfg.budget_ms} ms`;
  $("label").innerHTML = S.cfg.labels.map((l) => `<option>${l}</option>`).join("");
  $("filter").innerHTML += S.cfg.labels.map((l) => `<option>${l}</option>`).join("");
  $("variant").value = S.cfg.fusion.default_variant;
  buildSliders();
  await loadUsers();
  await loadAttempts();
  await startCamera(rememberedCamera());
  fpsMeter();

  $("camera").onchange = () => { stopLive(); startCamera($("camera").value); };
  $("live-toggle").onchange = () => ($("live-toggle").checked ? startLive() : stopLive());
  $("variant").onchange = () => S.ws?.send(JSON.stringify({ variant: $("variant").value }));
  $("user").onchange = () => { if (S.live) { stopLive(); $("live-toggle").checked = true; startLive(); } };
  $("enroll-btn").onclick = enroll;
  $("verify-btn").onclick = verify;
  $("auto-btn").onclick = autoRun;
  $("clip-btn").onclick = recordClip;
  $("filter").onchange = renderTable;
  $("eval-variant").onchange = renderAll;
  $("reload").onclick = loadAttempts;
  $("reset-thr").onclick = () => setSliders(S.serverThresholds);
  $("save-config").onclick = async () => {
    const r = await api("/api/config", { method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ thresholds: currentCfg().thresholds, save: true }) });
    S.serverThresholds = { ...r.thresholds }; S.cfg.thresholds = { ...r.thresholds };
    status("saved thresholds to config.yaml");
  };
  $("attempts").onclick = async (e) => {
    const id = e.target.dataset?.del;
    if (!id) return;
    await api(`/api/attempts/${id}`, { method: "DELETE" });
    S.attempts = S.attempts.filter((a) => a.id !== id); renderAll();
  };
  document.addEventListener("keydown", (e) => {
    if (e.code === "Space" && !["INPUT", "SELECT", "BUTTON", "TEXTAREA"].includes(document.activeElement.tagName)) { e.preventDefault(); verify(); }
  });
  addEventListener("resize", () => drawOverlay(S.lastLive?.box, "#888"));
}
init();
