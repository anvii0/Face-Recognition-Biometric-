// Kiosk: Verify (claim-ID 1:1 / open-set 1:N) with auto-capture + step-up, liveness-gated Enroll, People.
import { afterPaint, captureFrames, grabJpeg, listCameras, nextFrame, openCamera, rememberedCamera } from "./camera.js";

const $ = (id) => document.getElementById(id);
const video = $("video"), grab = $("grab");
const RESET_MS = 3000, READY_FRAMES = 3, BUDGET = 2000;
const GLYPH = { blink: "◉", turn_left: "←", turn_right: "→", smile: "☺" };
const LABEL = { blink: "Blink", turn_left: "Turn left", turn_right: "Turn right", smile: "Smile" };
const ICON = {
  ALLOW: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>',
  REQUEST_VERIFICATION: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round"><path d="M12 7v6M12 17.2v.1"/></svg>',
  DENY: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round"><path d="M7 7l10 10M17 7L7 17"/></svg>',
};

const S = {
  tab: "verify", mode: "claim", state: "idle", // idle | scanning | capturing | deciding | result | stepup
  ws: null, wsResolve: null, live: false, ready: 0, timer: null, sound: true, replace: false, enrollSid: null,
  enrollAbort: false, cfg: null,
};
try { S.sound = localStorage.getItem("hid.sound") !== "0"; } catch { /* storage blocked */ }

// ---------------- utilities ----------------
async function api(path, opts = {}) {
  const r = await fetch(path, { credentials: "same-origin", ...opts });
  if (r.status === 401) throw new Error("Not authorised — open this page once with ?token=…");
  const ct = r.headers.get("content-type") || "";
  const body = ct.includes("json") ? await r.json() : await r.text();
  if (!r.ok) { const e = new Error(typeof body.detail === "string" ? body.detail : body.detail?.message || r.statusText); e.status = r.status; throw e; }
  return body;
}
const json = (o) => ({ method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(o) });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function tone(kind) {
  if (!S.sound) return;
  try {
    const ctx = tone.ctx || (tone.ctx = new AudioContext());
    const seq = { ALLOW: [[660, 0], [990, 0.11]], DENY: [[220, 0], [180, 0.16]], REQUEST_VERIFICATION: [[520, 0]], tick: [[880, 0]] }[kind] || [];
    for (const [f, t] of seq) {
      const o = ctx.createOscillator(), g = ctx.createGain();
      o.type = kind === "DENY" ? "sawtooth" : "sine"; o.frequency.value = f;
      g.gain.setValueAtTime(0.0001, ctx.currentTime + t);
      g.gain.exponentialRampToValueAtTime(kind === "tick" ? 0.05 : 0.18, ctx.currentTime + t + 0.02);
      g.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + t + (kind === "tick" ? 0.08 : 0.22));
      o.connect(g).connect(ctx.destination); o.start(ctx.currentTime + t); o.stop(ctx.currentTime + t + 0.25);
    }
  } catch { /* audio not available */ }
}
function renderSound() {
  $("sound-btn").innerHTML = S.sound
    ? '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M11 5L6 9H3v6h3l5 4V5zM15.5 8.5a5 5 0 0 1 0 7M18.5 5.5a9 9 0 0 1 0 13"/></svg>'
    : '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M11 5L6 9H3v6h3l5 4V5zM16 9l5 5M21 9l-5 5"/></svg>';
}

// ---------------- camera chrome ----------------
function setRing(p) {
  const v = Math.max(0, Math.min(1, p));
  $("ring").style.strokeDashoffset = String(100 - v * 100);
  $("ring").classList.toggle("zero", v < 0.01); // hide the round line-cap dot at 0%
}
function setCoach(text, kind = "") { const c = $("coach"); c.textContent = text; c.className = "coach " + kind; }
function showPrompt(kind, sub = "") {
  const p = $("prompt");
  if (!kind) { p.classList.remove("show"); return; }
  $("prompt-glyph").textContent = GLYPH[kind] || ""; $("prompt-big").textContent = kind in LABEL ? promptText(kind) : kind;
  $("prompt-sub").textContent = sub; p.classList.add("show");
}
function promptText(kind) {
  return { blink: "Blink slowly", turn_left: "Turn your head left", turn_right: "Turn your head right", smile: "Give us a smile" }[kind];
}
function steps(listId, active, done = [], fail = null) {
  for (const li of $(listId).children) {
    const s = li.dataset.s;
    li.className = "step" + (s === fail ? " fail" : done.includes(s) ? " done" : s === active ? " active" : "");
  }
}
function statsHtml(d) {
  return [
    d.time != null ? [`${(d.time / 1000).toFixed(1)} s`, "decision time"] : null,
    d.confidence != null ? [`${Math.round(d.confidence * 100)}%`, "confidence"] : null,
  ].filter(Boolean).map(([b, s]) => `<div class="d-stat"><b class="mono">${b}</b><span>${s}</span></div>`).join("");
}
function showDecision(d) {
  const el = $("decision");
  el.className = "decision " + d.decision;
  $("d-icon").innerHTML = ICON[d.decision];
  $("d-title").textContent = { ALLOW: "Access granted", DENY: "Access denied", REQUEST_VERIFICATION: "Extra check needed" }[d.decision];
  $("d-name").textContent = d.decision === "ALLOW" ? (d.name ? `Welcome, ${d.name}` : "") : "";
  $("d-reason").textContent = d.decision === "ALLOW" ? "Live person verified" + (d.stepup ? " after step-up" : "") : d.friendly || d.reason;
  $("d-stats").innerHTML = statsHtml(d);
  requestAnimationFrame(() => el.classList.add("show"));
  tone(d.decision);
}
function hideDecision() { $("decision").classList.remove("show"); }
function setGauge(ms) {
  $("gauge-val").textContent = ms == null ? "—" : `${(ms / 1000).toFixed(1)} s`;
  const f = $("gauge-fill");
  f.style.width = ms == null ? "0" : `${Math.min(100, (ms / 3000) * 100)}%`;
  f.classList.toggle("over", ms != null && ms > BUDGET);
}

// Friendly reason text for the kiosk (full technical reason stays in logs / bench).
function friendly(r) {
  const s = r.reason || "";
  if (/screen replay|screen suspected/.test(s)) return "Screen replay suspected";
  if (/printed_photo suspected/.test(s)) return "Printed photo suspected";
  if (/mask suspected/.test(s)) return "Mask suspected";
  if (/spoof suspected/.test(s)) return "We couldn't confirm a live person";
  if (/cloud check unavailable/.test(s)) return "Taking a little longer than usual — one quick extra check";
  if (/presentation attack|natural face motion/.test(s)) return "We couldn't confirm a live person";
  if (/does not match/.test(s)) return "Face does not match this ID";
  if (/too dark/.test(s)) return "Too dark — let's do a quick extra check";
  if (/overexposed/.test(s)) return "Too bright — let's do a quick extra check";
  if (/blurry/.test(s)) return "Image was blurry — let's do a quick extra check";
  if (/too small/.test(s)) return "Please move closer";
  if (/head turned|tilted/.test(s)) return "Please look straight at the camera";
  if (/borderline/.test(s)) return "Almost there — one quick extra check";
  if (/identity could not|nobody enrolled/.test(s)) return "Identity could not be checked";
  if (/no face/.test(s)) return "No face detected";
  if (/step-up failed/.test(s)) return "Challenge not completed";
  return s.replace(/^(Denied|Step-up): /, "");
}

// ---------------- live coaching socket ----------------
function connectLive() {
  return new Promise((res, rej) => {
    const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/live?mode=none`);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => res(ws);
    ws.onerror = () => rej(new Error("live socket failed"));
    ws.onclose = () => { S.ws = null; S.live = false; $("live-tag").classList.add("hidden"); };
    ws.onmessage = (m) => { const r = S.wsResolve; S.wsResolve = null; r?.(JSON.parse(m.data)); };
  });
}
async function liveLoop() {
  if (S.live) return;
  try { S.ws = S.ws || await connectLive(); } catch { setCoach("Reconnecting…", "warn"); setTimeout(liveLoop, 1500); return; }
  S.live = true; $("live-tag").classList.remove("hidden");
  while (S.live && S.ws) {
    const t0 = performance.now();
    if (!["idle", "scanning"].includes(S.state) || S.tab !== "verify") { await sleep(120); continue; }
    await nextFrame(video);
    const blob = await grabJpeg(video, grab, 480, 0.7);
    const reply = new Promise((r) => { S.wsResolve = r; });
    S.ws.send(await blob.arrayBuffer());
    const msg = await reply;
    if (!msg?.rolling) continue;
    onLive(msg);
    const wait = 140 - (performance.now() - t0); // ~7 fps is plenty for coaching
    if (wait > 0) await sleep(wait);
  }
}
function centered(box) {
  if (!box) return false;
  const cx = box[0] + box[2] / 2, cy = box[1] + box[3] / 2;
  return Math.abs(cx - 0.5) < 0.16 && Math.abs(cy - 0.47) < 0.2;
}
function onLive(msg) {
  S.lastLive = performance.now();
  if (!["idle", "scanning"].includes(S.state)) return;
  const ok = msg.n_faces === 1 && !(msg.fails || []).length && centered(msg.box);
  let tip = msg.coach;
  if (msg.n_faces === 1 && !(msg.fails || []).length && !centered(msg.box)) tip = "Center your face in the oval";
  $("cam-card").classList.toggle("ready", ok);
  if (S.state === "idle") {
    setCoach(S.mode === "claim" ? (msg.n_faces ? tip : "Enter your badge ID to begin") : tip, ok ? "good" : "");
    setRing(0); steps("verify-steps", null);
    return;
  }
  steps("verify-steps", "position");
  S.ready = ok ? S.ready + 1 : 0;
  setRing(S.ready / READY_FRAMES * 0.5);
  setCoach(ok ? "Hold still…" : tip, ok ? "good" : "warn");
  if (S.ready >= READY_FRAMES) runVerify();
}

// ---------------- verify ----------------
function startScanning() {
  if (S.mode === "claim" && !$("claim-id").value.trim()) { $("claim-id").focus(); setCoach("Enter your badge ID to begin", "warn"); return; }
  hideDecision(); S.state = "scanning"; S.ready = 0;
  $("verify-start").disabled = true;
  // Fallback: if live coaching is unavailable (socket down / slow Wi-Fi), capture anyway after 2.5 s —
  // the verify pipeline applies the same quality gates server-side.
  const t0 = performance.now();
  const watchdog = () => {
    if (S.state !== "scanning") return;
    if (!S.lastLive || S.lastLive < t0) { if (performance.now() - t0 > 2500) return runVerify(); }
    setTimeout(watchdog, 250);
  };
  setTimeout(watchdog, 250);
}
function resetVerify() {
  clearTimeout(S.timer); hideDecision(); showPrompt(null); setRing(0);
  S.state = S.mode === "open" ? "scanning" : "idle"; S.ready = 0;
  if (S.mode === "claim") $("claim-id").value = "";
  $("verify-start").disabled = S.mode === "open";
  steps("verify-steps", null);
}
async function runVerify() {
  S.state = "capturing";
  steps("verify-steps", "liveness", ["position"]);
  setCoach("Scanning…", "good");
  const ringAnim = animateRing(0.5, 0.9, 500);
  const frames = await captureFrames(video, grab, 5, 500, 640, 0.85);
  const tEnd = performance.now(); // end of capture
  ringAnim.stop(); setRing(0.95);
  S.state = "deciding";
  steps("verify-steps", "identity", ["position", "liveness"]);
  const fd = new FormData();
  frames.forEach((b, i) => fd.append("files", b, `f${i}.jpg`));
  fd.append("mode", S.mode);
  const claimed = $("claim-id").value.trim();
  if (S.mode === "claim") fd.append("person_id", claimed);
  fd.append("source", "kiosk");
  let r;
  try { r = await api("/api/verify", { method: "POST", body: fd }); }
  catch (e) {
    r = { decision: e.status === 404 ? "DENY" : "REQUEST_VERIFICATION", reason: e.status === 404 ? "Unknown badge ID" : e.message, raw: {} };
  }
  const d = { decision: r.decision, name: r.raw?.user, reason: r.reason, friendly: r.id ? friendly(r) : r.reason, confidence: r.confidence };
  showDecision(d);
  await afterPaint();
  const e2e = performance.now() - tEnd; // capture end -> decision on screen
  setGauge(e2e); setRing(1);
  $("d-stats").innerHTML = statsHtml({ ...d, time: e2e });
  if (r.id) api(`/api/attempts/${r.id}`, { ...json({ client_e2e_ms: e2e }), method: "PATCH" }).catch(() => {});
  steps("verify-steps", null, ["position", "liveness", "identity", "decision"]);
  S.state = "result";
  if (r.decision === "REQUEST_VERIFICATION" && r.id) {
    S.timer = setTimeout(() => runStepUp(r, claimed), 1600);
  } else {
    S.timer = setTimeout(resetVerify, RESET_MS);
  }
}
function animateRing(from, to, ms) {
  let stop = false; const t0 = performance.now();
  const f = () => { if (stop) return; const p = Math.min(1, (performance.now() - t0) / ms); setRing(from + (to - from) * p); if (p < 1) requestAnimationFrame(f); };
  requestAnimationFrame(f);
  return { stop: () => { stop = true; } };
}

async function runStepUp(prev, claimed) {
  S.state = "stepup"; hideDecision();
  let st;
  try { st = await api("/api/stepup/start", json({ attempt_id: prev.id, mode: S.mode, person_id: S.mode === "claim" ? claimed : null })); }
  catch (e) { showDecision({ decision: "DENY", reason: e.message, friendly: "Extra check unavailable" }); S.timer = setTimeout(resetVerify, RESET_MS); return; }
  let last = null;
  while (S.state === "stepup") {
    const blob = await grabJpeg(video, grab, 640, 0.8);
    let s;
    try { s = await api(`/api/stepup/frame?sid=${st.sid}`, { method: "POST", headers: { "content-type": "image/jpeg" }, body: blob }); }
    catch (e) { showDecision({ decision: "DENY", reason: e.message, friendly: "Extra check failed" }); break; }
    const ch = s.challenge;
    if (ch?.current) {
      if (ch.current !== last) { tone("tick"); last = ch.current; }
      showPrompt(ch.current, `${Math.ceil(ch.time_left_s)} s`);
      setRing(ch.time_left_s / st.timeout_s);
      setCoach(ch.message || "Extra check — follow the prompt", "warn");
    }
    if (s.final) {
      showPrompt(null);
      const f = s.final;
      showDecision({ decision: f.decision, name: f.user, reason: f.reason, friendly: friendly(f), confidence: f.confidence, stepup: true });
      break;
    }
    await sleep(90);
  }
  S.state = "result";
  S.timer = setTimeout(resetVerify, RESET_MS);
}

function setMode(m) {
  S.mode = m;
  $("mode-claim").setAttribute("aria-pressed", m === "claim");
  $("mode-open").setAttribute("aria-pressed", m === "open");
  $("claim-field").style.display = m === "claim" ? "" : "none";
  $("verify-sub").textContent = m === "claim" ? "Enter or scan your badge ID, then look at the camera." : "Just look at the camera — we'll find you.";
  $("identity-sub").textContent = m === "claim" ? "1:1 against the claimed ID" : "1:N search across enrolled people";
  resetVerify();
}

// ---------------- enroll ----------------
function enrollValid() {
  return $("e-name").value.trim() && /^[A-Za-z0-9_-]{1,32}$/.test($("e-id").value.trim()) && $("e-consent").checked;
}
function enrollView(v) {
  for (const id of ["enroll-form", "enroll-run", "enroll-done", "enroll-fail"]) $(id).style.display = id === v ? "grid" : "none";
}
async function startEnroll() {
  $("e-error").classList.add("hidden");
  let st;
  try {
    st = await api("/api/enroll/start", json({ name: $("e-name").value.trim(), person_id: $("e-id").value.trim(),
      role: $("e-role").value.trim(), consent: $("e-consent").checked, replace: S.replace }));
  } catch (e) { $("e-error").textContent = e.message; $("e-error").classList.remove("hidden"); return; }
  S.enrollSid = st.sid; S.enrollAbort = false;
  enrollView("enroll-run");
  $("er-title").textContent = `Capturing ${$("e-name").value.trim()}`;
  $("er-sub").textContent = "Look straight at the camera.";
  $("er-chips").innerHTML = ""; $("er-count").textContent = "";
  steps("enroll-steps", "center");
  let last = null;
  while (!S.enrollAbort) {
    const blob = await grabJpeg(video, grab, 640, 0.85);
    let s;
    try { s = await api(`/api/enroll/frame?sid=${st.sid}`, { method: "POST", headers: { "content-type": "image/jpeg" }, body: blob }); }
    catch (e) { return enrollFailed(e.message, ""); }
    if (S.enrollAbort) return;
    if (s.phase === "center") {
      const collecting = s.progress > 0;
      steps("enroll-steps", collecting ? "liveness" : "center", collecting ? ["center"] : []);
      setRing(s.progress);
      setCoach(s.prompt, s.prompt.startsWith("Hold") ? "good" : "warn");
      $("er-sub").textContent = collecting ? "Checking for a live person…" : "Look straight at the camera.";
      $("cam-card").classList.toggle("ready", s.prompt.startsWith("Hold"));
    } else if (s.phase === "challenge") {
      const ch = s.challenge;
      steps("enroll-steps", "challenge", ["center", "liveness"]);
      if (ch.current !== last) { tone("tick"); last = ch.current; }
      showPrompt(ch.current, ch.message || "");
      setRing(s.progress);
      setCoach(ch.message || "Follow the prompt", "warn");
      $("er-sub").textContent = "Follow each prompt — they appear in random order.";
      const all = [...ch.done, ch.current];
      $("er-chips").innerHTML = all.map((k) => `<span class="chip ${ch.done.includes(k) ? "ok" : "now"}">${ch.done.includes(k) ? "✓ " : ""}${LABEL[k]}</span>`).join("")
        + Array.from({ length: Math.max(0, ch.remaining - 1) }, () => '<span class="chip">•••</span>').join("");
      $("er-count").textContent = `${Math.ceil(ch.time_left_s)} s left for this step`;
    } else if (s.phase === "passed") {
      showPrompt(null); setRing(1); steps("enroll-steps", "save", ["center", "liveness", "challenge"]);
      setCoach("Saving securely…", "good"); $("er-chips").innerHTML = ""; $("er-count").textContent = "";
      try {
        const done = await api("/api/enroll/complete", json({ sid: st.sid }));
        $("ed-photo").src = "data:image/jpeg;base64," + done.photo_b64;
        $("ed-title").textContent = `${done.person.name} is enrolled`;
        $("ed-sub").textContent = `ID ${done.person.person_id}${done.person.role ? " · " + done.person.role : ""} · live person confirmed`;
        enrollView("enroll-done"); tone("ALLOW"); setCoach("Enrollment complete", "good");
        S.replace = false;
      } catch (e) { return enrollFailed(e.message, ""); }
      return;
    } else if (s.phase === "failed") {
      const hint = /timed out|not completed/.test(s.reason) ? "Tip: follow each on-screen prompt within 10 seconds."
        : /liveness/.test(s.reason) ? "Tip: use good, even lighting and make sure nothing covers your face." : "";
      return enrollFailed(s.message || "We couldn't confirm a live person. Please try again.", hint);
    }
    await sleep(80);
  }
}
function enrollFailed(msg, hint) {
  showPrompt(null); setRing(0); tone("DENY");
  $("ef-msg").textContent = msg; $("ef-detail").textContent = hint;
  enrollView("enroll-fail"); setCoach("Enrollment not completed", "warn");
}
function cancelEnroll() {
  S.enrollAbort = true; showPrompt(null); setRing(0);
  if (S.enrollSid) api("/api/enroll/cancel", json({ sid: S.enrollSid })).catch(() => {});
  enrollView("enroll-form");
}

// ---------------- people ----------------
let PEOPLE = [];
async function loadPeople() {
  try { PEOPLE = await api("/api/people"); } catch (e) { $("people-grid").innerHTML = `<div class="empty">${e.message}</div>`; return; }
  renderPeople();
}
function esc(s) { return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
function renderPeople() {
  const q = $("people-search").value.trim().toLowerCase();
  const list = PEOPLE.filter((p) => !q || p.name.toLowerCase().includes(q) || p.person_id.toLowerCase().includes(q));
  $("people-count").textContent = `${PEOPLE.length} enrolled`;
  $("people-grid").innerHTML = list.length ? list.map((p) => `
    <div class="person">
      <img src="/api/people/${encodeURIComponent(p.person_id)}/photo.jpg?t=${Math.round(p.enrolled_at)}" alt="">
      <div><div class="nm">${esc(p.name)}</div><div class="id">${esc(p.person_id)}${p.role ? " · " + esc(p.role) : ""}</div></div>
      <span class="pill ${p.liveness_gated ? "" : "warn"}">${p.liveness_gated ? "liveness-checked" : "quick enroll"}</span>
      <div class="id">${new Date(p.enrolled_at * 1000).toLocaleString([], { dateStyle: "medium", timeStyle: "short" })}</div>
      <div class="row"><button class="btn btn-ghost" data-re="${esc(p.person_id)}">Re-enroll</button><button class="btn btn-danger" data-del="${esc(p.person_id)}">Delete</button></div>
    </div>`).join("") : `<div class="empty">${PEOPLE.length ? "No matches" : "Nobody enrolled yet — use the Enroll tab."}</div>`;
}
async function deletePerson(pid) {
  const p = PEOPLE.find((x) => x.person_id === pid);
  $("confirm-name").textContent = p ? p.name : pid;
  const dlg = $("confirm");
  const ok = await new Promise((res) => {
    $("confirm-yes").onclick = () => { dlg.close(); res(true); };
    $("confirm-no").onclick = () => { dlg.close(); res(false); };
    dlg.showModal();
  });
  if (!ok) return;
  await api(`/api/people/${encodeURIComponent(pid)}`, { method: "DELETE" });
  loadPeople();
}

// ---------------- tabs, connection, keyboard ----------------
function setTab(t) {
  if (S.tab === "enroll" && t !== "enroll" && $("enroll-run").style.display !== "none") cancelEnroll();
  S.tab = t;
  for (const b of document.querySelectorAll(".tab")) b.setAttribute("aria-selected", b.dataset.tab === t);
  for (const id of ["verify", "enroll", "people"]) $("tab-" + id).style.display = id === t ? "grid" : "none";
  $("cam-card").style.display = t === "people" ? "none" : "";
  $("main-layout").classList.toggle("people-layout", t === "people");
  showPrompt(null); hideDecision(); setRing(0);
  if (t === "verify") resetVerify();
  if (t === "enroll") { enrollView("enroll-form"); setCoach("Fill in the details to begin"); }
  if (t === "people") loadPeople();
}
async function ping() {
  const t0 = performance.now();
  try {
    await fetch("/healthz", { cache: "no-store" });
    $("conn-dot").className = "dot on"; $("conn-text").textContent = `online · ${Math.round(performance.now() - t0)} ms`;
  } catch { $("conn-dot").className = "dot off"; $("conn-text").textContent = "offline"; }
}
function onKey(e) {
  const typing = ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName);
  if (e.key === "Escape") { if (S.tab === "verify") resetVerify(); else if (S.tab === "enroll") cancelEnroll(); return; }
  if (e.key === "Enter") {
    if (S.tab === "verify" && S.state === "idle") { e.preventDefault(); startScanning(); }
    else if (S.tab === "enroll" && $("enroll-form").style.display !== "none" && enrollValid()) { e.preventDefault(); startEnroll(); }
    return;
  }
  if (typing) return;
  if (e.key === "1") setTab("verify"); else if (e.key === "2") setTab("enroll"); else if (e.key === "3") setTab("people");
  else if (e.key.toLowerCase() === "s") { S.sound = !S.sound; try { localStorage.setItem("hid.sound", S.sound ? "1" : "0"); } catch { /**/ } renderSound(); }
}

async function startCamera(id) {
  $("cam-error").classList.add("hidden");
  try {
    await openCamera(video, id);
    const cams = await listCameras(), cur = video.srcObject.getVideoTracks()[0].getSettings().deviceId;
    $("camera").innerHTML = cams.map((c, i) => `<option value="${c.deviceId}" ${c.deviceId === cur ? "selected" : ""}>${esc(c.label || "Camera " + (i + 1))}</option>`).join("");
    setCoach("Enter your badge ID to begin");
  } catch (e) { $("cam-error").textContent = e.message; $("cam-error").classList.remove("hidden"); }
}

async function init() {
  renderSound();
  for (const b of document.querySelectorAll(".tab")) b.onclick = () => setTab(b.dataset.tab);
  $("mode-claim").onclick = () => setMode("claim");
  $("mode-open").onclick = () => setMode("open");
  $("verify-start").onclick = startScanning;
  for (const id of ["e-name", "e-id", "e-consent"]) $(id).addEventListener("input", () => { $("e-start").disabled = !enrollValid(); });
  $("e-start").onclick = startEnroll;
  $("er-cancel").onclick = cancelEnroll;
  $("ed-again").onclick = () => { for (const id of ["e-name", "e-id", "e-role"]) $(id).value = ""; $("e-consent").checked = false; $("e-start").disabled = true; enrollView("enroll-form"); };
  $("ef-retry").onclick = () => enrollView("enroll-form");
  $("people-search").oninput = renderPeople;
  $("people-grid").onclick = (e) => {
    const del = e.target.dataset?.del, re = e.target.dataset?.re;
    if (del) deletePerson(del);
    if (re) {
      const p = PEOPLE.find((x) => x.person_id === re);
      S.replace = true; setTab("enroll"); enrollView("enroll-form");
      $("e-name").value = p.name; $("e-id").value = p.person_id; $("e-role").value = p.role || ""; $("e-consent").checked = false;
      $("e-start").disabled = true; $("e-consent").focus();
    }
  };
  $("sound-btn").onclick = () => onKey({ key: "s", preventDefault() {} });
  $("camera").onchange = () => startCamera($("camera").value);
  document.addEventListener("keydown", onKey);
  ping(); setInterval(ping, 5000);
  await startCamera(rememberedCamera());
  resetVerify();
  liveLoop();
}
init();
