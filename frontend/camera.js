// Shared camera helpers (kiosk + bench). Preview is mirrored with CSS only; grabbed frames are NOT mirrored.
const LS_KEY = "hid.cameraId";

export function rememberedCamera() {
  try { return localStorage.getItem(LS_KEY); } catch { return null; }
}

export async function listCameras() {
  const devs = await navigator.mediaDevices.enumerateDevices();
  return devs.filter((d) => d.kind === "videoinput");
}

export function friendlyError(e) {
  if (!window.isSecureContext)
    return "Insecure context: camera only works on https:// or http://localhost.\nFix: open via the CloudFront https URL or http://localhost:8000.";
  if (!navigator.mediaDevices?.getUserMedia)
    return "This browser has no camera API (getUserMedia). Use current Chrome.";
  switch (e?.name) {
    case "NotAllowedError": case "SecurityError":
      return "Camera permission denied.\nFix: click the camera icon in the address bar → Allow, then reload. On macOS also check System Settings → Privacy & Security → Camera → Google Chrome.";
    case "NotFoundError": case "OverconstrainedError":
      return "No camera found (or the selected one disappeared).\nFix: plug in / unlock the camera (Continuity Camera: keep iPhone near and locked), then pick it in the dropdown.";
    case "NotReadableError": case "AbortError":
      return "Camera is in use by another app or tab (Zoom, FaceTime, Photo Booth, another bench tab).\nFix: close that app/tab, then reload.";
    default:
      return `Camera error: ${e?.name || ""} ${e?.message || e}`;
  }
}

export async function openCamera(video, deviceId) {
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) throw new Error(friendlyError(null));
  if (video.srcObject) video.srcObject.getTracks().forEach((t) => t.stop());
  const video_c = { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } };
  if (deviceId) video_c.deviceId = { exact: deviceId }; else video_c.facingMode = "user";
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: video_c, audio: false });
  } catch (e) {
    if (deviceId && (e.name === "OverconstrainedError" || e.name === "NotFoundError")) return openCamera(video, null);
    const err = new Error(friendlyError(e)); err.cause = e; throw err;
  }
  video.srcObject = stream;
  await video.play();
  const id = stream.getVideoTracks()[0].getSettings().deviceId;
  try { if (id) localStorage.setItem(LS_KEY, id); } catch { /* storage may be blocked */ }
  return stream;
}

// Wait for the next decoded video frame (requestVideoFrameCallback where available).
export function nextFrame(video) {
  return new Promise((res) => {
    if (video.requestVideoFrameCallback) video.requestVideoFrameCallback(() => res());
    else setTimeout(res, 33);
  });
}

export function grabJpeg(video, canvas, maxW = 640, quality = 0.9) {
  const s = Math.min(1, maxW / video.videoWidth);
  canvas.width = Math.round(video.videoWidth * s);
  canvas.height = Math.round(video.videoHeight * s);
  canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height); // un-mirrored
  return new Promise((res) => canvas.toBlob(res, "image/jpeg", quality));
}

// n frames spread over ~spanMs.
export async function captureFrames(video, canvas, n = 5, spanMs = 500, maxW = 640, quality = 0.9) {
  const out = [];
  const gap = spanMs / Math.max(n - 1, 1);
  for (let i = 0; i < n; i++) {
    const t0 = performance.now();
    await nextFrame(video);
    out.push(await grabJpeg(video, canvas, maxW, quality));
    const wait = gap - (performance.now() - t0);
    if (i < n - 1 && wait > 0) await new Promise((r) => setTimeout(r, wait));
  }
  return out;
}

// Resolves after the browser has painted (for honest "decision shown" timing).
export function afterPaint() {
  return new Promise((res) => requestAnimationFrame(() => setTimeout(res, 0)));
}
