"""Face detection, quality gates and liveness cues. Pure CPU (OpenCV + ONNX Runtime)."""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

MODELS = Path(__file__).resolve().parents[1] / "models"
# (onnx file, crop scale) — the upstream ensemble; scores are averaged
MINIFAS = [("minifasnet_v2_2.7.onnx", 2.7), ("minifasnet_v1se_4.0.onnx", 4.0)]


@dataclass
class Face:
    box: tuple[int, int, int, int]  # x, y, w, h in frame pixels
    landmarks: np.ndarray            # 5x2: right eye, left eye, nose, right mouth, left mouth
    score: float


@dataclass
class FrameResult:
    width: int
    height: int
    faces: list[Face]
    quality: dict[str, float] = field(default_factory=dict)
    liveness: float | None = None    # MiniFASNet real-class probability (0..1)
    moire: float | None = None       # raw high-frequency peak energy share
    crops: dict[str, np.ndarray] = field(default_factory=dict)


def decode_jpeg(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode image")
    return img


class Vision:
    def __init__(self, score_threshold: float = 0.7, max_side: int = 640) -> None:
        self.max_side = max_side
        self._det_lock = threading.Lock()  # FaceDetectorYN is stateful (input size); not thread-safe
        self.detector = cv2.FaceDetectorYN.create(str(MODELS / "yunet.onnx"), "", (320, 320),
                                                  score_threshold, 0.3, 5)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        self.sessions = [(ort.InferenceSession(str(MODELS / f), opts, providers=["CPUExecutionProvider"]), s)
                         for f, s in MINIFAS]

    # ---------- detection ----------
    def downscale(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        s = self.max_side / max(h, w)
        return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img

    def detect(self, img: np.ndarray) -> list[Face]:
        h, w = img.shape[:2]
        with self._det_lock:
            self.detector.setInputSize((w, h))
            _, raw = self.detector.detect(img)
        faces = [] if raw is None else [
            Face(tuple(int(v) for v in r[:4]), r[4:14].reshape(5, 2).copy(), float(r[14])) for r in raw]
        return sorted(faces, key=lambda f: f.box[2] * f.box[3], reverse=True)

    # ---------- quality ----------
    @staticmethod
    def quality(img: np.ndarray, face: Face) -> dict[str, float]:
        x, y, w, h = face.box
        H, W = img.shape[:2]
        roi = img[max(y, 0):min(y + h, H), max(x, 0):min(x + w, W)]
        grey = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        grey = cv2.resize(grey, (112, int(112 * grey.shape[0] / max(grey.shape[1], 1))))
        re, le, nose = face.landmarks[0], face.landmarks[1], face.landmarks[2]
        eye_d = float(np.linalg.norm(le - re)) or 1.0
        mid = (re + le) / 2
        return {
            "blur": float(cv2.Laplacian(grey, cv2.CV_64F).var()),
            "brightness": float(grey.mean()),
            "face_frac": h / H,
            "yaw": float((nose[0] - mid[0]) / eye_d),
            "roll_deg": float(math.degrees(math.atan2(le[1] - re[1], le[0] - re[0]))),
        }

    # ---------- MiniFASNet ----------
    @staticmethod
    def crop(img: np.ndarray, box: tuple[int, int, int, int], scale: float, out: int = 80) -> np.ndarray:
        """Upstream CropImage logic. YuNet's box is made square ((w+h)/2) to mimic the
        RetinaFace boxes MiniFASNet was trained on (verified close on a sample image)."""
        H, W = img.shape[:2]
        x, y, w, h = box
        side = (w + h) / 2
        cx, cy = x + w / 2, y + h / 2
        scale = min((H - 1) / side, (W - 1) / side, scale)
        nw = nh = side * scale
        x0, y0, x1, y1 = cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2
        if x0 < 0: x1 -= x0; x0 = 0  # noqa: E701
        if y0 < 0: y1 -= y0; y0 = 0  # noqa: E701
        if x1 > W - 1: x0 -= x1 - W + 1; x1 = W - 1  # noqa: E701
        if y1 > H - 1: y0 -= y1 - H + 1; y1 = H - 1  # noqa: E701
        return cv2.resize(img[int(y0):int(y1) + 1, int(x0):int(x1) + 1], (out, out))

    def liveness(self, img: np.ndarray, face: Face, crops: dict[str, np.ndarray] | None = None) -> float:
        probs = np.zeros(3)
        for sess, scale in self.sessions:
            c = self.crop(img, face.box, scale)
            if crops is not None:
                crops[f"minifas_{scale}"] = c
            logits = sess.run(None, {"input": c.transpose(2, 0, 1)[None].astype(np.float32)})[0][0]
            e = np.exp(logits - logits.max())
            probs += e / e.sum()
        return float(probs[1] / len(self.sessions))  # class 1 == real

    # ---------- FFT / moire cue ----------
    @staticmethod
    def moire(img: np.ndarray, face: Face, crops: dict[str, np.ndarray] | None = None) -> float:
        """Share of high-band spectral energy held by the top 1% of bins. Screens and some prints
        produce sharp periodic peaks (moire / pixel grid); natural skin has smooth fall-off.
        UNCALIBRATED heuristic — its usefulness is measured in Phase 3, not assumed."""
        x, y, w, h = face.box
        H, W = img.shape[:2]
        side = int(max(w, h) * 1.2)
        cx, cy = x + w // 2, y + h // 2
        x0, y0 = max(cx - side // 2, 0), max(cy - side // 2, 0)
        roi = cv2.cvtColor(img[y0:min(y0 + side, H), x0:min(x0 + side, W)], cv2.COLOR_BGR2GRAY)
        g = cv2.resize(roi, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32)
        g = (g - g.mean()) * np.outer(np.hanning(128), np.hanning(128))
        mag = np.abs(np.fft.fftshift(np.fft.fft2(g)))
        yy, xx = np.mgrid[-64:64, -64:64]
        r = np.hypot(xx, yy)
        band = mag[(r >= 16) & (r <= 60)] ** 2
        k = max(int(band.size * 0.01), 1)
        score = float(np.sort(band)[-k:].sum() / (band.sum() + 1e-9))
        if crops is not None:
            spec = np.log1p(mag)
            crops["fft"] = cv2.normalize(spec, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return score


def motion_score(landmarks: list[np.ndarray]) -> float | None:
    """Non-rigid landmark motion across frames: RMS residual after the best similarity transform
    to frame 0, as a fraction of inter-eye distance. A flat photo/screen moves rigidly (~0)."""
    if len(landmarks) < 2:
        return None
    ref = landmarks[0].astype(np.float32)
    eye_d = float(np.linalg.norm(ref[1] - ref[0])) or 1.0
    res = []
    for lm in landmarks[1:]:
        m, _ = cv2.estimateAffinePartial2D(lm.astype(np.float32), ref)
        if m is None:
            continue
        warped = lm @ m[:, :2].T + m[:, 2]
        res.append(float(np.sqrt(((warped - ref) ** 2).sum(1).mean())) / eye_d)
    return float(np.mean(res)) if res else None
