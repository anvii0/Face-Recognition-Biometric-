"""Fetch pinned model weights and convert MiniFASNet (PyTorch) -> ONNX.

Run once on a dev machine (needs requirements-tools.txt, i.e. torch):
    .venv/bin/python scripts/fetch_models.py
Outputs into models/: minifasnet_v2_2.7.onnx, minifasnet_v1se_4.0.onnx, yunet.onnx
The server only needs the .onnx files (deploy uploads them; torch is never installed on EC2).
"""
from __future__ import annotations

import hashlib
import ssl
import subprocess
import sys
import tempfile
import urllib.request
from collections import OrderedDict
from pathlib import Path

import certifi
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
SFAS_REPO = "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing"
SFAS_COMMIT = "b6d5f04ad78778917853b25c778acef6d5626d15"
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/48b10fa505aeb51fbd127fc60a5c660040e823b7/"
             "models/face_detection_yunet/face_detection_yunet_2023mar.onnx")
# (pth file, output onnx, constructor name, crop scale)
MINIFAS = [
    ("2.7_80x80_MiniFASNetV2.pth", "minifasnet_v2_2.7.onnx", "MiniFASNetV2"),
    ("4_0_0_80x80_MiniFASNetV1SE.pth", "minifasnet_v1se_4.0.onnx", "MiniFASNetV1SE"),
]


def download(url: str, dest: Path) -> None:
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(url, timeout=60, context=ctx) as r:
        dest.write_bytes(r.read())


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def convert(sfas_dir: Path) -> None:
    import onnxruntime as ort
    import torch

    sys.path.insert(0, str(sfas_dir))  # only for the model *definition* module
    from src.model_lib import MiniFASNet as M  # type: ignore[import-not-found]

    for pth, out, ctor in MINIFAS:
        state = torch.load(sfas_dir / "resources/anti_spoof_models" / pth, map_location="cpu",
                           weights_only=True)
        state = OrderedDict((k[7:] if k.startswith("module.") else k, v) for k, v in state.items())
        model = getattr(M, ctor)(conv6_kernel=(5, 5))  # get_kernel(80, 80) == (5, 5)
        model.load_state_dict(state)
        model.eval()
        dummy = torch.zeros(1, 3, 80, 80)
        torch.onnx.export(model, dummy, str(MODELS / out), input_names=["input"],
                          output_names=["logits"], dynamic_axes={"input": {0: "n"}, "logits": {0: "n"}},
                          opset_version=17, dynamo=False)
        # Parity check: torch vs onnxruntime on random 0-255 input (upstream feeds raw BGR 0-255)
        x = torch.rand(4, 3, 80, 80) * 255
        with torch.no_grad():
            ref = torch.softmax(model(x), dim=1).numpy()
        sess = ort.InferenceSession(str(MODELS / out), providers=["CPUExecutionProvider"])
        logits = sess.run(None, {"input": x.numpy()})[0]
        got = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
        err = float(np.abs(ref - got).max())
        print(f"{out}: parity max|diff|={err:.2e} sha256={sha256(MODELS / out)[:16]}")
        assert err < 1e-4, "ONNX parity check failed"


def main() -> None:
    MODELS.mkdir(exist_ok=True)
    yunet = MODELS / "yunet.onnx"
    if not yunet.exists():
        download(YUNET_URL, yunet)
    print(f"yunet.onnx sha256={sha256(yunet)[:16]} size={yunet.stat().st_size}")
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "sfas"
        subprocess.run(["git", "clone", "-q", SFAS_REPO, str(d)], check=True)
        subprocess.run(["git", "-C", str(d), "checkout", "-q", SFAS_COMMIT], check=True)
        convert(d)


if __name__ == "__main__":
    main()
