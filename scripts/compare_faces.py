"""Check Rekognition CompareFaces on real images (nothing is stored or uploaded to S3).

    .venv/bin/python scripts/compare_faces.py SAME_PERSON_A.jpg SAME_PERSON_B.jpg [OTHER_PERSON.jpg]
Prints similarity for each pair and the latency of each call.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.matcher import RekognitionMatcher  # noqa: E402

paths = [Path(p) for p in sys.argv[1:]]
if len(paths) < 2:
    sys.exit(__doc__)
m = RekognitionMatcher()
imgs = {p.name: p.read_bytes() for p in paths}
m.compare(imgs[paths[0].name], imgs[paths[0].name])  # warm-up (first real call measured ~2 s)
for a, b in itertools.combinations(imgs, 2):
    sim, err, ms = m.compare(imgs[a], imgs[b])
    print(f"{a} vs {b}: similarity={'—' if sim is None else f'{sim:.2f}'} err={err} ({ms:.0f} ms)")
