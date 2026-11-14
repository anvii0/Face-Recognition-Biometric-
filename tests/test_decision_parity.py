"""backend/decision.py and frontend/decision.js must agree (the bench re-decides attempts in the browser)."""
import json
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from backend.decision import decide

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")


def cases(n=600, seed=7):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        q = {"blur": rng.uniform(0, 300), "brightness": rng.uniform(0, 255), "face_frac": rng.uniform(0, 0.8),
             "yaw": rng.uniform(-0.6, 0.6), "roll_deg": rng.uniform(-40, 40)}
        if rng.random() < 0.6:  # bias towards passing quality so the deeper branches get exercised
            q.update(blur=200, brightness=120, face_frac=0.4, yaw=0.0, roll_deg=0.0)
        out.append({
            "raw": {"n_faces": rng.choice([0, 1, 1, 1, 1, 2]), "quality": q, "minifas": rng.random(),
                    "moire": rng.choice([None, rng.uniform(0, 0.8)]), "motion": rng.choice([None, rng.uniform(0, 0.05)]),
                    "similarity": rng.choice([None, rng.uniform(0, 100), rng.uniform(85, 100)]),
                    "match_error": rng.choice([None, "x"]), "user": "u"},
            "variant": rng.choice("ABC"),
            "th": {"t_lo": round(rng.uniform(0.1, 0.5), 2), "t_hi": round(rng.uniform(0.5, 0.95), 2),
                   "t_match": round(rng.uniform(80, 99), 1), "t_match_lo": round(rng.uniform(20, 70), 1)},
        })
    return out


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_python_and_js_decisions_match(cfg, tmp_path):
    cs = cases()
    for c in cs:
        if c["raw"]["n_faces"] == 0:
            c["raw"]["quality"] = None
    cfgs = [{**cfg, "thresholds": c["th"]} for c in cs]
    (tmp_path / "in.json").write_text(json.dumps([{"raw": c["raw"], "cfg": k, "variant": c["variant"]} for c, k in zip(cs, cfgs)]))
    script = (f"import {{ decide }} from '{(ROOT / 'frontend/decision.js').as_uri()}';"
              "import fs from 'fs'; const xs = JSON.parse(fs.readFileSync(process.argv[1]));"
              "console.log(JSON.stringify(xs.map(x => decide(x.raw, x.cfg, x.variant))));")
    js = json.loads(subprocess.run([NODE, "--input-type=module", "-e", script, str(tmp_path / "in.json")],
                                   capture_output=True, text=True, check=True).stdout)
    mismatches = []
    for c, k, j in zip(cs, cfgs, js):
        p = decide(c["raw"], k, c["variant"])
        if (p["decision"], p["confidence"], p["liveness"]) != (j["decision"], j["confidence"], j["liveness"]) \
                or p["reason"].split(":")[0] != j["reason"].split(":")[0]:
            mismatches.append((c, p, j))
    assert not mismatches, f"{len(mismatches)} mismatches, first: {mismatches[0]}"
    assert {p["decision"] for p in js} == {"ALLOW", "REQUEST_VERIFICATION", "DENY"}  # all branches covered
