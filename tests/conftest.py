import copy
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg():
    return copy.deepcopy(yaml.safe_load((ROOT / "config.yaml").read_text()))


GOOD_Q = {"blur": 200.0, "brightness": 120.0, "face_frac": 0.4, "yaw": 0.0, "roll_deg": 0.0}


@pytest.fixture
def raw():
    def make(**kw):
        r = {"n_faces": 1, "quality": dict(GOOD_Q), "minifas": 0.9, "moire": 0.2, "motion": 0.05,
             "similarity": 99.0, "match_error": None, "user": "alice",
             "face_box": {"Left": 0.4, "Top": 0.3, "Width": 0.2, "Height": 0.3},
             "cloud": {"vlm": {"live": 0.95, "attack": "none", "evidence": "natural depth and shading"},
                       "devices": [], "weak": [],
                       "faces": [{"box": {"Left": 0.4, "Top": 0.3, "Width": 0.2, "Height": 0.3}, "yaw": 0.0,
                                  "brightness": 70.0, "sharpness": 60.0, "occluded": False}],
                       "errors": {}, "ms": {"vlm": 950.0, "labels": 450.0, "faces": 400.0}}}
        r.update(kw)
        return r
    return make
