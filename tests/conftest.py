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
             "similarity": 99.0, "match_error": None, "user": "alice"}
        r.update(kw)
        return r
    return make
