"""Decision-layer boundary cases (thresholds are inclusive at t_hi / t_match, exclusive below t_lo / t_match_lo)."""
import pytest

from backend.decision import ALLOW, DENY, STEP_UP, decide, fuse


def th(cfg, **kw):
    cfg["thresholds"].update(kw)
    return cfg


def test_allow_when_live_and_matched(cfg, raw):
    d = decide(raw(), cfg, "A")
    assert d["decision"] == ALLOW and "alice" in d["reason"]


@pytest.mark.parametrize("live,expected", [(0.70, ALLOW), (0.6999, STEP_UP), (0.35, STEP_UP), (0.3499, DENY)])
def test_liveness_boundaries(cfg, raw, live, expected):
    cfg = th(cfg, t_lo=0.35, t_hi=0.70)
    assert decide(raw(minifas=live), cfg, "A")["decision"] == expected


@pytest.mark.parametrize("sim,expected", [(90.0, ALLOW), (89.99, STEP_UP), (50.0, STEP_UP), (49.99, DENY)])
def test_match_boundaries(cfg, raw, sim, expected):
    cfg = th(cfg, t_match=90.0, t_match_lo=50.0)
    assert decide(raw(similarity=sim), cfg, "A")["decision"] == expected


def test_liveness_deny_beats_match(cfg, raw):
    d = decide(raw(minifas=0.1, similarity=99.9), cfg, "A")
    assert d["decision"] == DENY and "presentation attack" in d["reason"]


def test_no_face_and_multiple_faces_step_up(cfg, raw):
    assert decide(raw(n_faces=0, quality=None), cfg, "A")["decision"] == STEP_UP
    d = decide(raw(n_faces=2), cfg, "A")
    assert d["decision"] == STEP_UP and "2 faces" in d["reason"]


@pytest.mark.parametrize("field,value,needle", [
    ("brightness", 20.0, "too dark"), ("brightness", 250.0, "overexposed"), ("blur", 5.0, "blurry"),
    ("face_frac", 0.05, "too small"), ("yaw", 0.9, "head turned"), ("roll_deg", 45.0, "tilted")])
def test_quality_gates_step_up_even_if_scores_good(cfg, raw, field, value, needle):
    r = raw()
    r["quality"][field] = value
    d = decide(r, cfg, "A")
    assert d["decision"] == STEP_UP and needle in d["reason"]


def test_quality_gate_takes_precedence_over_spoof(cfg, raw):
    """Low quality never yields a hard DENY: the step-up challenge resolves it."""
    r = raw(minifas=0.0)
    r["quality"]["brightness"] = 10.0
    assert decide(r, cfg, "A")["decision"] == STEP_UP


def test_matcher_error_steps_up(cfg, raw):
    d = decide(raw(similarity=None, match_error="Rekognition unreachable"), cfg, "A")
    assert d["decision"] == STEP_UP and "unreachable" in d["reason"]


def test_variant_b_moire_penalty_and_reason(cfg, raw):
    f = cfg["fusion"]
    r = raw(minifas=0.6, moire=f["moire_hi"] + 0.1)  # full moire evidence -> penalty w_moire
    assert fuse(r, f, "A")["liveness"] == pytest.approx(0.6)
    assert fuse(r, f, "B")["liveness"] == pytest.approx(0.6 - f["w_moire"])
    d = decide(r, cfg, "B")
    assert d["decision"] == DENY and "screen replay suspected" in d["reason"]


def test_variant_c_motion_penalty(cfg, raw):
    f = cfg["fusion"]
    r = raw(minifas=0.8, moire=0.0, motion=0.0)
    assert fuse(r, f, "C")["liveness"] == pytest.approx(0.8 - f["w_motion"])
    assert fuse(r, f, "B")["liveness"] == pytest.approx(0.8)  # motion ignored outside C


def test_missing_motion_is_not_penalised(cfg, raw):
    assert fuse(raw(minifas=0.8, motion=None, moire=0.0), cfg["fusion"], "C")["liveness"] == pytest.approx(0.8)


def test_confidence_in_unit_range(cfg, raw):
    for r in (raw(), raw(minifas=0.1), raw(similarity=10.0), raw(minifas=0.5)):
        assert 0.0 <= decide(r, cfg, "C")["confidence"] <= 1.0
