"""Cloud signals (ported from hid-gate): strict parsing, time budget / fail-closed, variants D and E, geometry."""
import copy
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from backend.cloud_signals import CloudSignals, ParseError, _timed, parse_gesture, parse_liveness
from backend.decision import ALLOW, DENY, STEP_UP, decide, device_over_face, face_centre_in_box

PHONE_OVER = {"name": "Mobile Phone", "conf": 97.0, "box": {"Left": 0.3, "Top": 0.2, "Width": 0.4, "Height": 0.6}}
PHONE_OFF = {"name": "Mobile Phone", "conf": 97.0, "box": {"Left": 0.0, "Top": 0.0, "Width": 0.15, "Height": 0.2}}


def with_cloud(raw, **kw):
    r = copy.deepcopy(raw)
    r["cloud"].update(kw)
    return r


# ---------------- strict parsing ----------------
def test_parse_liveness_valid_and_fenced():
    d = parse_liveness('{"live": 0.9, "attack": "none", "evidence": "natural depth"}')
    assert d == {"live": 0.9, "attack": "none", "evidence": "natural depth"}
    assert parse_liveness('```json\n{"live": 1, "attack": "screen", "evidence": "bezel"}\n```')["live"] == 1.0


def test_parse_liveness_truncates_evidence_to_ten_words():
    d = parse_liveness('{"live": 0.2, "attack": "printed_photo", "evidence": "' + " ".join(["w"] * 15) + '"}')
    assert len(d["evidence"].split()) == 10


@pytest.mark.parametrize("reply", [
    'Sure! {"live": 0.9, "attack": "none", "evidence": "x"}',         # prose around the JSON
    '{"live": 0.9, "attack": "none"}',                                # missing field
    '{"live": "0.9", "attack": "none", "evidence": "x"}',             # wrong type
    '{"live": true, "attack": "none", "evidence": "x"}',              # bool is not a number
    '{"live": 1.4, "attack": "none", "evidence": "x"}',               # out of range
    '{"live": 0.9, "attack": "deepfake", "evidence": "x"}',           # unknown attack
    '[{"live": 0.9}]', "", "I cannot help with that."])
def test_parse_liveness_rejects(reply):
    with pytest.raises(ParseError):
        parse_liveness(reply)


def test_parse_gesture():
    assert parse_gesture('{"fingers": 3, "live": 0.8}') == {"fingers": 3, "live": 0.8}
    for bad in ('{"fingers": 7, "live": 0.8}', '{"fingers": 2.5, "live": 0.8}', '{"fingers": 2}'):
        with pytest.raises(ParseError):
            parse_gesture(bad)


def test_timed_classifies_parse_error_and_exceptions():
    def boom():
        raise ParseError("x")
    assert _timed(boom)[2] == "parse_error"
    assert _timed(lambda: 1 / 0)[2] == "ZeroDivisionError"
    assert _timed(lambda: 5)[0] == 5


def test_finish_marks_unfinished_checks_as_timeout():
    pool = ThreadPoolExecutor(3)
    futs = {"vlm": pool.submit(_timed, lambda: time.sleep(0.5) or {"live": 0.9}),
            "labels": pool.submit(_timed, lambda: {"devices": [PHONE_OFF], "weak": ["Paper"]}),
            "faces": pool.submit(_timed, lambda: [])}
    t0 = time.perf_counter()
    out = CloudSignals.finish(object(), futs, time.perf_counter() + 0.1)  # finish() does not use self
    assert time.perf_counter() - t0 < 0.4                      # never waits past the deadline
    assert out["errors"] == {"vlm": "timeout"} and out["vlm"] is None and out["ms"]["vlm"] is None
    assert out["devices"] == [PHONE_OFF] and out["weak"] == ["Paper"] and out["faces"] == []


# ---------------- geometry ----------------
def test_face_centre_in_box_inclusive_edges():
    face = {"Left": 0.4, "Top": 0.4, "Width": 0.2, "Height": 0.2}          # centre (0.5, 0.5)
    assert face_centre_in_box(face, {"Left": 0.5, "Top": 0.5, "Width": 0.1, "Height": 0.1})
    assert not face_centre_in_box(face, {"Left": 0.51, "Top": 0.0, "Width": 0.4, "Height": 1.0})


def test_device_over_face_prefers_rekognition_box_then_local(raw):
    r = with_cloud(raw(), devices=[PHONE_OVER, PHONE_OFF])
    assert device_over_face(r)["box"] == PHONE_OVER["box"]
    r = with_cloud(raw(), devices=[PHONE_OFF])
    assert device_over_face(r) is None
    r = with_cloud(raw(), devices=[PHONE_OVER], faces=[])          # Rekognition saw no face -> our box
    assert device_over_face(r) is not None
    r["face_box"] = None
    assert device_over_face(r) is None


# ---------------- variant E (fused, fail closed) ----------------
def test_e_allows_when_everything_agrees(cfg, raw):
    d = decide(raw(), cfg, "E")
    assert d["decision"] == ALLOW and "natural depth" in d["reason"]


@pytest.mark.parametrize("live,expected", [(0.30, DENY), (0.31, STEP_UP), (0.7499, STEP_UP), (0.75, ALLOW)])
def test_e_vlm_boundaries(cfg, raw, live, expected):
    r = with_cloud(raw(), vlm={"live": live, "attack": "none", "evidence": "e"})
    assert decide(r, cfg, "E")["decision"] == expected


def test_e_vlm_spoof_reason_quotes_evidence_and_local_score(cfg, raw):
    r = with_cloud(raw(), vlm={"live": 0.1, "attack": "screen", "evidence": "phone bezel visible"})
    d = decide(r, cfg, "E")
    assert d["decision"] == DENY and '"phone bezel visible"' in d["reason"] and "local 0.90" in d["reason"]


def test_e_device_over_face_denies_even_with_high_vlm(cfg, raw):
    d = decide(with_cloud(raw(), devices=[PHONE_OVER]), cfg, "E")
    assert d["decision"] == DENY and "Mobile Phone" in d["reason"]
    assert decide(with_cloud(raw(), devices=[PHONE_OFF]), cfg, "E")["decision"] == ALLOW


@pytest.mark.parametrize("key,err", [("vlm", "timeout"), ("vlm", "parse_error"), ("labels", "timeout"),
                                     ("labels", "ThrottlingException")])
def test_e_missing_cloud_signal_never_allows(cfg, raw, key, err):
    r = with_cloud(raw(), errors={key: err})
    r["cloud"]["vlm" if key == "vlm" else "devices"] = None
    d = decide(r, cfg, "E")
    assert d["decision"] == STEP_UP and err in d["reason"]


def test_e_without_any_cloud_data_steps_up(cfg, raw):
    r = raw()
    r["cloud"] = None
    assert decide(r, cfg, "E")["decision"] == STEP_UP


def test_e_local_liveness_and_identity_denies(cfg, raw):
    assert decide(raw(minifas=0.1), cfg, "E")["decision"] == DENY                  # local spoof, VLM 0.95
    d = decide(raw(similarity=10.0), cfg, "E")
    assert d["decision"] == DENY and "does not match" in d["reason"]


def test_e_local_borderline_steps_up(cfg, raw):
    d = decide(raw(minifas=0.6), cfg, "E")
    assert d["decision"] == STEP_UP and "local liveness 0.60" in d["reason"]


def test_e_spoof_evidence_beats_poor_quality(cfg, raw):
    r = with_cloud(raw(), vlm={"live": 0.05, "attack": "printed_photo", "evidence": "paper edges"})
    r["quality"]["brightness"] = 10.0
    assert decide(r, cfg, "E")["decision"] == DENY          # a dark photo is still a photo
    r["cloud"]["vlm"]["live"] = 0.9
    assert decide(r, cfg, "E")["decision"] == STEP_UP       # just dark -> step-up


def test_e_rekognition_occluded_is_a_quality_failure(cfg, raw):
    r = raw()
    r["cloud"]["faces"][0]["occluded"] = True
    d = decide(r, cfg, "E")
    assert d["decision"] == STEP_UP and "covered" in d["reason"]


# ---------------- variant D (hid-gate rules as ported) ----------------
def test_d_allow_and_sim_boundaries(cfg, raw):
    assert decide(raw(), cfg, "D")["decision"] == ALLOW
    assert decide(raw(similarity=94.9), cfg, "D")["decision"] == STEP_UP    # < d_sim_strong 95
    assert decide(raw(similarity=80.0), cfg, "D")["decision"] == STEP_UP
    assert decide(raw(similarity=79.9), cfg, "D")["decision"] == DENY       # < d_sim_min 80


def test_d_ignores_local_liveness(cfg, raw):
    assert decide(raw(minifas=0.0), cfg, "D")["decision"] == ALLOW


def test_d_device_with_missing_vlm_denies_and_weak_hint_steps_up(cfg, raw):
    r = with_cloud(raw(), devices=[PHONE_OVER], vlm=None, errors={"vlm": "timeout"})
    assert decide(r, cfg, "D")["decision"] == DENY
    d = decide(with_cloud(raw(), weak=["Paper"]), cfg, "D")
    assert d["decision"] == STEP_UP and "Paper" in d["reason"]


def test_d_vlm_timeout_steps_up(cfg, raw):
    d = decide(with_cloud(raw(), vlm=None, errors={"vlm": "timeout"}), cfg, "D")
    assert d["decision"] == STEP_UP and "timeout" in d["reason"]


def test_d_rekognition_face_checks(cfg, raw):
    assert decide(with_cloud(raw(), faces=None, errors={"faces": "timeout"}), cfg, "D")["decision"] == STEP_UP
    assert decide(with_cloud(raw(), faces=[]), cfg, "D")["decision"] == STEP_UP


def test_abc_unaffected_by_cloud_signals(cfg, raw):
    r = with_cloud(raw(), vlm={"live": 0.0, "attack": "screen", "evidence": "x"}, devices=[PHONE_OVER])
    for v in "ABC":
        assert decide(r, cfg, v) == decide(raw(), cfg, v)
