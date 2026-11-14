"""Active-challenge state machines on synthetic signal sequences (no camera, no models)."""
import random

import pytest

from backend.challenge import ChallengeRunner, _Tracker

OPEN = {"blink_l": 0.05, "blink_r": 0.05, "smile": 0.1}
SHUT = {"blink_l": 0.8, "blink_r": 0.75, "smile": 0.1}
WINK = {"blink_l": 0.8, "blink_r": 0.05, "smile": 0.1}


@pytest.fixture
def cc(cfg):
    return cfg["challenge"]


def run(tr, seq):
    return [tr.update(s, y) for s, y in seq]


def test_blink_needs_open_closed_open(cc):
    assert run(_Tracker("blink", cc), [(OPEN, 0), (SHUT, 0), (OPEN, 0)])[-1] is True


def test_blink_rejects_eyes_always_closed_or_open(cc):
    assert not any(run(_Tracker("blink", cc), [(SHUT, 0)] * 10))   # printed photo with closed eyes
    assert not any(run(_Tracker("blink", cc), [(OPEN, 0)] * 10))   # static photo


def test_wink_is_not_a_blink(cc):
    assert not any(run(_Tracker("blink", cc), [(OPEN, 0), (WINK, 0), (OPEN, 0)] * 3))


def test_turn_left_requires_frontal_start_and_sustained_turn(cc):
    s = cc["left_yaw_sign"]
    assert run(_Tracker("turn_left", cc), [(None, 0.0), (None, 0.3 * s), (None, 0.3 * s)])[-1] is True
    assert not any(run(_Tracker("turn_left", cc), [(None, 0.3 * s)] * 5))  # never frontal first
    assert not any(run(_Tracker("turn_left", cc), [(None, 0.0)] + [(None, -0.3 * s)] * 5))  # wrong way


def test_turn_single_frame_spike_is_ignored(cc):
    s = cc["left_yaw_sign"]
    assert not any(run(_Tracker("turn_right", cc), [(None, 0.0), (None, -0.3 * s), (None, 0.0), (None, -0.3 * s), (None, 0.0)]))


def test_runner_passes_all_in_random_order(cc):
    r = ChallengeRunner(["blink", "turn_left"], cc, random.Random(1))
    s = cc["left_yaw_sign"]
    script = {"blink": [(OPEN, 0.0), (SHUT, 0.0), (OPEN, 0.0)], "turn_left": [(OPEN, 0.0), (OPEN, 0.3 * s), (OPEN, 0.3 * s)]}
    for _ in range(2):
        for sig, yaw in script[r.cur.kind]:
            st = r.update(sig, yaw, now=r.cur.started + 1)
    assert st["status"] == "passed" and sorted(st["done"]) == ["blink", "turn_left"]


def test_runner_timeout_retries_once_then_fails(cc):
    r = ChallengeRunner(["blink"], cc, random.Random(0))
    first = r.cur.kind
    st = r.update(OPEN, 0.0, now=r.cur.started + cc["timeout_s"] + 1)
    assert st["status"] == "running" and r.cur.kind != first  # replaced by a different challenge
    st = r.update(OPEN, 0.0, now=r.cur.started + cc["timeout_s"] + 1)
    assert st["status"] == "failed"
