from dataclasses import replace
import math

import pytest

from reach.controller import decide, map_vector, predict_tip
from reach.types import Frame, HandObs, Instance, TargetObs


def test_mapping_directions_deadband_distance_and_flip():
    right = map_vector((100, 200), (300, 200), 640, 480)
    assert right[3].pan > 0 and right[3].noise > 0
    assert map_vector((300, 200), (100, 200), 640, 480)[3].pan < 0
    assert map_vector((100, 300), (100, 100), 640, 480)[3].freq_hz > 440
    assert map_vector((100, 100), (100, 300), 640, 480)[3].freq_hz < 440
    assert map_vector((100, 100), (110, 100), 640, 480)[3].pan == 0
    far = map_vector((0, 0), (640, 480), 640, 480)
    near = map_vector((0, 0), (1, 1), 640, 480)
    assert far[3].pan == 1 and far[3].freq_hz == 220 and far[3].pulse_hz == 2
    assert near[3].pulse_hz > 11
    flip = map_vector((100, 200), (300, 100), 640, 480, flipped=True)
    assert flip[3].pan < 0 and flip[3].freq_hz < 440


def test_prediction_speed_and_dual_clamp():
    assert predict_tip((0, 0), (49, 0), (100, 0)) == (0, 0)
    assert predict_tip((0, 0), (1000, 0), (1000, 0)) == (40, 0)
    assert predict_tip((0, 0), (1000, 0), (20, 0)) == (5, 0)
    assert predict_tip((0, 0), (1000, 0), (20, 0), k=0) == (0, 0)


def test_decision_silence_and_input_frame_ids():
    frame = Frame(10, 1, 1, 640, 480, "hash", "file")
    hand = HandObs(9, True, (100, 200), (100, 230), (), (0, 0), 1)
    target = TargetObs(8, "wallet", (Instance("a", (300, 100, 400, 200), 1, None),), 0, "sam", 100)
    value = decide(hand, target, frame)
    assert value.cue.pan > 0 and value.hand_frame_id == 9 and value.target_frame_id == 8
    assert decide(replace(hand, present=False), target, frame).cue is None
    assert decide(hand, None, frame).cue is None
    with pytest.raises(ValueError):
        map_vector((math.nan, 0), (1, 1), 640, 480)
