from types import SimpleNamespace as NS
from dataclasses import asdict
import json

import numpy as np
import pytest

from reach.hand import HandSelector, HandTracker, OneEuro, forearm_hull
from reach.types import Frame


def points(x=200, y=250):
    p = np.array([(x+(i%4)*10, y-(i//4)*20) for i in range(21)], float)
    p[0] = (x, y+60)
    p[8] = (x, y-50)
    return p


def candidate(x, label, score=.99):
    return {"points": points(x), "label": label, "score": score}


def test_filter_reduces_jitter_and_tracks_motion():
    filt = OneEuro(beta=0)
    values = [filt.update((100+(-1)**i*2, 100), i/30)[0][0] for i in range(120)]
    assert np.std(values[30:]) < .4
    filt = OneEuro()
    for i in range(60):
        point, velocity = filt.update((i*5, 100), i/30)
    assert 280 < point[0] <= 295 and 140 < velocity[0] < 151
    filt.reset()
    assert filt.update((0, 0), 10) == ((0.0, 0.0), (0.0, 0.0))


def test_selection_votes_dominance_and_ten_frame_switch():
    selector = HandSelector(votes=1)
    first, _ = selector.choose([candidate(100, "Left"), candidate(300, "Right")], 1)
    assert first["label"] == "Right"
    original = first["id"]
    for i in range(1, 10):
        value, _ = selector.choose([candidate(100, "Right"), candidate(300, "Left")], 1+i/30)
        assert value["id"] == original
    value, switched = selector.choose([candidate(100, "Right"), candidate(300, "Left")], 1+10/30)
    assert switched and value["id"] != original


def test_approaching_hand_without_clear_dominance():
    selector = HandSelector(votes=1, switch_frames=1)
    selector.choose([candidate(100, "Left", .5), candidate(300, "Right", .5)], 1, np.array([200, 100]))
    value, _ = selector.choose([candidate(120, "Left", .5), candidate(330, "Right", .5)], 1.1, np.array([200, 100]))
    assert value["points"][0][0] == 120


def test_forearm_hull_extends_beyond_wrist():
    p = points()
    hull = np.asarray(forearm_hull(p))
    assert hull[:, 1].max() > p[0, 1]+100
    assert hull[:, 0].min() < p[:, 0].min()-10


def test_observe_timestamps_duplicates_hold_and_loss():
    p = points()/np.array([640, 480])
    result = NS(hand_landmarks=[[NS(x=x, y=y) for x, y in p]],
                handedness=[[NS(category_name="Right", score=.99)]])
    calls = []
    class Detector:
        def detect_for_video(self, image, stamp):
            calls.append(stamp)
            return result if len(calls) == 1 else NS(hand_landmarks=[], handedness=[])
    tracker = HandTracker({}, detector=Detector())
    image = np.zeros((480, 640, 3), np.uint8)
    frame = lambda i, t: Frame(i, t, 0, 640, 480, "hash", "file")
    first = tracker.observe(frame(0, 1), image)
    assert tracker.observe(frame(0, 1), image) is first and len(calls) == 1
    assert len(first.landmarks_px) == 21 and first.handedness == "Right"
    assert tracker.observe(frame(1, 1.0001), image).present and tracker.held
    assert calls == [1000, 1001]
    assert not tracker.observe(frame(2, 1.31), image).present
    tracker._upside.extend([np.bool_(True)]*30)
    import json
    assert type(tracker.upside_down) is bool
    json.dumps(tracker.stats(), allow_nan=False)


@pytest.mark.parametrize('tip', [(40, 40), (150, 150)])
def test_contact_events_serialize_on_initial_and_reacquired_hand(tip):
    from reach.brain import Brain
    from reach.types import Command, Instance, TargetObs
    landmarks = points().copy()
    landmarks[8] = tip
    normalized = landmarks/np.array([640, 480])
    detected = NS(hand_landmarks=[[NS(x=x, y=y) for x,y in normalized]],
                  handedness=[[NS(category_name='Right', score=.99)]])
    returned = normalized.copy()
    returned[8] = np.asarray((150, 150) if tip == (40, 40) else (40, 40))/[640, 480]
    reacquired = NS(hand_landmarks=[[NS(x=x, y=y) for x,y in returned]],
                    handedness=detected.handedness)
    absent = NS(hand_landmarks=[], handedness=[])
    class Detector:
        calls = 0
        def detect_for_video(self, image, stamp):
            self.calls += 1
            return absent if self.calls == 2 else reacquired if self.calls == 3 else detected
    tracker = HandTracker({'hand': {'switch_frames': 1}}, detector=Detector())
    brain = Brain()
    target = TargetObs(0, 'pen', (Instance('1', (100, 100, 200, 200), 1, None),),
                       0, 'tracker', 0)
    image = np.zeros((480, 640, 3), np.uint8)
    contact_events = []
    try:
        for i,stamp in enumerate((1, 1.4, 1.5)):
            hand = tracker.observe(Frame(i, stamp, 0, 640, 480, 'synthetic', 'file'), image)
            assert hand.present is (i != 1)
            commands = [Command('find', 'pen', 'synthetic', None, stamp)] if i == 0 else []
            decision, effects = brain.step(stamp, hand=hand, target=target, commands=commands)
            json.dumps(asdict(decision), allow_nan=False)
            for effect in effects:
                if effect.kind == 'log':
                    json.dumps(effect.payload, allow_nan=False)
                    if effect.key == 'contact':
                        contact_events.append(effect.payload)
            if hand.present:
                assert all(type(value) is float for value in hand.fingertip_px)
        assert len(contact_events) == 2
        assert all(not event['fired'] for event in contact_events)
    finally:
        tracker.close()
