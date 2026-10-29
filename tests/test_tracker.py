from dataclasses import replace
import cv2
import numpy as np
import pytest
from reach.tracker import TargetTracker
from reach.types import Frame, Instance, HandObs


def frame(i):
    return Frame(i, i/30, 0, 640, 480, str(i), 'file')


def texture():
    rng = np.random.default_rng(5)
    image = rng.integers(0, 256, (480, 640, 3), np.uint8)
    return cv2.GaussianBlur(image, (5, 5), 0)


@pytest.mark.parametrize('angle,scale,dx', [(0, 1, 9), (2, 1, 0), (0, 1.03, 0)])
def test_motion_and_late_seed(angle, scale, dx):
    image = texture()
    affine = cv2.getRotationMatrix2D((320, 240), angle, scale)
    affine[0, 2] += dx
    warped = cv2.warpAffine(image, affine, (640, 480))
    t = TargetTracker({'motion': {'width': 640}})
    t.update(frame(0), image)
    t.update(frame(1), warped)
    instance = Instance('1', (200, 200, 270, 260), .9, None)
    assert t.reseed(0, instance)
    result = t.current()
    expected = cv2.transform(np.float32([[[200, 200], [270, 200], [270, 260], [200, 260]]]), affine)[0]
    assert np.allclose(result.box_xyxy, [*expected.min(0), *expected.max(0)], atol=2)
    assert not t.stale


def test_hand_patch_is_excluded_and_lost_motion_cannot_be_used():
    image = texture()
    affine = np.float32([[1, 0, 6], [0, 1, 3]])
    moved = cv2.warpAffine(image, affine, (640, 480))
    moved[220:400, 200:400] = 0
    hand = HandObs(1, True, None, None, ((190, 210), (410, 210), (410, 410), (190, 410)), (0, 0), 1)
    t = TargetTracker()
    t.update(frame(0), image, hand)
    t.update(frame(1), moved, hand)
    assert t.reseed(0, Instance('1', (220, 230, 270, 280), .9, None))
    assert np.allclose(t.current().box_xyxy, (226, 233, 276, 283), atol=2)
    t.update(frame(2), np.zeros_like(image))
    assert t.current() is None and t.stale
    assert not t.reseed(0, Instance('1', (200, 200, 250, 250), .9, None))


def test_history_eviction_and_duplicate_frame():
    t = TargetTracker({'motion': {'history': 2}})
    image = texture()
    for i in range(3):
        t.update(frame(i), image)
    assert t.compose(0) is None
    assert len(t.history) == 2
    t.update(frame(2), image)
    assert len(t.history) == 2


def test_clicked_target_outlives_transform_ring():
    t = TargetTracker({'motion': {'history': 2}})
    image = texture()
    t.update(frame(0), image)
    assert t.reseed(0, Instance('click', (200, 200, 250, 250), 1, None))
    for i in range(1, 5):
        t.update(frame(i), image)
    assert t.compose(0) is None
    assert t.current() is not None and not t.stale
    assert np.allclose(t.current().box_xyxy, (200, 200, 250, 250), atol=2)
