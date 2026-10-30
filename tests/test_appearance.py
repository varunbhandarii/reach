import cv2
import numpy as np
from reach.appearance import compare_appearance, appearance_changed
from reach.masks import encode_rle
from reach.types import Instance, HandObs


def scene():
    image = np.full((180, 240, 3), (160, 70, 20), np.uint8)
    cv2.circle(image, (70, 90), 20, (245, 245, 245), -1)
    mask = np.zeros((40, 40), np.uint8)
    cv2.circle(mask, (20, 20), 18, 1, -1)
    return image, Instance('ball', (50, 70, 90, 110), .9, encode_rle(mask))


def test_independently_moved_object_is_rejected():
    source, box = scene()
    current = np.full_like(source, (160, 70, 20))
    cv2.circle(current, (180, 90), 20, (245, 245, 245), -1)
    assert appearance_changed(compare_appearance(source, box, current, box))


def test_preserved_object_and_mask_are_accepted():
    image, box = scene()
    assert not appearance_changed(compare_appearance(image, box, image.copy(), box))


def test_hand_occlusion_is_not_treated_as_object_disappearance():
    source, box = scene()
    occluded = source.copy()
    occluded[70:110, 50:90] = (100, 150, 200)
    hand = HandObs(1, True, None, None, ((45, 65), (95, 65), (95, 115), (45, 115)), (0, 0), 1)
    assert compare_appearance(source, box, occluded, box, hand) is None
