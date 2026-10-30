"""Conservative rejection of visibly changed, unobscured late target patches."""

import cv2
import numpy as np

from .masks import decode_rle
from .tracker import hand_mask


def _patch(image, instance):
    x0, y0, x1, y1 = instance.box_xyxy
    patch = image[y0:y1, x0:x1]
    mask = (decode_rle(instance.mask_rle) if instance.mask_rle is not None
            else np.ones((y1-y0, x1-x0), np.uint8))
    if patch.shape[:2] != mask.shape or not mask.any():
        return None
    return patch, (mask != 0).astype(np.uint8)


def compare_appearance(source_image, source_instance, current_image, mapped_instance, hand=None):
    """Return distances, or None if hand occlusion/small masks prevent a check.

    This checks only gross appearance changes. It does not establish identity,
    and deliberately preserves propagation while the hand covers the target.
    """
    source = _patch(source_image, source_instance)
    current = _patch(current_image, mapped_instance)
    if source is None or current is None:
        return None
    src, src_mask = source
    dst, dst_mask = current
    if min(cv2.countNonZero(src_mask), cv2.countNonZero(dst_mask)) < 32:
        return None
    if hand is not None and hand.present:
        if len(hand.hull_px) < 3:
            return None  # Missing occlusion geometry is not evidence of absence.
        mask = hand_mask(current_image.shape[:2], hand)
        x0, y0, x1, y1 = mapped_instance.box_xyxy
        overlap = np.count_nonzero((mask[y0:y1, x0:x1] != 0) & (dst_mask != 0))
        if overlap / cv2.countNonZero(dst_mask) >= .15:
            return None

    def histogram(patch, mask, channels, bins, ranges):
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        value = cv2.calcHist([hsv], channels, mask, bins, ranges)
        return cv2.normalize(value, None, alpha=1, norm_type=cv2.NORM_L1)

    distances = {}
    for name, channels, bins, ranges in (
        ("hs", [0, 1], [12, 8], [0, 180, 0, 256]),
        ("hsv", [0, 1, 2], [8, 8, 8], [0, 180, 0, 256, 0, 256]),
    ):
        a = histogram(src, src_mask, channels, bins, ranges)
        b = histogram(dst, dst_mask, channels, bins, ranges)
        distances[name] = float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))
    return distances


def appearance_changed(distances, threshold=.8):
    return distances is not None and distances['hs'] > threshold and distances['hsv'] > threshold
