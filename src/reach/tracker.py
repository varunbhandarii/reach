"""Hand-masked table motion and bounded transform history for late detections."""

from collections import OrderedDict, deque
from dataclasses import replace
import time

import cv2
import numpy as np

from .masks import decode_rle, encode_rle


def box_iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    overlap = max(0, x1-x0)*max(0, y1-y0)
    return overlap / max(1, (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-overlap)


def hand_mask(shape, hand):
    mask = np.zeros(shape, np.uint8)
    if hand is not None and hand.present and len(hand.hull_px) >= 3:
        # The contract hull already includes the dilated forearm.
        points = np.asarray(hand.hull_px, np.float32)
        if np.isfinite(points).all():
            cv2.fillConvexPoly(mask, cv2.convexHull(points.astype(np.int32)), 255)
    return mask


def transform_instance(instance, matrix, width, height):
    x0, y0, x1, y1 = instance.box_xyxy
    points = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    mapped = cv2.perspectiveTransform(points[None], matrix)[0]
    if not np.isfinite(mapped).all():
        return None
    left, top = np.maximum(np.floor(mapped.min(0)), [0, 0]).astype(int)
    right, bottom = np.minimum(np.ceil(mapped.max(0)), [width, height]).astype(int)
    if right <= left or bottom <= top:
        return None
    rle = None
    if instance.mask_rle is not None:
        # Local raster -> source coordinates -> current image -> new local raster.
        into_image = np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]], float)
        into_local = np.array([[1, 0, -left], [0, 1, -top], [0, 0, 1]], float)
        mask = cv2.warpPerspective(decode_rle(instance.mask_rle), into_local @ matrix @ into_image,
                                   (right-left, bottom-top), flags=cv2.INTER_NEAREST)
        rle = encode_rle(mask)
    return replace(instance, box_xyxy=(int(left), int(top), int(right), int(bottom)), mask_rle=rle)


class TargetTracker:
    def __init__(self, cfg=None):
        self.cfg = (cfg or {}).get("motion", {})
        self.depth = int(self.cfg.get("history", 90))
        self.min_inliers = int(self.cfg.get("min_inliers", 15))
        self.method = self.cfg.get("method", "homography")
        if self.method not in ("homography", "kcf"):
            raise ValueError("motion.method must be homography or kcf")
        self.history = OrderedDict()
        self.frame = self.gray = self.mask = self.image = None
        self._seed = self._seed_id = self._kcf = None
        self.stale = True
        self.roll_total = 0.0
        self._costs = deque(maxlen=600)
        self.motion = {"inliers": 0, "scale": 1.0, "rotation_deg": 0.0, "ms": 0.0}

    def _estimate(self, current, current_mask):
        height, width = current.shape
        max_width = int(self.cfg.get("width", 640))
        factor = min(1.0, max_width / width)
        size = (round(width*factor), round(height*factor))
        a, b = cv2.resize(self.gray, size), cv2.resize(current, size)
        old_mask = cv2.resize(self.mask, size, interpolation=cv2.INTER_NEAREST)
        new_mask = cv2.resize(current_mask, size, interpolation=cv2.INTER_NEAREST)
        allowed = cv2.bitwise_not(old_mask)
        allowed[:size[1]//3] = 0
        p0 = cv2.goodFeaturesToTrack(a, 300, .01, 4, mask=allowed, blockSize=5)
        if p0 is None or len(p0) < self.min_inliers:
            return np.eye(3), 0, 1.0, 0.0
        options = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 20, .03))
        p1, ok, _ = cv2.calcOpticalFlowPyrLK(a, b, p0, None, **options)
        if p1 is None:
            return np.eye(3), 0, 1.0, 0.0
        back, ok2, _ = cv2.calcOpticalFlowPyrLK(b, a, p1, None, **options)
        if back is None:
            return np.eye(3), 0, 1.0, 0.0
        q = p1.reshape(-1, 2)
        finite = np.isfinite(q).all(1)
        indices = np.clip(np.nan_to_num(q).astype(int), 0, [size[0]-1, size[1]-1])
        good = finite & (ok.ravel() == 1) & (ok2.ravel() == 1) & (np.linalg.norm((p0-back).reshape(-1, 2), axis=1) < 1)
        good &= new_mask[indices[:, 1], indices[:, 0]] == 0
        src, dst = p0.reshape(-1, 2)[good], q[good]
        if len(src) < self.min_inliers:
            return np.eye(3), 0, 1.0, 0.0
        matrix, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 2*factor)
        similarity, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2*factor)
        if matrix is None or not np.isfinite(matrix).all() or abs(matrix[2, 2]) < 1e-9:
            return np.eye(3), 0, 1.0, 0.0
        matrix /= matrix[2, 2]
        scaling = np.diag([factor, factor, 1.0])
        matrix = np.linalg.inv(scaling) @ matrix @ scaling
        determinant = np.linalg.det(matrix[:2, :2])
        scale = float(np.sqrt(max(0, determinant)))
        rotation = float(np.degrees(np.arctan2(similarity[1, 0], similarity[0, 0]))) if similarity is not None else 0.0
        return matrix, int(inliers.sum()), scale, rotation

    def update(self, frame, image, hand=None):
        if self.frame is not None and frame.id <= self.frame.id:
            return self.current(), self.stale
        start = time.perf_counter()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = hand_mask(gray.shape, hand)
        previous = None if self.frame is None else self.frame.id
        valid = False
        matrix, inliers, scale, rotation = np.eye(3), 0, 1.0, 0.0
        if self.gray is not None and self.gray.shape == gray.shape:
            matrix, inliers, scale, rotation = self._estimate(gray, mask)
            valid = inliers >= self.min_inliers and .8 <= scale <= 1.25
        self.history[frame.id] = (previous, matrix if valid else np.eye(3), valid)
        while len(self.history) > self.depth:
            self.history.popitem(last=False)
        self.frame, self.gray, self.mask, self.image = frame, gray, mask, image
        if valid:
            self.roll_total += rotation
        self.motion = {"frame_id": frame.id, "inliers": inliers, "scale": scale,
                       "rotation_deg": rotation, "roll_total_deg": self.roll_total,
                       "ms": (time.perf_counter()-start)*1000}
        self._costs.append(self.motion["ms"])
        if self.method == "kcf" and self._kcf is not None:
            ok, box = self._kcf.update(image)
            if ok:
                x, y, w, h = box
                self._seed = replace(self._seed, box_xyxy=(int(x), int(y), int(x+w), int(y+h)), mask_rle=None)
                self._seed_id = frame.id
            else:
                self.stale = True
        return self.current(), self.stale

    def compose(self, frame_id):
        if self.frame is None or frame_id not in self.history:
            return None
        matrix, cursor = np.eye(3), self.frame.id
        # Walk backwards: older transforms multiply on the right.
        while cursor != frame_id:
            if cursor not in self.history:
                return None
            previous, transform, valid = self.history[cursor]
            if not valid or previous is None:
                return None
            matrix = matrix @ transform
            cursor = previous
        return matrix

    def map_instance(self, frame_id, instance):
        matrix = self.compose(frame_id)
        return None if matrix is None else transform_instance(instance, matrix, self.frame.width, self.frame.height)

    def reseed(self, frame_id, instance):
        mapped = self.map_instance(frame_id, instance)
        if mapped is None:
            return False
        self._seed, self._seed_id = instance, frame_id
        self.stale = False
        if self.method == "kcf":
            factory = getattr(cv2, "TrackerKCF_create", None) or cv2.legacy.TrackerKCF_create
            self._kcf = factory()
            x0, y0, x1, y1 = mapped.box_xyxy
            self._kcf.init(self.image, (x0, y0, x1-x0, y1-y0))
            self._seed, self._seed_id = mapped, self.frame.id
        return True

    def clear(self):
        self._seed = self._seed_id = self._kcf = None
        self.stale = True

    def current(self):
        if self._seed is None:
            return None
        mapped = self.map_instance(self._seed_id, self._seed)
        if mapped is None:
            self.stale = True
            return None
        return mapped

    def stats(self):
        values = np.asarray(self._costs)
        return {**self.motion, "stale": self.stale, "roll_warning": abs(self.roll_total) > 20,
                "motion_ms_p50": float(np.median(values)) if len(values) else None,
                "motion_ms_p95": float(np.percentile(values, 95)) if len(values) else None}
