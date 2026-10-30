"""Pure fingertip-to-target mapping and bounded motion prediction."""

import math
import time

import numpy as np

from .types import CueParams, Decision


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def target_center(target):
    if target is None or target.chosen is None or not 0 <= target.chosen < len(target.instances):
        return None
    x0, y0, x1, y1 = target.instances[target.chosen].box_xyxy
    if x1 <= x0 or y1 <= y0:
        return None
    return ((x0+x1)/2, (y0+y1)/2)


def predict_tip(point, velocity, target, k=.5, horizon=.14):
    point, velocity, target = np.asarray(point, float), np.asarray(velocity, float), np.asarray(target, float)
    speed = float(np.linalg.norm(velocity))
    if speed < 50 or k == 0:
        return tuple(map(float, point))
    delta = velocity*k*horizon
    limit = min(40, .25*float(np.linalg.norm(target-point)))
    magnitude = float(np.linalg.norm(delta))
    if magnitude > limit:
        delta *= limit/magnitude
    return tuple(map(float, point+delta))


def map_vector(point, target, width, height, *, flipped=False, gain=.2):
    if width <= 0 or height <= 0 or not all(math.isfinite(x) for x in (*point, *target)):
        raise ValueError("Coordinates must be finite and dimensions positive")
    dx, dy = (target[0]-point[0])/width, (target[1]-point[1])/height
    if flipped:
        dx, dy = -dx, -dy
    distance = math.hypot(dx, dy)
    pan = 0.0 if abs(dx) < .03 else clamp(dx/.4, -1, 1)
    frequency = 440*2**clamp(-dy/.3, -1, 1)
    pulse = 2*6**(1-clamp(distance/.5, 0, 1))
    noise = 10**(-18/20) if abs(dy) < .04 else 0.0
    return dx, dy, distance, CueParams(pan, frequency, pulse, False, noise, gain)


def decide(hand, target, frame, *, seq=0, prediction=.5, flipped=False, gain=.2, now=None):
    now = time.perf_counter() if now is None else now
    center = target_center(target)
    ids = {"hand_frame_id": hand.frame_id if hand else None,
           "target_frame_id": target.frame_id if target else None}
    if hand is None or not hand.present or hand.fingertip_px is None or center is None:
        state = "HAND_LOST" if center is not None else "IDLE" if target is None else "SEARCHING"
        return Decision(seq, now, state, None, None, None, None, None, None, **ids)
    point = predict_tip(hand.fingertip_px, hand.velocity_px_s, center, prediction)
    dx, dy, distance, cue = map_vector(point, center, frame.width, frame.height, flipped=flipped, gain=gain)
    return Decision(seq, now, "GUIDING", dx, dy, distance, cue, None, None, **ids)
