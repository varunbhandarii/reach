"""Hand landmarks, hand selection and adaptive fingertip filtering."""

from collections import deque
from dataclasses import replace
import math
from pathlib import Path
import time

import cv2
import numpy as np

from .types import HandObs


class OneEuro:
    def __init__(self, min_cutoff=1.0, beta=.01, derivative_cutoff=1.0):
        self.min_cutoff, self.beta, self.derivative_cutoff = min_cutoff, beta, derivative_cutoff
        self.reset()

    def reset(self):
        self.value = self.raw = self.stamp = None
        self.velocity = np.zeros(2, float)

    @staticmethod
    def alpha(dt, cutoff):
        return 1 / (1 + 1/(2*math.pi*cutoff*dt))

    def update(self, point, stamp):
        raw = np.asarray(point, float)
        if self.value is None:
            self.value, self.raw, self.stamp = raw.copy(), raw.copy(), stamp
            return tuple(self.value), (0.0, 0.0)
        dt = max(1e-6, stamp-self.stamp)
        derivative = (raw-self.raw)/dt
        a = self.alpha(dt, self.derivative_cutoff)
        self.velocity += a*(derivative-self.velocity)
        cutoff = self.min_cutoff + self.beta*np.abs(self.velocity)
        self.value += self.alpha(dt, cutoff)*(raw-self.value)
        self.raw, self.stamp = raw.copy(), stamp
        return tuple(map(float, self.value)), tuple(map(float, self.velocity))


def forearm_hull(points, dilation=15):
    points = np.asarray(points, float)
    wrist = points[0]
    knuckles = points[[5, 9, 13, 17]].mean(axis=0)
    backward = wrist-knuckles
    length = max(1.0, float(np.linalg.norm(backward)))
    perpendicular = np.array([-backward[1], backward[0]])/length
    half_width = max(8, float(np.linalg.norm(points[5]-points[17]))/2)
    end = wrist+3*backward
    all_points = np.vstack([points, end+half_width*perpendicular, end-half_width*perpendicular])
    angles = np.arange(16)*2*np.pi/16
    offsets = dilation*np.column_stack([np.cos(angles), np.sin(angles)])
    expanded = (all_points[:, None, :] + offsets).reshape(-1, 2).astype(np.float32)
    return tuple(tuple(map(float, p)) for p in cv2.convexHull(expanded).reshape(-1, 2))


class HandSelector:
    """Match hands by wrist proximity, vote labels, and require sustained switches."""

    def __init__(self, dominant="Right", votes=15, switch_frames=10):
        self.dominant, self.votes, self.switch_frames = dominant.title(), votes, switch_frames
        self.tracks = {}
        self.selected = self.pending = None
        self.pending_count = self.next_id = 0

    def choose(self, candidates, stamp, target=None):
        available = {key for key, track in self.tracks.items() if stamp-track["time"] <= .3}
        assigned = []
        for item in candidates:
            wrist = np.asarray(item["points"][0])
            key = min(available, key=lambda k: np.linalg.norm(self.tracks[k]["wrist"]-wrist), default=None)
            if key is not None and np.linalg.norm(self.tracks[key]["wrist"]-wrist) > 180:
                key = None
            if key is None:
                key, self.next_id = self.next_id, self.next_id+1
                self.tracks[key] = {"votes": deque(maxlen=self.votes), "tip": None, "time": stamp}
            available.discard(key)
            track = self.tracks[key]
            score = item["score"]
            probability = score if item["label"] == self.dominant else 1-score
            track["votes"].append((item["label"], probability))
            tip = np.asarray(item["points"][8])
            dt = max(.001, stamp-track["time"])
            approach = ((np.linalg.norm(track["tip"]-target)-np.linalg.norm(tip-target))/dt
                        if target is not None and track["tip"] is not None else 0)
            track.update(wrist=wrist, tip=tip, time=stamp)
            votes = track["votes"]
            label = max(("Left", "Right"), key=lambda value: sum(v[0] == value for v in votes))
            assigned.append({**item, "id": key, "prob": sum(v[1] for v in votes)/len(votes),
                             "handedness": label, "approach": approach})
        self.tracks = {key: value for key, value in self.tracks.items() if stamp-value["time"] <= .3}
        if not assigned:
            self.pending = None
            self.pending_count = 0
            return None, False
        dominant = [c for c in assigned if c["prob"] > .7 and all(o["prob"] < .3 for o in assigned if o is not c)]
        preferred = dominant[0] if dominant else max(assigned, key=lambda c: (c["approach"], c["id"] == self.selected))
        current = next((c for c in assigned if c["id"] == self.selected), None)
        previous = self.selected
        if self.selected is None:
            self.selected = preferred["id"]
        elif preferred["id"] != self.selected:
            if self.pending == preferred["id"]:
                self.pending_count += 1
            else:
                self.pending, self.pending_count = preferred["id"], 1
            if self.pending_count >= self.switch_frames:
                self.selected = preferred["id"]
                self.pending = None
                self.pending_count = 0
        else:
            self.pending = None
            self.pending_count = 0
        chosen = next((c for c in assigned if c["id"] == self.selected), None)
        return chosen, previous != self.selected


class HandTracker:
    def __init__(self, cfg, *, detector=None):
        self.cfg = cfg.get("hand", {})
        self._detector = detector
        self._mp = None
        self._last_id = self._last_timestamp = -1
        self._last_observation = None
        self._last_seen = None
        self._selector = HandSelector(self.cfg.get("dominant", "Right"),
            int(self.cfg.get("label_vote_frames", 15)), int(self.cfg.get("switch_frames", 10)))
        self._filter = OneEuro(float(self.cfg.get("min_cutoff", 1)), float(self.cfg.get("beta", .01)),
                               float(self.cfg.get("derivative_cutoff", 1)))
        self._costs = deque(maxlen=18000)
        self.last_detect_ms = None
        self.fingertip_raw_px = None
        self.held = False
        self._upside = deque(maxlen=30)

    def _open(self):
        if self._detector is not None:
            return
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        path = Path(self.cfg.get("model_path", "models/hand_landmarker.task"))
        if not path.is_file():
            raise ValueError("Hand model missing; set hand.model_path to hand_landmarker.task")
        confidence = float(self.cfg.get("confidence", .5))
        self._mp = mp
        self._detector = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=confidence, min_hand_presence_confidence=confidence,
            min_tracking_confidence=confidence))

    def observe(self, frame, image, target=None):
        if frame.id <= self._last_id:
            if frame.id == self._last_id:
                return self._last_observation
            raise ValueError("Hand frames must be monotonically increasing")
        self._open()
        width = int(self.cfg.get("input_width", frame.width))
        work = image
        if 0 < width < frame.width:
            work = cv2.resize(image, (width, round(width*frame.height/frame.width)))
        rgb = np.ascontiguousarray(cv2.cvtColor(work, cv2.COLOR_BGR2RGB))
        value = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb) if self._mp else rgb
        timestamp = max(int(frame.t_mono*1000), self._last_timestamp+1)
        start = time.perf_counter()
        result = self._detector.detect_for_video(value, timestamp)
        self.last_detect_ms = (time.perf_counter()-start)*1000
        self._costs.append(self.last_detect_ms)
        self._last_timestamp, self._last_id = timestamp, frame.id
        candidates = []
        for i, landmarks in enumerate(result.hand_landmarks):
            points = tuple((float(p.x*frame.width), float(p.y*frame.height)) for p in landmarks)
            if len(points) != 21 or not np.isfinite(points).all():
                continue
            category = result.handedness[i][0]
            candidates.append({"points": points, "label": category.category_name.title(),
                               "score": float(category.score)})
        chosen, switched = self._selector.choose(candidates, frame.t_mono, target)
        self.held = False
        if chosen is None:
            if self._last_seen is not None and frame.t_mono-self._last_seen <= float(self.cfg.get("hold_ms", 300))/1000:
                self.held = True
                observation = replace(self._last_observation, frame_id=frame.id, velocity_px_s=(0.0, 0.0))
            else:
                self._filter.reset()
                self.fingertip_raw_px = None
                observation = HandObs(frame.id, False, None, None, (), (0.0, 0.0), 0)
        else:
            if switched or self._last_seen is None or frame.t_mono-self._last_seen > .3:
                self._filter.reset()
            points = chosen["points"]
            self.fingertip_raw_px = points[8]
            tip, velocity = self._filter.update(points[8], frame.t_mono)
            palm = tuple(map(float, np.mean(np.asarray(points)[[0, 5, 9, 13, 17]], axis=0)))
            observation = HandObs(frame.id, True, tip, palm, forearm_hull(points), velocity,
                                  chosen["score"], points, chosen["handedness"])
            self._last_seen = frame.t_mono
            self._upside.append(points[0][1] < np.mean([points[i][1] for i in (5, 9, 13, 17)]))
        self._last_observation = observation
        return observation

    def stats(self):
        values = tuple(self._costs)
        return {"hand_ms_p50": float(np.median(values)) if values else None,
                "hand_ms_p95": float(np.percentile(values, 95)) if values else None,
                "processed": len(values), "input_width": self.cfg.get("input_width"),
                "upside_down": len(self._upside) >= 15 and sum(self._upside)/len(self._upside) > .7}

    def close(self):
        if self._detector is not None and hasattr(self._detector, "close"):
            self._detector.close()
