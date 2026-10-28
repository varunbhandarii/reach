"""Synthetic camera stub for integration without hardware."""

from collections import deque
from hashlib import sha256
import time

import cv2
import numpy as np

from .types import Frame


class Camera:
    """Produces a fixed image on demand; does not open a camera."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._running = False
        self._next_id = 0
        self._frames = deque()

    def start(self) -> None:
        self._running = True

    def latest(self) -> tuple[Frame, np.ndarray]:
        if not self._running:
            raise RuntimeError("Camera.start() must be called before latest()")
        camera = self.cfg.get("camera", {})
        width, height = int(camera.get("width", 640)), int(camera.get("height", 480))
        image = np.full((height, width, 3), 96, dtype=np.uint8)
        ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            raise RuntimeError("Could not encode the synthetic frame")
        now = time.perf_counter()
        frame = Frame(self._next_id, now, time.time(), width, height,
                      sha256(jpeg.tobytes()).hexdigest(), camera.get("source", "file"))
        self._next_id += 1
        self._frames.append((frame, image))
        self._prune(now)
        return frame, image.copy()

    def _prune(self, now: float) -> None:
        while self._frames and now - self._frames[0][0].t_mono > 3.0:
            self._frames.popleft()

    def get(self, frame_id: int) -> np.ndarray | None:
        self._prune(time.perf_counter())
        return next((image.copy() for frame, image in self._frames if frame.id == frame_id), None)

    def stop(self) -> None:
        self._running = False
        self._frames.clear()
