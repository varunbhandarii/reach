"""Fixed hand observation stub for integration without a model."""

from .types import Frame, HandObs


class HandTracker:
    def __init__(self, cfg):
        self.cfg = cfg

    def observe(self, frame: Frame, image) -> HandObs:
        w, h = frame.width, frame.height
        return HandObs(frame.id, True, (w * 0.5, h * 0.5), (w * 0.5, h * 0.7),
                       ((w * 0.4, h * 0.5), (w * 0.6, h * 0.5),
                        (w * 0.6, h * 0.8), (w * 0.4, h * 0.8)), (0.0, 0.0), 1.0)
