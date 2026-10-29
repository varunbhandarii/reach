"""Perception interface for camera, controller and overlay integration."""

from .types import TargetObs


class Perception:
    def __init__(self, cfg, camera, log):
        self.cfg, self.camera, self.log = cfg, camera, log
        self._target = None

    def start(self):
        pass

    def stop(self):
        pass

    def set_phrase(self, phrase: str | None):
        self._target = None

    def select_point(self, x: float, y: float):
        pass

    def set_state(self, state: str):
        pass

    def on_hand(self, hand):
        pass

    def latest(self) -> TargetObs | None:
        return self._target

    def stats(self) -> dict:
        return {}
