"""Deterministic target tracker placeholder for component integration."""

from typing import Any

from .types import Frame, Instance


class TargetTracker:
    """Hold the most recently supplied detection until a tracker is connected."""

    def __init__(self, cfg: Any) -> None:
        del cfg
        self._instance: Instance | None = None

    def reseed(self, frame: Frame, image: Any, instance: Instance) -> None:
        del frame, image
        self._instance = instance

    def update(self, frame: Frame, image: Any) -> Instance | None:
        del frame, image
        return self._instance
