"""Inactive voice stub. Does not open a microphone or network connection."""

from collections.abc import Callable

from .types import Command


class VoiceClient:
    def __init__(self, cfg, on_command: Callable[[Command], None]):
        self.cfg = cfg
        self.on_command = on_command
        self.running = False

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False
