"""Silent audio stub that retains requested parameters for integration."""

from .types import CueParams


class CueEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.running = False
        self.cue: CueParams | None = None
        self.ducked = False
        self.last_earcon: str | None = None

    def start(self) -> None:
        self.running = True

    def set(self, cue: CueParams | None) -> None:
        self.cue = cue

    def earcon(self, name: str) -> None:
        if name not in {"found", "lost", "reacquired", "contact", "wrong"}:
            raise ValueError(f"Unknown earcon: {name}")
        self.last_earcon = name

    def duck(self, on: bool) -> None:
        self.ducked = on

    def stop(self) -> None:
        self.running = False
        self.cue = None
