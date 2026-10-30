"""Silent speech stub that stores the most recent utterance."""


class Speaker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.last_text: str | None = None

    def say(self, text: str) -> None:
        self.last_text = text

    def busy(self) -> bool:
        return False

    def stop(self) -> None:
        self.last_text = None

# Engine interface for the speech-stream worker.
from .say_engine import SayEngine
