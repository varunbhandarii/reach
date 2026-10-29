"""Shared immutable messages. Coordinates are source-image pixels."""

from dataclasses import dataclass
from typing import Literal

CONTRACT_VERSION = "1"
Source = Literal["android_mjpeg", "iphone_continuity", "usb_webcam", "file"]
CommandKind = Literal[
    "find", "choose", "cancel", "start_over", "repeat", "recheck", "yes", "no", "stop", "unknown"
]


@dataclass(frozen=True)
class Frame:
    id: int
    t_mono: float
    t_wall: float
    width: int
    height: int
    jpeg_sha256: str
    source: Source


@dataclass(frozen=True)
class HandObs:
    frame_id: int
    present: bool
    fingertip_px: tuple[float, float] | None
    palm_px: tuple[float, float] | None
    hull_px: tuple[tuple[float, float], ...]
    velocity_px_s: tuple[float, float]
    score: float


@dataclass(frozen=True)
class Instance:
    sam_id: str
    box_xyxy: tuple[int, int, int, int]
    conf: float | None
    mask_rle: dict | None


@dataclass(frozen=True)
class TargetObs:
    frame_id: int
    phrase: str
    instances: tuple[Instance, ...]
    chosen: int | None
    source: Literal["sam", "tracker", "none"]
    age_ms: float


@dataclass(frozen=True)
class Command:
    kind: CommandKind
    phrase: str | None
    transcript: str
    turn_id: int | None
    t_mono: float


@dataclass(frozen=True)
class CueParams:
    pan: float
    freq_hz: float
    pulse_hz: float
    continuous: bool
    noise: float
    gain: float


@dataclass(frozen=True)
class Decision:
    seq: int
    t_mono: float
    state: str
    dx: float | None
    dy: float | None
    d: float | None
    cue: CueParams | None
    speech: str | None
    earcon: str | None
