"""Shared immutable messages. Coordinates are source-image pixels.

Event envelope stays seq/t_mono/t_wall/trial/type/data. New payloads include:
command (kind, phrase, transcript, turn_id, source, untested_noun), state
(from, to, reason, phrase), contact (event, inside, still, overlap, rel_speed,
hold_frames, hand_frame_id, target_frame_id), voice_session/voice_turn and
speech request/start/end/cancel/drop. Hand payloads add landmarks_px; trial_end
adds contact_fired_s while preserving manual outcome provenance. Read-back
responses carry request identity; brain_in/brain_out retain deterministic
inputs/outputs. readback_req/readback_resp and record keep API/file timing.
Log effects use key=event type and payload=flat event data.
"""

from dataclasses import dataclass
from typing import Literal

CONTRACT_VERSION = "5"
Source = Literal["android_mjpeg", "iphone_continuity", "usb_webcam", "file"]
CommandKind = Literal[
    "find", "choose", "cancel", "start_over", "repeat", "recheck", "yes", "no", "stop", "unknown", "describe"
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
    landmarks_px: tuple[tuple[float, float], ...] = ()
    handedness: str | None = None


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
    source: Literal["voice", "voice_fast", "ptt", "key", "trial", "spark_parse"] = "voice"
    language: str | None = None
    spoken_phrase: str | None = None


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
    hand_frame_id: int | None = None
    target_frame_id: int | None = None


@dataclass(frozen=True)
class Effect:
    kind: Literal["speak", "cancel_speech", "earcon", "set_phrase", "force_query", "perception_state", "log", "readback", "choose_instance", "exclude_instance", "describe"]
    text: str | None = None
    key: str | None = None
    priority: int = 0
    payload: dict | None = None


@dataclass(frozen=True)
class ReadBack:
    request_id: int
    target_frame_id: int
    phrase: str
    matches: bool | None
    object_name: str | None
    hint: str | None
    confidence: float | None
    latency_ms: float | None
    error: str | None = None
    same_object: bool | None = None
    mode: Literal["pair", "single"] | None = None


@dataclass(frozen=True)
class SceneList:
    request_id: int
    frame_id: int
    objects: tuple[tuple[str,str], ...]
    latency_ms: float | None
    error: str | None = None
