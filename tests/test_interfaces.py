"""Exercise the message flow without camera, audio or network access."""

import time

import pytest

from reach.audio import CueEngine
from reach.capture import Camera
from reach.config import load_config
from reach.hand import HandTracker
from reach.tts import Speaker
from reach.types import Command, CueParams
from reach.voice import VoiceClient


def test_output_stubs_do_not_trigger_io():
    cue = CueParams(0, 440, 4, False, 0, .2)
    audio = CueEngine({})
    audio.set(cue)
    audio.duck(True)
    audio.earcon("contact")
    assert audio.cue is cue and audio.ducked and audio.last_earcon == "contact"
    audio.set(None)
    assert audio.cue is None
    audio.stop()


def test_local_configuration_merges_nested_defaults(tmp_path):
    shared = tmp_path / "config.yaml"
    shared.write_text("camera:\n  source: usb_webcam\n  width: 640\n", encoding="utf-8")
    shared.with_name("config.local.yaml").write_text("camera:\n  device_index: 2\n", encoding="utf-8")
    assert load_config(shared)["camera"] == {"source": "usb_webcam", "width": 640, "device_index": 2}
