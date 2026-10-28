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


def test_simulated_observation_flow_and_buffer(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(time, "perf_counter", lambda: now[0])
    cfg = {"camera": {"source": "file", "width": 640, "height": 480}}
    camera = Camera(cfg)
    with pytest.raises(RuntimeError):
        camera.latest()
    camera.start()
    frame, image = camera.latest()
    hand = HandTracker(cfg).observe(frame, image)
    assert frame.id == 0 and hand.frame_id == frame.id
    assert hand.present and image.shape == (480, 640, 3)
    assert len(frame.jpeg_sha256) == 64
    image[:] = 0
    assert camera.get(frame.id).mean() == 96
    now[0] += 3.01
    assert camera.get(frame.id) is None
    second, _ = camera.latest()
    assert second.id > frame.id
    camera.stop()
    assert camera.get(second.id) is None


def test_output_stubs_do_not_trigger_io():
    cue = CueParams(0, 440, 4, False, 0, .2)
    audio = CueEngine({})
    audio.start()
    audio.set(cue)
    audio.duck(True)
    audio.earcon("contact")
    assert audio.cue is cue and audio.ducked and audio.last_earcon == "contact"
    audio.set(None)
    assert audio.cue is None
    audio.stop()
    speaker = Speaker({})
    speaker.say("Found the mug")
    assert not speaker.busy()
    speaker.stop()
    commands = []
    voice = VoiceClient({}, commands.append)
    voice.start()
    assert voice.running and commands == []
    voice.on_command(Command("find", "mug", "find mug", None, 10.0))
    assert commands[0].phrase == "mug"
    voice.stop()


def test_local_configuration_merges_nested_defaults(tmp_path):
    shared = tmp_path / "config.yaml"
    shared.write_text("camera:\n  source: usb_webcam\n  width: 640\n", encoding="utf-8")
    shared.with_name("config.local.yaml").write_text("camera:\n  device_index: 2\n", encoding="utf-8")
    assert load_config(shared)["camera"] == {"source": "usb_webcam", "width": 640, "device_index": 2}
