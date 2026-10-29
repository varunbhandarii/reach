import types

import numpy as np
import pytest

from reach.audio import CueEngine, Tone, EARCONS, select_output
from reach.types import CueParams


def render(tone, cue, seconds=.5, **kwargs):
    blocks = []
    for _ in range(round(seconds*tone.fs/tone.blocksize)):
        out = np.empty((tone.blocksize, 2), np.float32)
        tone.render(out, cue, **kwargs)
        blocks.append(out)
    return np.concatenate(blocks)


def test_phase_continuity_pan_and_log_pitch():
    tone = Tone()
    cue = CueParams(1, 880, 4, True, 0, .15)
    signal = render(tone, cue)
    assert np.max(np.abs(signal[-5000:, 0])) < 1e-4
    wave = signal[-10000:, 1]
    freq = np.fft.rfftfreq(len(wave), 1/48000)[np.argmax(abs(np.fft.rfft(wave)))]
    assert abs(freq-880) < 6
    assert np.max(np.abs(np.diff(signal[-5000:, 1]))) < .02
    muted = render(tone, None)
    assert np.max(abs(muted[-1000:])) < 1e-5


def test_duck_earcons_and_callback_error_silence():
    cue = CueParams(0, 440, 4, True, 0, .2)
    tone = Tone()
    plain = render(tone, cue)
    duck = render(tone, cue, ducked=True)
    ratio = np.sqrt(np.mean(duck[-4000:]**2)/np.mean(plain[-4000:]**2))
    assert .14 < ratio < .18
    for name in EARCONS:
        tone.trigger(name)
        signal = render(tone, None, seconds=.3)
        assert np.isfinite(signal).all() and .02 < np.max(abs(signal)) < 1
    engine = CueEngine({})
    engine._tone = types.SimpleNamespace(render=lambda *a: (_ for _ in ()).throw(ValueError()))
    out = np.ones((256, 2), np.float32)
    engine._callback(out, 256, None, types.SimpleNamespace(output_underflow=True))
    assert not out.any()
    assert engine.stats()["underflows"] == 1 and engine.stats()["callback_errors"] == 1


def test_device_names_are_scoped_to_host(monkeypatch):
    from reach import audio
    monkeypatch.setattr(audio.sd, "query_hostapis", lambda: [
        {"name": "MME", "default_output_device": 0},
        {"name": "Windows WASAPI", "default_output_device": 1}])
    monkeypatch.setattr(audio.sd, "query_devices", lambda: [
        {"name": "Headphones", "hostapi": 0, "max_output_channels": 2},
        {"name": "Headphones", "hostapi": 1, "max_output_channels": 2}])
    assert select_output({"host_api": "Windows WASAPI", "output_device": "Headphones"})[0] == 1
    with pytest.raises(ValueError):
        select_output({"host_api": "Windows WASAPI", "output_device": 0})
