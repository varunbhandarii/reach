"""Continuous stereo cues with smoothed controls and precomputed earcons."""

import math
import sys

import numpy as np
import sounddevice as sd

from .types import CueParams


EARCONS = ("found", "lost", "reacquired", "contact", "wrong")


def select_output(cfg):
    hosts, devices = sd.query_hostapis(), sd.query_devices()
    name = cfg.get("host_api", "auto")
    if name == "auto":
        name = "Windows WASAPI" if sys.platform == "win32" else "Core Audio" if sys.platform == "darwin" else "ALSA"
    host = next((i for i, h in enumerate(hosts) if h["name"] == name), None)
    if host is None:
        raise ValueError(f"Audio host API unavailable: {name}")
    selected = cfg.get("output_device")
    if selected is None:
        selected = hosts[host]["default_output_device"]
    elif not str(selected).isdigit():
        matches = [i for i, d in enumerate(devices) if d["hostapi"] == host
                   and d["max_output_channels"] >= 2 and str(selected).casefold() in d["name"].casefold()]
        if len(matches) != 1:
            raise ValueError("Output name must identify one stereo device on the selected host API")
        selected = matches[0]
    selected = int(selected)
    if not 0 <= selected < len(devices):
        raise ValueError("No output device; use reach devices")
    info = devices[selected]
    if info["hostapi"] != host or info["max_output_channels"] < 2:
        raise ValueError("Select a stereo output on the chosen host API")
    return selected, name, info


def pluck(fs, frequencies, seconds):
    t = np.arange(round(fs * seconds)) / fs
    signal = np.mean([np.sin(2 * np.pi * frequency * t) for frequency in frequencies], axis=0)
    envelope = np.minimum(1, t / .005) * np.exp(-t / max(.025, seconds / 2))
    envelope *= np.clip((seconds-t) / .015, 0, 1)
    return (signal * envelope).astype(np.float32)


def make_earcons(fs):
    gap = np.zeros(round(fs*.015), np.float32)
    note = lambda f, s: pluck(fs, [f], s)
    buffers = {"found": np.concatenate([note(1046.5, .09), gap, note(1318.5, .13)]),
            "lost": np.concatenate([note(1318.5, .09), gap, note(1046.5, .13)]),
            "reacquired": note(1396.9, .13),
            "contact": pluck(fs, [1046.5, 1318.5, 1568], .15),
            "wrong": np.concatenate([note(330, .08), gap, note(294, .08)])}
    for buffer in buffers.values():
        buffer /= max(.001, float(np.max(np.abs(buffer))))
    return buffers


class Tone:
    """Block synthesizer; device-free rendering is useful for waveform checks."""

    def __init__(self, fs=48000, blocksize=256, gain=.2):
        self.fs, self.blocksize = fs, blocksize
        self.phase = self.pulse_phase = 0.0
        self.current = np.array([0, math.log(440), 4, 0, 0, 0, 1.0])
        self.earcons = make_earcons(fs)
        self.earcon = None
        self.position = 0
        self.base_gain = gain
        noise = np.random.default_rng(7).standard_normal(round(fs))
        spectrum = np.fft.rfft(noise)
        bands = np.fft.rfftfreq(len(noise), 1/fs)
        spectrum[(bands < 850) | (bands > 1550)] = 0
        self.noise = np.fft.irfft(spectrum).astype(np.float32)
        self.noise /= max(.001, float(np.sqrt(np.mean(self.noise**2))))
        self.noise_pos = 0
        self._ramps = np.arange(1, blocksize+1)

    def trigger(self, name):
        self.earcon, self.position = self.earcons[name], 0

    def render(self, out, cue, ducked=False, lost=False):
        frames = len(out)
        samples = self._ramps if frames == self.blocksize else np.arange(1, frames+1)
        cue = cue or CueParams(0, math.exp(self.current[1]), 4, False, 0, 0)
        target = np.array([cue.pan, math.log(cue.freq_hz), cue.pulse_hz,
                           0 if lost else cue.gain, float(cue.continuous), cue.noise,
                           10**(-16/20) if ducked else 1.0])
        # pitch 25 ms; pan/gain 8 ms; lost 30 ms; duck release 125 ms.
        tau = np.array([.008, .025, .008, .030 if lost else .008, .008, .008,
                        .008 if ducked else .125])
        controls = target + (self.current-target) * np.exp(-samples[:, None] / (self.fs*tau))
        self.current = controls[-1].copy()
        pan, log_freq, pulse, gain, continuous, noise_gain, duck = controls.T
        angles = self.phase + np.cumsum(2*np.pi*np.exp(log_freq)/self.fs)
        pulses = self.pulse_phase + np.cumsum(2*np.pi*pulse/self.fs)
        self.phase, self.pulse_phase = float(angles[-1] % (2*np.pi)), float(pulses[-1] % (2*np.pi))
        indices = (np.arange(frames) + self.noise_pos) % len(self.noise)
        self.noise_pos = int((self.noise_pos + frames) % len(self.noise))
        envelope = continuous + (1-continuous) * (.5-.5*np.cos(pulses))
        mono = gain * duck * (envelope*np.sin(angles) + self.noise[indices]*noise_gain)
        active = self.earcon is not None
        if active:
            mono *= .5
        theta = (pan+1)*np.pi/4
        out[:, 0] = mono*np.cos(theta)
        out[:, 1] = mono*np.sin(theta)
        if active:
            count = min(frames, len(self.earcon)-self.position)
            ear = self.earcon[self.position:self.position+count] * min(.65, self.base_gain*2.5)
            out[:count, 0] += ear*.70710678
            out[:count, 1] += ear*.70710678
            self.position += count
            if self.position >= len(self.earcon):
                self.earcon = None
        np.clip(out, -1, 1, out=out)


class CueEngine:
    def __init__(self, cfg, *, stream_factory=None):
        self.cfg = cfg.get("audio", {})
        self._stream_factory = stream_factory or sd.OutputStream
        self.cue = None
        self.ducked = self.running = False
        self.last_earcon = None
        self._command = (0, None)
        self._seen_command = 0
        self._stream = self._tone = None
        self._underflows = self._callback_errors = 0
        self._error = None
        self._lost = False
        self._device = None

    def start(self):
        if self.running:
            return
        index, host, info = select_output(self.cfg)
        fs = round(info["default_samplerate"])
        size = int(self.cfg.get("blocksize", 256))
        self._tone = Tone(fs, size, float(self.cfg.get("gain", .2)))
        extra = sd.WasapiSettings(exclusive=bool(self.cfg.get("exclusive", False))) if host == "Windows WASAPI" else None
        self._stream = self._stream_factory(device=index, channels=2, samplerate=fs, blocksize=size,
            dtype="float32", latency=self.cfg.get("latency", "low"), extra_settings=extra, callback=self._callback)
        self._device = {"name": info["name"], "host_api": host, "samplerate": fs, "blocksize": size}
        try:
            self._stream.start()
            self.running = True
        except Exception:
            self._stream.close()
            self._stream = None
            raise

    def set(self, cue):
        if cue is not None:
            values = (cue.pan, cue.freq_hz, cue.pulse_hz, cue.noise, cue.gain)
            if not all(math.isfinite(value) for value in values):
                raise ValueError("Cue values must be finite")
            if not (-1 <= cue.pan <= 1 and 20 <= cue.freq_hz <= 2000
                    and 0 <= cue.pulse_hz <= 30 and 0 <= cue.noise <= 1 and 0 <= cue.gain <= .5):
                raise ValueError("Cue values outside supported range")
            self._lost = False
        self.cue = cue

    def earcon(self, name):
        if name not in EARCONS:
            raise ValueError(f"Unknown earcon: {name}")
        self.last_earcon = name
        self._command = (self._command[0]+1, name)
        if name == "lost":
            self._lost = True

    def duck(self, on):
        self.ducked = bool(on)

    def _callback(self, outdata, frames, timing, status):
        try:
            if status.output_underflow:
                self._underflows += 1
            command = self._command
            if command[0] != self._seen_command:
                self._tone.trigger(command[1])
                self._seen_command = command[0]
            cue = self.cue
            self._tone.render(outdata, cue, self.ducked, self._lost)
        except Exception:
            outdata.fill(0)
            self._callback_errors += 1
            self._error = "audio callback failed"

    def stats(self):
        return {"underflows": self._underflows, "callback_errors": self._callback_errors,
                "cpu_load": float(self._stream.cpu_load) if self._stream else 0.0,
                "reported_output_latency_ms": float(self._stream.latency)*1000 if self._stream else None,
                "ducked": self.ducked, "earcon": self.last_earcon, "device": self._device,
                "error": self._error, "active": bool(self._stream.active) if self._stream else False}

    def stop(self):
        self.cue = None
        self.running = False
        if self._stream is not None:
            try:
                self._stream.stop()
            finally:
                self._stream.close()
                self._stream = None
