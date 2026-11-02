"""Continuous stereo cues with smoothed controls and precomputed earcons."""

import math
import sys
import time

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
        level = min(.65, gain*2.5)*.70710678
        self.earcons = {name: np.column_stack((buffer*level, buffer*level))
                       for name, buffer in make_earcons(fs).items()}
        self.earcon = None
        self.position = 0
        self.speech = None
        self.speech_position = 0
        self.base_gain = gain
        noise = np.random.default_rng(7).standard_normal(round(fs))
        spectrum = np.fft.rfft(noise)
        bands = np.fft.rfftfreq(len(noise), 1/fs)
        spectrum[(bands < 850) | (bands > 1550)] = 0
        self.noise = np.fft.irfft(spectrum).astype(np.float32)
        self.noise /= max(.001, float(np.sqrt(np.mean(self.noise**2))))
        self._noise_period = len(self.noise)
        self.noise = np.concatenate((self.noise, self.noise[:blocksize]))
        self.noise_pos = 0
        self._ramps = np.arange(1, blocksize+1)
        self._coefficients = {}
        for lost in (False, True):
            for ducked in (False, True):
                tau = np.array([.008, .025, .008, .030 if lost else .008, .008, .008,
                                .008 if ducked else .125])
                self._coefficients[lost, ducked] = np.exp(-self._ramps[:, None]/(self.fs*tau))

        self._target = np.zeros(7)
        self._controls = np.empty((blocksize, 7))
        self._work = [np.empty(blocksize) for _ in range(9)]

    def trigger(self, name):
        self.earcon, self.position = self.earcons[name], 0

    def render(self, out, cue, ducked=False, lost=False):
        frames = len(out)
        if not 0 < frames <= self.blocksize:
            raise ValueError('Unexpected audio block size')
        ducked = bool(ducked or self.speech is not None)
        target = self._target
        target[0] = cue.pan if cue else 0
        target[1] = math.log(cue.freq_hz) if cue else self.current[1]
        target[2] = cue.pulse_hz if cue else 4
        target[3] = cue.gain if cue and not lost else 0
        target[4] = float(cue.continuous) if cue else 0
        target[5] = cue.noise if cue else 0
        target[6] = 10**(-16/20) if ducked else 1
        controls = self._controls[:frames]
        coefficients = self._coefficients[bool(lost), ducked][:frames]
        for column in range(7):
            value = controls[:, column]
            np.subtract(self.current[column], target[column], out=value)
            np.multiply(value, coefficients[:, column], out=value)
            np.add(value, target[column], out=value)
            self.current[column] = value[-1]
        freq, angles, pulses, envelope, mono, temp, theta, left, right = (v[:frames] for v in self._work)
        np.exp(controls[:, 1], out=freq)
        np.multiply(freq, 2*np.pi/self.fs, out=freq)
        np.cumsum(freq, out=angles)
        np.add(angles, self.phase, out=angles)
        self.phase = float(angles[-1] % (2*np.pi))
        np.sin(angles, out=mono)
        np.multiply(controls[:, 2], 2*np.pi/self.fs, out=pulses)
        np.cumsum(pulses, out=pulses)
        np.add(pulses, self.pulse_phase, out=pulses)
        self.pulse_phase = float(pulses[-1] % (2*np.pi))
        np.cos(pulses, out=envelope)
        np.multiply(envelope, -.5, out=envelope)
        np.add(envelope, .5, out=envelope)
        np.subtract(1, controls[:, 4], out=temp)
        np.multiply(temp, envelope, out=envelope)
        np.add(envelope, controls[:, 4], out=envelope)
        np.multiply(mono, envelope, out=mono)
        noise = self.noise[self.noise_pos:self.noise_pos+frames]
        self.noise_pos = int((self.noise_pos+frames) % self._noise_period)
        np.multiply(noise, controls[:, 5], out=temp)
        np.add(mono, temp, out=mono)
        np.multiply(mono, controls[:, 3], out=mono)
        np.multiply(mono, controls[:, 6], out=mono)
        if self.earcon is not None:
            np.multiply(mono, .5, out=mono)
        np.add(controls[:, 0], 1, out=theta)
        np.multiply(theta, np.pi/4, out=theta)
        np.cos(theta, out=left)
        np.sin(theta, out=right)
        np.multiply(mono, left, out=out[:, 0])
        np.multiply(mono, right, out=out[:, 1])
        if self.earcon is not None:
            count = min(frames, len(self.earcon)-self.position)
            np.add(out[:count], self.earcon[self.position:self.position+count], out=out[:count])
            self.position += count
            if self.position >= len(self.earcon):
                self.earcon = None
        if self.speech is not None:
            count = min(frames, len(self.speech)-self.speech_position)
            np.add(out[:count], self.speech[self.speech_position:self.speech_position+count], out=out[:count])
            self.speech_position += count
            if self.speech_position >= len(self.speech):
                self.speech = None
        np.clip(out, -1, 1, out=out)


class CueEngine:
    def __init__(self, cfg, *, stream_factory=None):
        self.cfg = cfg.get("audio", {})
        self._speech_gain = float(cfg.get('tts', {}).get('gain', .5))
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
        self._speech_command = (0, None, None)
        self._speech_seen = 0
        self._speech_id = None
        self._speech_events = [[None, None, 0.0] for _ in range(64)]
        self._speech_write = self._speech_read = 0
        self._speech_event_drops = 0
        self._muted = False
        self._recorder = None

    def begin_recording(self,path,log=None,ring_s=4):
        from .recording import AudioRecorder
        if not self.running or self._recorder is not None:
            raise RuntimeError('Start the output stream before recording it once')
        self._recorder = AudioRecorder(path,self.sample_rate,self._device['blocksize'],log,ring_s)

    @property
    def sample_rate(self):
        return self._device['samplerate'] if self._device else 48000

    def speak(self, samples, ident):
        mono = np.asarray(samples, np.float32).reshape(-1)
        if not len(mono) or not np.isfinite(mono).all():
            raise ValueError('Speech samples must be finite and nonempty')
        stereo = np.repeat((mono*self._speech_gain)[:, None], 2, axis=1)
        np.clip(stereo, -1, 1, out=stereo)
        stereo.setflags(write=False)
        self._speech_command = (self._speech_command[0]+1, stereo, ident)

    def cancel_speech(self):
        self._speech_command = (self._speech_command[0]+1, None, None)

    def set_muted(self, value):
        self._muted = bool(value)

    def _speech_event(self, event, ident, stamp):
        if self._speech_write-self._speech_read >= len(self._speech_events):
            self._speech_read += 1
            self._speech_event_drops += 1
        row = self._speech_events[self._speech_write % len(self._speech_events)]
        row[0], row[1], row[2] = event, ident, stamp
        self._speech_write += 1

    def poll_speech_events(self):
        result = []
        while self._speech_read < self._speech_write:
            result.append(tuple(self._speech_events[self._speech_read % len(self._speech_events)]))
            self._speech_read += 1
        return result

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
            speech = self._speech_command
            if speech[0] != self._speech_seen:
                if self._speech_id is not None:
                    self._speech_event('cancel', self._speech_id, time.perf_counter())
                self._tone.speech, self._tone.speech_position = speech[1], 0
                self._speech_id = speech[2]
                self._speech_seen = speech[0]
                if self._speech_id is not None:
                    self._speech_event('start', self._speech_id, time.perf_counter())
            command = self._command
            if command[0] != self._seen_command:
                self._tone.trigger(command[1])
                self._seen_command = command[0]
            cue = self.cue
            self._tone.render(outdata, cue, self.ducked, self._lost)
            if self._speech_id is not None and self._tone.speech is None:
                self._speech_event('end', self._speech_id, time.perf_counter())
                self._speech_id = None
            if self._muted:
                outdata.fill(0)
        except Exception:
            outdata.fill(0)
            self._callback_errors += 1
            self._error = "audio callback failed"
        if self._recorder is not None:
            stamp = time.perf_counter()
            try:
                delta = float(timing.outputBufferDacTime)-float(timing.currentTime)
                if math.isfinite(delta) and 0 <= delta < 2:
                    stamp += delta
            except (AttributeError,TypeError,ValueError):
                pass
            self._recorder.accept(outdata,stamp)

    def stats(self):
        return {"underflows": self._underflows, "callback_errors": self._callback_errors,
                "cpu_load": float(self._stream.cpu_load) if self._stream else 0.0,
                "reported_output_latency_ms": float(self._stream.latency)*1000 if self._stream else None,
                "ducked": self.ducked, "earcon": self.last_earcon, "device": self._device,
                "speech_active": self._speech_id is not None, "speech_event_drops": self._speech_event_drops,
                "error": self._error, "active": bool(self._stream.active) if self._stream else False}

    def recording_stats(self):
        return self._recorder.stats() if self._recorder is not None else None

    def stop(self):
        self.cancel_speech()
        self.cue = None
        self.running = False
        if self._stream is not None:
            try:
                self._stream.stop()
            finally:
                self._stream.close()
                self._stream = None
        if self._recorder is not None:
            self._recorder.stop()
