"""Local speech rendered in a worker and played in the guidance audio stream."""

from collections import deque
from dataclasses import dataclass
import math
import sys
import threading
import time

import numpy as np


def trim_silence(samples, rate, threshold_db=-40, padding_s=.020):
    samples = np.asarray(samples, np.float32).reshape(-1)
    active = np.flatnonzero(np.abs(samples) >= 10**(threshold_db/20))
    if not len(active):
        return np.zeros(max(1, round(rate*.05)), np.float32)
    pad = round(rate*padding_s)
    return samples[max(0, active[0]-pad):min(len(samples), active[-1]+pad+1)].copy()


class NullEngine:
    name = 'null'
    def __init__(self, cfg=None):
        pass
    def synthesize(self, text, sample_rate):
        return np.zeros(round(sample_rate*.05), np.float32)
    def close(self):
        pass


class SilentSpeaker:
    """Captions without opening a synthesis engine or output device."""
    busy = False
    last_text = None
    def __init__(self, *args, **kwargs):
        pass
    def start(self):
        pass
    def say(self, text, key=None, priority=0):
        self.last_text = text
    def cancel(self, **kwargs):
        self.last_text = None
    def poll(self):
        pass
    def stats(self):
        return {'busy': False, 'text': self.last_text, 'priority': None, 'engine': 'disabled', 'error': None}
    def stop(self):
        pass


class SapiEngine:
    name = 'sapi'
    def __init__(self, cfg=None):
        import comtypes
        from comtypes.client import CreateObject
        self._comtypes, self._create = comtypes, CreateObject
        comtypes.CoInitialize()
        try:
            self._voice = CreateObject('SAPI.SpVoice', dynamic=True)
            self._voice.Rate = max(-10, min(10, round(10*math.log2(float((cfg or {}).get('rate', 210))/200))))
            selected = (cfg or {}).get('voice')
            if selected:
                tokens = self._voice.GetVoices()
                matches = [tokens.Item(i) for i in range(tokens.Count)
                           if selected.casefold() in tokens.Item(i).GetDescription().casefold()]
                if len(matches) != 1:
                    raise ValueError('Speech voice must identify one installed voice')
                self._voice.Voice = matches[0]
        except Exception:
            comtypes.CoUninitialize()
            raise
    def synthesize(self, text, sample_rate):
        stream = self._create('SAPI.SpMemoryStream', dynamic=True)
        stream.Format.Type = 38
        self._voice.AudioOutputStream = stream
        self._voice.Speak(text, 0)
        data = bytes(stream.GetData())
        result = np.frombuffer(data, dtype='<i2').astype(np.float32)/32768
        if sample_rate != 48000 and len(result):
            count = max(1, round(len(result)*sample_rate/48000))
            result = np.interp(np.arange(count)*48000/sample_rate, np.arange(len(result)), result).astype(np.float32)
        return result
    def close(self):
        self._voice = None
        self._comtypes.CoUninitialize()


class SayEngine:
    """Adapter slot for the macOS engine; the worker owns its lifecycle."""
    name = 'say'
    def __init__(self, cfg=None):
        from .say_engine import SayEngine as Engine
        self.engine = Engine(cfg or {})
    def synthesize(self, text, sample_rate):
        return self.engine.synthesize(text, sample_rate)
    def close(self):
        if hasattr(self.engine, 'close'):
            self.engine.close()


def phrase_texts(objects):
    texts = ['I see more than one.', 'Bring your hand into view.', 'Checking.', 'Cancelled.',
             'Stopped.', 'Ready.', 'Voice is offline. Use the keys.', 'Connection problem. Still trying.']
    for obj in objects:
        texts.extend([f'Looking for the {obj}.', f"I can't see the {obj} yet. Move the phone slowly.",
            f"I couldn't find the {obj}.", f'Lost the {obj}. Hold the phone still.', f'I lost the {obj}.',
            f"You're touching the {obj}."])
        texts.extend(f'Found the {obj}, {where}.' for where in
            ('to your left', 'to your right', 'straight ahead', 'ahead on your left', 'ahead on your right'))
        texts.extend(f'Trial {i}. Find the {obj}.' for i in range(1, 6))
    return list(dict.fromkeys(texts))


@dataclass
class Utterance:
    ident: int
    text: str
    key: str | None
    priority: int
    requested: float
    processing: bool = False


class Speaker:
    def __init__(self, cfg, audio, log=None, *, engine_factory=None, clock=time.perf_counter):
        self.cfg, self.audio, self.log = cfg.get('tts', {}), audio, log
        self._factory, self._clock = engine_factory, clock
        self._condition = threading.Condition()
        self._stop = False
        self._thread = None
        self._current = None
        self._waiting = deque()
        self._cache = {}
        self._preload = deque(phrase_texts(cfg.get('objects', ['wallet', 'pen', 'ball', 'apple'])))
        self._requests = {}
        self._next = 0
        self._engine_name = None
        self._error = None
        self._fallback = False
        self.last_text = None
        self._latencies = deque(maxlen=1000)

    def _emit(self, event, job=None, **data):
        if self.log:
            self.log.emit('speech', {'event': event, **({'id': job.ident, 'key': job.key,
                'text': job.text, 'priority': job.priority} if job else {}), **data})

    def start(self):
        with self._condition:
            if self._thread:
                return
            self._thread = threading.Thread(target=self._run, name='speech', daemon=True)
            self._thread.start()

    @property
    def busy(self):
        with self._condition:
            return self._current is not None or bool(self._waiting)

    def say(self, text, key=None, priority=0):
        if not text or len(text) > 300 or priority not in (0, 1, 2):
            raise ValueError('Speech needs short text and priority 0, 1 or 2')
        self.start()
        with self._condition:
            if self._stop:
                return None
            job = Utterance(self._next, text, key, priority, self._clock())
            self._next += 1
            self._requests[job.ident] = job
            self.last_text = text
            self._emit('request', job)
            if self._current is None:
                self._current = job
            elif priority >= self._current.priority:
                old = self._current
                self.audio.cancel_speech()
                self._emit('cancel', old, reason='replaced')
                self._requests.pop(old.ident, None)
                self._current = job
            else:
                if len(self._waiting) >= 2:
                    oldest_info = next((j for j in self._waiting if j.priority == 0), self._waiting[0])
                    self._waiting.remove(oldest_info)
                    self._emit('drop', oldest_info, reason='waiting queue full')
                    self._requests.pop(oldest_info.ident, None)
                self._waiting.append(job)
            self._condition.notify_all()
            return job.ident

    def cancel(self, *, info_only=False):
        with self._condition:
            if self._current and (not info_only or self._current.priority == 0):
                self.audio.cancel_speech()
                self._emit('cancel', self._current, reason='barge in' if info_only else 'cancelled')
                self._requests.pop(self._current.ident, None)
                self._current = None
            retained = deque()
            for job in self._waiting:
                if info_only and job.priority > 0:
                    retained.append(job)
                else:
                    self._emit('drop', job, reason='cancelled')
                    self._requests.pop(job.ident, None)
            self._waiting = retained
            if self._current is None and self._waiting:
                self._current = self._waiting.popleft()
            self._condition.notify_all()

    def poll(self):
        for event, ident, stamp in self.audio.poll_speech_events():
            with self._condition:
                job = self._requests.get(ident)
                if job is None:
                    continue
                extra = {}
                if event == 'start':
                    extra['first_audio_ms'] = max(0, (stamp-job.requested)*1000)
                    self._latencies.append(extra['first_audio_ms'])
                if event == 'end':
                    if self._current and self._current.ident == ident:
                        self._current = self._waiting.popleft() if self._waiting else None
                        self._condition.notify_all()
                self._emit(event, job, **extra)
                if event in ('end', 'cancel'):
                    self._requests.pop(ident, None)

    def _run(self):
        engine = None
        try:
            factory = self._factory
            if factory is None:
                name = self.cfg.get('engine', 'auto')
                name = ('sapi' if sys.platform == 'win32' else 'say' if sys.platform == 'darwin' else 'null') if name == 'auto' else name
                factory = {'sapi': SapiEngine, 'say': SayEngine, 'null': NullEngine}[name]
            try:
                engine = factory(self.cfg)
            except Exception as exc:
                self._fallback = True
                self._error = f'Local speech unavailable ({type(exc).__name__}); captions remain available'
                self._emit('drop', reason=self._error)
                engine = NullEngine()
            self._engine_name = engine.name
            while True:
                with self._condition:
                    if self._stop:
                        break
                    job = self._current
                    if job is not None and not job.processing:
                        job.processing = True
                        text = job.text
                    elif self._preload:
                        text, job = self._preload.popleft(), None
                    else:
                        self._condition.wait(.02)
                        continue
                rate = self.audio.sample_rate
                cache_key = (engine.name, self.cfg.get('voice'), self.cfg.get('rate', 210), rate, text)
                start = self._clock()
                cached = cache_key in self._cache
                if not cached:
                    raw = engine.synthesize(text, rate)
                    samples = trim_silence(raw, rate, float(self.cfg.get('trim_db', -40)))
                    samples.setflags(write=False)
                    self._cache[cache_key] = samples
                samples = self._cache[cache_key]
                synth_ms = (self._clock()-start)*1000
                if job is not None:
                    with self._condition:
                        if self._current is not job or self._stop:
                            continue
                        self._emit('ready', job, synth_ms=synth_ms, cached=cached, duration_s=len(samples)/rate)
                        self.audio.speak(samples, job.ident)
        except Exception as exc:
            self._error = f'Speech worker stopped ({type(exc).__name__})'
            self.audio.cancel_speech()
            with self._condition:
                self._current = None
                self._waiting.clear()
            try:
                self._emit('drop', reason=self._error)
            except Exception:
                pass
        finally:
            if engine:
                engine.close()

    def stats(self):
        with self._condition:
            current = self._current
            latencies = tuple(self._latencies)
            return {'busy': current is not None or bool(self._waiting), 'text': current.text if current else None,
                    'caption': self.last_text,
                    'priority': current.priority if current else None, 'engine': self._engine_name,
                    'cached_phrases': len(self._cache), 'fallback': self._fallback, 'error': self._error,
                    'first_audio_p50_ms': float(np.median(latencies)) if latencies else None,
                    'first_audio_p95_ms': float(np.percentile(latencies, 95)) if latencies else None}

    def stop(self):
        self.cancel()
        with self._condition:
            self._stop = True
            self._condition.notify_all()
        if self._thread:
            self._thread.join(timeout=6)
            if self._thread.is_alive():
                raise RuntimeError('Speech worker did not stop within its deadline')
