"""Bounded microphone streaming, turn handling and reconnecting transcription."""

import asyncio
from collections import deque
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import re
import sys
import threading
import time

import numpy as np
import sounddevice as sd
import websockets
from websockets.exceptions import ConnectionClosed

from .config import PROJECT_ROOT
from .types import Command

URI = 'wss://api.meta.ai/v1/asr/realtime'


def reconnect_delay(code, attempt):
    if code in (1000, 1011):
        return 0.0
    if code == 1013:
        return float(min(8, 2*2**min(attempt, 3)))
    return float(min(8, .5*2**min(attempt, 5)))


def select_input(cfg):
    hosts, devices = sd.query_hostapis(), sd.query_devices()
    host_name = cfg.get('host_api', 'auto')
    if host_name == 'auto':
        host_name = 'Windows WASAPI' if sys.platform == 'win32' else 'Core Audio' if sys.platform == 'darwin' else 'ALSA'
    host = next((i for i,h in enumerate(hosts) if h['name'] == host_name), None)
    if host is None:
        raise ValueError('Microphone host API unavailable')
    selected = cfg.get('input_device')
    if selected is None and sys.platform == 'darwin':
        selected = 'MacBook'
    if selected is None:
        selected = hosts[host]['default_input_device']
    elif not str(selected).isdigit():
        matches = [i for i,d in enumerate(devices) if d['hostapi'] == host and d['max_input_channels'] > 0
                   and str(selected).casefold() in d['name'].casefold()]
        if len(matches) != 1:
            raise ValueError('Input name must identify one microphone on the selected host API')
        selected = matches[0]
    selected = int(selected)
    if not 0 <= selected < len(devices) or devices[selected]['hostapi'] != host or devices[selected]['max_input_channels'] < 1:
        raise ValueError('Selected microphone is unavailable on the chosen host API')
    return selected, host_name, devices[selected]


class VoiceClient:
    def __init__(self, cfg, log, post, *, parser=None, audio_source=None, uri=URI,
                 api_key=None, stream_factory=None, on_speech_start=None, speaking=None,
                 output_probe=None, clock=time.perf_counter):
        self.root_cfg, self.cfg = cfg, cfg.get('voice', {})
        self.log, self.post = log, post
        self._parser, self._audio_source, self._uri, self._key = parser, audio_source, uri, api_key
        self._stream_factory = stream_factory or sd.RawInputStream
        self._on_speech_start, self._speaking = on_speech_start, speaking or (lambda: False)
        self._output_probe, self._clock = output_probe, clock
        self._thread = self._loop = self._stop_event = self._queue = self._stream = None
        self._closed = threading.Event()
        self.running = False
        self._connected = False
        self._sessions = self._reconnects = self._turns = self._dropped = self._overflows = 0
        self._last_code = None
        self._error = None
        self._input_name = None
        self._stereo_ok = True
        self._latencies = deque(maxlen=1200)
        self._last_transcript = ''
        self._last_activity = self._clock()
        self._active_turns = set()
        self._ptt_task = self._ptt_queue = self._ptt_end = None
        self._ptt_latency = deque(maxlen=100)
        self._ptt_started = self._ptt_finished = None
        self._ptt_pending = False

    def _emit(self, kind, data):
        if self.log is not None:
            self.log.emit(kind, data)

    def _session_event(self, event, mode='ENDPOINTING', **data):
        self._emit('voice_session', {'event': event, 'mode': mode, **data})

    def start(self):
        if self._thread is not None:
            raise RuntimeError('Voice client can only be started once')
        self.running = True
        self._thread = threading.Thread(target=self._thread_main, name='voice', daemon=True)
        self._thread.start()

    def _thread_main(self):
        try:
            asyncio.run(self._run())
        except Exception as exc:
            self._error = f'voice stopped ({type(exc).__name__})'
            try:
                self._session_event('error', reason=self._error)
            except Exception:
                pass
        finally:
            self._connected = self.running = False
            self._closed.set()

    def _enqueue(self, pcm):
        if self._stop_event.is_set():
            return
        if self._ptt_queue is not None and not self._ptt_end.is_set():
            if self._ptt_queue.full():
                self._ptt_queue.get_nowait()
                self._dropped += 1
            self._ptt_queue.put_nowait(pcm)
        if self._queue.full():
            self._queue.get_nowait()
            self._dropped += 1
        self._queue.put_nowait(pcm)

    def _callback(self, data, frames, timing, status):
        self._overflows += bool(status)
        try:
            self._loop.call_soon_threadsafe(self._enqueue, bytes(data))
        except RuntimeError:
            pass

    async def _feed(self):
        async for pcm in self._audio_source:
            self._enqueue(bytes(pcm))
            if self._stop_event.is_set():
                break

    async def _run(self):
        self._loop = asyncio.get_running_loop()
        self._stop_event = asyncio.Event()
        self._queue = asyncio.Queue(maxsize=2)
        if self._closed.is_set():
            return
        if self._key is None:
            from dotenv import dotenv_values
            self._key = os.environ.get('MODEL_API_KEY') or dotenv_values(PROJECT_ROOT/'.env').get('MODEL_API_KEY')
        if not self._key:
            raise ValueError('Model API key missing')
        feed = None
        try:
            if self._audio_source is not None:
                self._input_name = 'injected audio source'
                feed = asyncio.create_task(self._feed())
            else:
                device, host, info = select_input(self.cfg)
                self._input_name = info['name']
                extra = sd.WasapiSettings(auto_convert=True) if host == 'Windows WASAPI' else None
                self._stream = self._stream_factory(device=device, samplerate=24000, channels=1,
                    dtype='int16', blocksize=1920, latency='low', extra_settings=extra, callback=self._callback)
                self._stream.start()
                if self._output_probe:
                    output = self._output_probe()
                    self._stereo_ok = output['max_output_channels'] >= 2 and output['default_samplerate'] >= 44100
                    if not self._stereo_ok:
                        self._session_event('error', reason='Stereo output unavailable after microphone opened')
            attempts = 0
            while not self._stop_event.is_set():
                code = None
                try:
                    code = await self._endpoint_session()
                except ConnectionClosed as exc:
                    code = exc.rcvd.code if exc.rcvd else 1006
                except (OSError, TimeoutError, websockets.exceptions.WebSocketException):
                    code = 1006
                except ValueError:
                    code = 1008
                self._connected = False
                self._last_code = code
                if self._stop_event.is_set():
                    break
                if code == 1008:
                    self._error = 'Voice handshake or configuration rejected; use keys'
                    self._session_event('error', code=code, reason=self._error)
                    break
                delay = reconnect_delay(code, attempts)
                self._reconnects += 1
                self._session_event('error', code=code, reason='reconnecting', delay_s=delay)
                attempts = 0 if code in (1000, 1011) else attempts+1
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=delay)
                except TimeoutError:
                    pass
        finally:
            if self._ptt_end:
                self._ptt_end.set()
            if self._ptt_task:
                try:
                    await asyncio.wait_for(asyncio.shield(self._ptt_task), 6)
                except (TimeoutError, Exception):
                    self._ptt_task.cancel()
                    await asyncio.gather(self._ptt_task, return_exceptions=True)
            if feed:
                feed.cancel()
                await asyncio.gather(feed, return_exceptions=True)
            if self._stream:
                self._stream.stop()
                self._stream.close()
            self._key = None

    def _handshake(self, mode):
        path = Path(self.cfg.get('keywords_file', 'config/keywords.txt'))
        if not path.is_absolute():
            path = PROJECT_ROOT/path
        words = path.read_text(encoding='utf-8').splitlines() if path.exists() else []
        words += self.root_cfg.get('objects', ['wallet', 'pen', 'ball', 'apple'])
        return {'authorization': {'accessToken': 'Bearer '+self._key}, 'audioEncoding': 'PCM_24KHZ',
                'model': self.cfg.get('model', 'muse-voice-transcribe-1.0'), 'mode': mode,
                'partialMode': 'CUMULATIVE', 'emitAudioProgress': True,
                'languageBias': ['English'], 'keywords': list(dict.fromkeys(w.strip() for w in words if w.strip()))}

    async def _ack(self, ws, mode):
        self._session_event('open', mode)
        await ws.send(json.dumps(self._handshake(mode)))
        message = json.loads(await asyncio.wait_for(ws.recv(), 10))
        if not isinstance(message, dict) or not isinstance(message.get('sessionId'), str) or message.get('type') == 'error':
            raise ValueError('Voice handshake rejected')
        self._sessions += 1
        self._session_event('ack', mode, session_id=message['sessionId'])
        return message['sessionId']

    def _parse(self, text, turn_id, source):
        if self._parser is None:
            from .grammar import parse
            parser = parse
        else:
            parser = self._parser
        command = parser(text, self.root_cfg.get('objects', ['wallet', 'pen', 'ball', 'apple']), self.cfg.get('vocabulary', []))
        fields = {'t_mono': self._clock(), 'turn_id': turn_id}
        if 'source' in Command.__dataclass_fields__:
            fields['source'] = source
        return replace(command, **fields)

    def _post(self, command):
        self.post(command)

    async def _endpoint_session(self):
        async with websockets.connect(self._uri, open_timeout=10, max_size=2**20, close_timeout=3) as ws:
            session = await self._ack(ws, 'ENDPOINTING')
            self._connected = True
            self._active_turns.clear()
            turns, fast = {}, {}
            active_id = None
            opened = self._clock()

            async def send():
                while True:
                    pcm = await self._queue.get()
                    await ws.send(bytes(len(pcm)) if self._ptt_pending else pcm)

            async def receive():
                nonlocal active_id
                async for raw in ws:
                    if not isinstance(raw, str):
                        continue
                    try:
                        event = json.loads(raw)
                    except ValueError:
                        continue
                    now = self._clock()
                    kind, tid = event.get('type'), event.get('turnId')
                    if kind == 'error':
                        raise ValueError('Voice server rejected session')
                    if kind == 'speechStart':
                        active_id = tid
                        self._active_turns.add(tid)
                        ignore = self._ptt_pending or (self.cfg.get('ignore_during_speech', False) and self._speaking())
                        turns[tid] = {'start': now, 'end': None, 'ignore': ignore}
                        if not ignore and self.cfg.get('barge_in', True) and self._on_speech_start:
                            self._on_speech_start()
                    elif kind == 'speechEnd':
                        turns.setdefault(tid, {'start': None, 'ignore': self._ptt_pending})['end'] = now
                    elif kind == 'transcript':
                        text = event.get('transcript', '')
                        if not isinstance(text, str):
                            continue
                        self._last_transcript = text
                        normalized = text.strip().lower().strip('.!?')
                        info = turns.get(active_id, {})
                        token = active_id
                        if (self.cfg.get('fast_stop', True) and normalized in ('stop', 'cancel')
                                and not info.get('ignore', self._ptt_pending)
                                and (token not in fast or now-fast[token][1] > 3)):
                            command = Command(normalized, None, text, active_id, now)
                            if 'source' in Command.__dataclass_fields__:
                                command = replace(command, source='voice_fast')
                            self._post(command)
                            fast[token] = (normalized, now)
                    elif kind == 'speechComplete':
                        text = event.get('transcript', '')
                        if not isinstance(text, str):
                            continue
                        self._last_transcript = text
                        self._active_turns.discard(tid)
                        info = turns.pop(tid, {'start': None, 'end': None, 'ignore': self._ptt_pending})
                        latency = (now-info['end'])*1000 if info.get('end') is not None else None
                        if latency is not None:
                            self._latencies.append(latency)
                        self._turns += 1
                        self._emit('voice_turn', {'session_id': session, 'turn_id': tid, 'transcript': text,
                            't_speech_start': info.get('start'), 't_speech_end': info.get('end'), 't_complete': now,
                            'audio_processed_ms': event.get('audioProcessedMs'), 'latency_ms': latency,
                            'latency_source': 'server_event_receipt', 'mode': 'ENDPOINTING', 'ignored': bool(info['ignore'])})
                        match = fast.pop(tid, fast.pop(None, None))
                        command = self._parse(text, tid, 'voice') if not info['ignore'] else None
                        duplicate = match and now-match[1] <= 3 and command is not None and command.kind == match[0]
                        if command is not None and not duplicate:
                            self._post(command)
                    self._last_activity = now
                return ws.close_code

            async def control():
                while not self._stop_event.is_set():
                    if self._clock()-opened >= float(self.cfg.get('rotate_min', 55))*60 and not self._active_turns and not self._ptt_pending:
                        return 'rotate'
                    await asyncio.sleep(.05)
                return 'stop'

            sender, receiver, controller = asyncio.create_task(send()), asyncio.create_task(receive()), asyncio.create_task(control())
            try:
                done, _ = await asyncio.wait((sender, receiver, controller), return_when=asyncio.FIRST_COMPLETED)
                if receiver in done:
                    return receiver.result()
                if sender in done:
                    sender.result()
                sender.cancel()
                await asyncio.gather(sender, return_exceptions=True)
                await ws.send(json.dumps({'type': 'endStream'}))
                try:
                    await asyncio.wait_for(asyncio.shield(receiver), 5)
                except TimeoutError:
                    self._session_event('error', session_id=session, reason='final drain timed out')
                return ws.close_code or 1000
            finally:
                for task in (sender, receiver, controller):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(sender, receiver, controller, return_exceptions=True)
                self._session_event('close', session_id=session, code=ws.close_code)
                self._connected = False

    def stats(self):
        values = tuple(self._latencies)
        ptt = tuple(self._ptt_latency)
        return {'connected': self._connected, 'sessions': self._sessions, 'reconnects': self._reconnects,
                'turns': self._turns, 'dropped_frames': self._dropped, 'input_overflows': self._overflows,
                'last_close_code': self._last_code, 'latency_p50_ms': float(np.median(values)) if values else None,
                'latency_p95_ms': float(np.percentile(values, 95)) if values else None,
                'ptt_p50_ms': float(np.median(ptt)) if ptt else None, 'ptt_p95_ms': float(np.percentile(ptt, 95)) if ptt else None,
                'input_name': self._input_name, 'last_transcript': self._last_transcript,
                'stereo_ok': self._stereo_ok, 'ptt_active': self._ptt_pending, 'error': self._error}

    def toggle_ptt(self):
        if self._loop is None or not self._connected or self._loop.is_closed():
            return False
        self._loop.call_soon_threadsafe(self._toggle_ptt)
        return True

    def _toggle_ptt(self):
        if self._ptt_pending:
            if not self._ptt_end.is_set():
                self._ptt_finished = self._clock()
                self._ptt_end.set()
            return
        self._ptt_pending = True
        self._ptt_started, self._ptt_finished = self._clock(), None
        self._ptt_queue = asyncio.Queue(maxsize=63)
        self._ptt_end = asyncio.Event()
        self._ptt_task = asyncio.create_task(self._ptt_session())

    async def _ptt_session(self):
        final = None
        session = None
        try:
            async with websockets.connect(self._uri, open_timeout=10, max_size=2**20, close_timeout=3) as ws:
                session = await self._ack(ws, 'PUSH_TO_TALK')

                async def send():
                    while not self._ptt_end.is_set() or not self._ptt_queue.empty():
                        if self._clock()-self._ptt_started > 60:
                            self._ptt_finished = self._clock()
                            self._ptt_end.set()
                        try:
                            pcm = await asyncio.wait_for(self._ptt_queue.get(), .05)
                        except TimeoutError:
                            continue
                        await ws.send(pcm)
                    await ws.send(json.dumps({'type': 'endStream'}))

                async def receive():
                    nonlocal final
                    async for raw in ws:
                        if not isinstance(raw, str):
                            continue
                        event = json.loads(raw)
                        if event.get('type') == 'error':
                            raise ValueError('PTT server error')
                        if event.get('type') == 'transcript':
                            text = event.get('transcript', '')
                            if isinstance(text, str):
                                self._last_transcript = text
                            if event.get('final') is True and isinstance(text, str):
                                final = event
                                now = self._clock()
                                latency = (now-self._ptt_finished)*1000 if self._ptt_finished is not None else None
                                if latency is not None:
                                    self._ptt_latency.append(latency)
                                self._emit('voice_turn', {'session_id': session, 'turn_id': event.get('turnId'),
                                    'transcript': text, 't_speech_start': self._ptt_started,
                                    't_speech_end': self._ptt_finished, 't_complete': now,
                                    'audio_processed_ms': event.get('audioProcessedMs'), 'latency_ms': latency,
                                    'latency_source': 'second_tap_to_final', 'mode': 'PUSH_TO_TALK', 'ignored': False})
                                self._post(self._parse(text, event.get('turnId'), 'ptt'))
                sender, receiver = asyncio.create_task(send()), asyncio.create_task(receive())
                try:
                    done, _ = await asyncio.wait((sender, receiver), return_when=asyncio.FIRST_COMPLETED)
                    if sender in done:
                        sender.result()
                        await asyncio.wait_for(asyncio.shield(receiver), 10)
                    else:
                        receiver.result()
                    if final is None:
                        raise ValueError('PTT closed without a final result')
                finally:
                    for task in (sender, receiver):
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(sender, receiver, return_exceptions=True)
                    self._session_event('close', 'PUSH_TO_TALK', session_id=session, code=ws.close_code)
        except Exception as exc:
            self._session_event('error', 'PUSH_TO_TALK', session_id=session,
                                reason=f'push-to-talk failed ({type(exc).__name__})')
        finally:
            self._ptt_pending = False
            self._ptt_queue = self._ptt_end = None

    def stop(self):
        self._closed.set()
        if self._loop and self._stop_event and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._stop_event.set)
        if self._thread:
            self._thread.join(timeout=22)
            if self._thread.is_alive():
                raise RuntimeError('Voice client did not stop within its deadline')
        self.running = False
