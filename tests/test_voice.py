import asyncio
from types import SimpleNamespace as NS
import json

import websockets
from websockets.asyncio.server import serve

from reach.types import Command
from reach.voice import VoiceClient, reconnect_delay


def parse(text, objects, vocabulary):
    word = text.strip().lower().strip('.!?')
    return Command('find' if word == 'find the wallet' else word if word in ('stop', 'cancel') else 'unknown',
                   'wallet' if word == 'find the wallet' else None, text, None, 0)


async def source():
    while True:
        yield bytes(3840)
        await asyncio.sleep(.005)


async def until(predicate, timeout=8):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(.01)


def test_fake_server_fast_stop_reconnect_and_final_drain():
    async def scenario():
        commands, events, acknowledgements, clean = [], [], [], []
        count = 0
        async def handler(ws):
            nonlocal count
            count += 1
            number = count
            handshake = json.loads(await ws.recv())
            acknowledgements.append(handshake)
            await ws.send(json.dumps({'sessionId': f'synthetic-{number}'}))
            if number in (1, 2):
                await ws.close(code=1011 if number == 1 else 1013)
                return
            for event in [
                {'type': 'speechStart', 'turnId': 1, 'audioProcessedMs': 80},
                {'type': 'transcript', 'transcript': 'stop'},
                {'type': 'transcript', 'transcript': 'stop'},
                {'type': 'speechEnd', 'turnId': 1, 'audioProcessedMs': 240},
                {'type': 'speechComplete', 'turnId': 1, 'transcript': 'stop'},
                {'type': 'futureEvent'},
                {'type': 'transcript', 'transcript': "don't stop"},
            ]:
                await ws.send(json.dumps(event))
            async for value in ws:
                if isinstance(value, str) and json.loads(value).get('type') == 'endStream':
                    await ws.send(json.dumps({'type': 'speechComplete', 'turnId': 2, 'transcript': 'find the wallet'}))
                    clean.append(True)
                    await ws.close(code=1000)
                    return
        async with serve(handler, '127.0.0.1', 0) as server:
            port = server.sockets[0].getsockname()[1]
            log = NS(emit=lambda kind, data: events.append((kind, data)))
            client = VoiceClient({}, log, commands.append, parser=parse, audio_source=source(),
                                 uri=f'ws://127.0.0.1:{port}', api_key='synthetic')
            client.start()
            try:
                await until(lambda: len(commands) == 1)
            finally:
                await asyncio.to_thread(client.stop)
            assert [c.kind for c in commands] == ['stop', 'find']
            assert clean and not client._thread.is_alive()
            assert client.stats()['sessions'] == 3 and client.stats()['reconnects'] == 2
            delays = [d['delay_s'] for k,d in events if k == 'voice_session' and 'delay_s' in d]
            assert delays == [0, 2]
            assert all(h['authorization']['accessToken'] == 'Bearer synthetic' for h in acknowledgements)
            assert all(h['audioEncoding'] == 'PCM_24KHZ' for h in acknowledgements)
            turn = next(d for k,d in events if k == 'voice_turn' and d['turn_id'] == 1)
            assert turn['speech_start_audio_processed_ms'] == 80
            assert turn['speech_end_audio_processed_ms'] == turn['audio_processed_ms'] == 240
    asyncio.run(scenario())


def test_frame_queue_drops_oldest_and_backoff_bounds():
    async def scenario():
        client = VoiceClient({}, None, lambda c: None)
        client._queue, client._stop_event = asyncio.Queue(maxsize=2), asyncio.Event()
        for value in range(100):
            client._enqueue(bytes([value]))
        assert client._queue.qsize() == 2 and client.stats()['dropped_frames'] == 98
        assert await client._queue.get() == bytes([98])
        assert await client._queue.get() == bytes([99])
    asyncio.run(scenario())
    assert [reconnect_delay(1013, i) for i in range(5)] == [2, 4, 8, 8, 8]
    assert [reconnect_delay(1006, i) for i in range(6)] == [.5, 1, 2, 4, 8, 8]


def test_push_to_talk_buffers_before_ack_and_silences_endpointing():
    async def scenario():
        commands, normal_packets, ptt_packets = [], [], []
        ptt_seen = asyncio.Event()
        async def noisy_source():
            while True:
                yield b'\x01\x00'*1920
                await asyncio.sleep(.01)
        async def handler(ws):
            handshake = json.loads(await ws.recv())
            ptt = handshake['mode'] == 'PUSH_TO_TALK'
            if ptt:
                ptt_seen.set()
                await asyncio.sleep(.1)
            await ws.send(json.dumps({'sessionId': 'ptt' if ptt else 'normal'}))
            async for message in ws:
                if isinstance(message, bytes):
                    (ptt_packets if ptt else normal_packets).append(message)
                elif json.loads(message).get('type') == 'endStream':
                    if ptt:
                        await ws.send(json.dumps({'type': 'transcript', 'final': True, 'transcript': 'find the wallet'}))
                    await ws.close(code=1000)
                    return
        async with serve(handler, '127.0.0.1', 0) as server:
            port = server.sockets[0].getsockname()[1]
            client = VoiceClient({}, None, commands.append, parser=parse, audio_source=noisy_source(),
                                 uri=f'ws://127.0.0.1:{port}', api_key='synthetic')
            client.start()
            try:
                await until(lambda: client.stats()['connected'])
                assert client.toggle_ptt()
                await ptt_seen.wait()
                await asyncio.sleep(.15)
                assert client.toggle_ptt()
                await until(lambda: len(commands) == 1)
                assert commands[0].kind == 'find'
                assert len(ptt_packets) >= 5 and all(any(p) for p in ptt_packets)
                assert any(not any(p) for p in normal_packets)
                assert client.stats()['ptt_p50_ms'] is not None
            finally:
                await asyncio.to_thread(client.stop)
    asyncio.run(scenario())


def test_airpods_input_is_warned_even_when_metadata_reports_stereo(monkeypatch):
    import pytest
    pytest.importorskip('reach.speech_devices',reason='Shared device checks not integrated yet')
    events=[]
    client=None
    class Stream:
        def __init__(self,**kwargs): pass
        def start(self): client._stop_event.set()
        def stop(self): pass
        def close(self): pass
    monkeypatch.setattr('reach.voice.select_input',lambda cfg: (0,'Core Audio',{'name':'AirPods Pro'}))
    client=VoiceClient({},NS(emit=lambda kind,data: events.append((kind,data))),lambda c: None,
        stream_factory=Stream,api_key='synthetic',output_probe=lambda: {
            'name':'AirPods Pro','max_output_channels':2,'default_samplerate':48000})
    asyncio.run(client._run())
    assert not client.stats()['stereo_ok']
    assert any('AirPods microphone' in data.get('reason','') for _,data in events)


def test_real_grammar_preserves_command_provenance():
    import pytest
    pytest.importorskip('reach.grammar',reason='Shared grammar not integrated yet')
    client=VoiceClient({'objects':['wallet','pen','ball','apple']},None,lambda c: None,
                       clock=lambda: 12.5)
    command=client._parse('Where is my wallet?',7,'voice')
    assert (command.kind,command.phrase,command.turn_id,command.t_mono,command.source)==('find','wallet',7,12.5,'voice')
