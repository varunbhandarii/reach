import sys
import threading
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest

from reach.audio import CueEngine, Tone
from reach.tts import Speaker, NullEngine, SapiEngine, trim_silence
from reach.types import CueParams


def engine():
    audio = CueEngine({})
    audio._tone = Tone()
    audio._device = {'samplerate': 48000, 'blocksize': 256}
    return audio


def block(audio):
    output = np.empty((256, 2), np.float32)
    audio._callback(output, 256, None, NS(output_underflow=False))
    return output


def test_speech_mix_is_stereo_ducked_and_cancellable():
    audio = engine()
    cue = CueParams(0, 440, 4, True, 0, .2)
    audio.set(cue)
    for _ in range(100):
        normal = block(audio)
    audio.speak(np.zeros(24000, np.float32), 3)
    for _ in range(50):
        ducked = block(audio)
    assert .14 < np.sqrt(np.mean(ducked**2)/np.mean(normal**2)) < .18
    audio.cancel_speech()
    block(audio)
    events = audio.poll_speech_events()
    assert [e[0] for e in events] == ['start', 'cancel']
    audio.speak(np.full(512, .2, np.float32), 4)
    audio.set(None)
    for _ in range(50):
        block(audio)
    assert [e[0] for e in audio.poll_speech_events()] == ['start', 'end']
    assert audio.stats()['callback_errors'] == 0


def test_trim_keeps_padding_and_silence_is_nonempty():
    raw = np.concatenate([np.zeros(4800), np.ones(4800)*.2, np.zeros(4800)]).astype(np.float32)
    assert len(trim_silence(raw, 48000)) == 4800+2*960
    assert len(trim_silence(np.zeros(100), 48000)) == 2400


def test_speaker_priorities_bound_queue_and_barge_in():
    ready, release = threading.Event(), threading.Event()
    class Slow(NullEngine):
        def synthesize(self, text, sample_rate):
            ready.set()
            release.wait(2)
            return super().synthesize(text, sample_rate)
    events = []
    audio = engine()
    speaker = Speaker({'objects': []}, audio, NS(emit=lambda k,d: events.append(d)), engine_factory=Slow)
    try:
        speaker.say('Critical.', priority=2)
        assert ready.wait(1)
        speaker.say('First info.', priority=0)
        speaker.say('Second info.', priority=0)
        speaker.say('Prompt.', priority=1)
        assert [j.text for j in speaker._waiting] == ['Second info.', 'Prompt.']
        speaker.cancel(info_only=True)
        assert speaker._current.text == 'Critical.'
        assert [j.text for j in speaker._waiting] == ['Prompt.']
        release.set()
        end = time.perf_counter()+3
        while speaker.busy and time.perf_counter() < end:
            block(audio)
            speaker.poll()
            time.sleep(.002)
        assert not speaker.busy
        assert any(e['event'] == 'start' and 'first_audio_ms' in e for e in events)
        assert any(e['event'] == 'drop' and e.get('text') == 'First info.' for e in events)
    finally:
        release.set()
        speaker.stop()


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows SAPI only')
def test_sapi_memory_synthesis_and_trim():
    try:
        sapi = SapiEngine({})
    except Exception:
        pytest.skip('No installed SAPI voice available')
    try:
        samples = sapi.synthesize('Found the wallet.', 48000)
        trimmed = trim_silence(samples, 48000)
        assert samples.dtype == np.float32 and np.isfinite(samples).all()
        assert np.max(np.abs(trimmed)) > .01 and .05*48000 < len(trimmed) <= len(samples)
    finally:
        sapi.close()


def test_found_phrase_cache_includes_every_spoken_direction():
    from reach.brain import Brain
    from reach.tts import phrase_texts
    from reach.types import HandObs
    texts=phrase_texts(['wallet'])
    brain=Brain({'brain':{'direction_words':'plain'}})
    spoken=set()
    for x in (100,320,540):
        for y in (80,240,400):
            hand=HandObs(1,True,(320,240),(320,240),(),(0,0),1)
            spoken.add(brain._where(hand,(x,y),(640,480),False))
    assert len(spoken)==9
    for where in spoken:
        assert f'Found the wallet, {where}.' in texts
    assert 'Found the wallet.' in texts


def test_startup_wait_populates_cache_before_live_commands():
    audio=engine()
    speaker=Speaker({'objects':[]},audio,engine_factory=NullEngine)
    try:
        assert speaker.wait_ready(timeout=2)
        assert speaker.stats()['cache_ready']
        assert speaker.stats()['cached_phrases']>=10
        assert not speaker.busy
    finally:
        speaker.stop()


def test_confirmation_acknowledgements_are_preloaded():
    from reach.tts import phrase_texts
    assert {'I heard yes.','I heard no.'}.issubset(phrase_texts([]))
