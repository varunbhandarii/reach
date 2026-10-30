from pathlib import Path
import wave
import numpy as np
from reach.say_engine import SayEngine, trim_silence
from reach.speech_devices import stereo_warning, mac_builtin_input


def test_say_uses_arguments_and_returns_trimmed_resampled_mono():
    seen=[]
    def run(args,**kwargs):
        seen.append(args)
        assert kwargs['check'] and kwargs['capture_output']
        samples=np.zeros(24000,dtype='<i2');samples[4000:10000]=12000
        with wave.open(args[args.index('-o')+1],'wb') as stream:
            stream.setnchannels(1);stream.setsampwidth(2);stream.setframerate(24000);stream.writeframes(samples.tobytes())
    audio=SayEngine(run=run).synthesize('Checking. $(literal)',48000)
    assert seen[0][-2:]==['--','Checking. $(literal)']
    assert audio.dtype==np.float32 and audio.ndim==1 and len(audio)<24000
    assert not Path(seen[0][2]).exists()


def test_stereo_profile_guard_and_builtin_selection():
    assert stereo_warning({'max_output_channels':2,'default_samplerate':48000}) is None
    assert stereo_warning({'max_output_channels':1,'default_samplerate':16000})
    assert mac_builtin_input([{'name':'AirPods Pro','max_input_channels':1},{'name':'MacBook Air Microphone','max_input_channels':1}])==1


def test_known_airpods_mic_warns_even_with_stereo_metadata():
    warning=stereo_warning({'name':'AirPods Pro','max_output_channels':2,'default_samplerate':48000},input_name='AirPods Pro')
    assert warning and 'metadata' in warning
    assert stereo_warning({'name':'AirPods Pro','max_output_channels':2,'default_samplerate':48000},input_name='MacBook Air Microphone') is None


def test_empty_synthesis_is_not_reported_as_success():
    import pytest
    def run(args,**kwargs):
        with wave.open(args[args.index('-o')+1],'wb') as stream:
            stream.setnchannels(1);stream.setsampwidth(2);stream.setframerate(48000);stream.writeframes(b'')
    with pytest.raises(RuntimeError): SayEngine(run=run).synthesize('Checking.')


def test_prerender_avoids_repeated_synthesis_and_cache_is_independent():
    calls=[]
    def run(args,**kwargs):
        calls.append(args)
        with wave.open(args[args.index('-o')+1],'wb') as stream:
            stream.setnchannels(1);stream.setsampwidth(2);stream.setframerate(48000)
            stream.writeframes(np.full(4800,12000,dtype='<i2').tobytes())
    engine=SayEngine(run=run)
    assert engine.pre_render(['Checking.'])==1
    a=engine.synthesize('Checking.');a[:]=0
    assert engine.synthesize('Checking.').max()>0 and len(calls)==1
