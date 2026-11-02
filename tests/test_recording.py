import wave
import threading
from types import SimpleNamespace as NS

import cv2
import numpy as np

from reach.audio import CueEngine, Tone
from reach.recording import AudioRecorder, OverlayRecorder
from reach.types import CueParams


def test_recording_is_exact_rendered_stereo_mix_and_never_mutates_output(tmp_path):
    events=[]
    engine=CueEngine({})
    engine._tone=Tone(48000,256)
    engine._recorder=AudioRecorder(tmp_path/'output.wav',48000,256,NS(emit=lambda k,d:events.append((k,d))))
    engine.set(CueParams(.3,440,4,False,.1,.2))
    engine.speak(np.full(500,.02,np.float32),1)
    engine.earcon('contact')
    expected=[]
    for i in range(10):
        if i==7:
            engine.set_muted(True)
        block=np.empty((256,2),np.float32)
        engine._callback(block,256,NS(outputBufferDacTime=10,currentTime=10),NS(output_underflow=False))
        expected.append((block*32767).astype('<i2'))
    engine.stop()
    with wave.open(str(tmp_path/'output.wav'),'rb') as stream:
        assert (stream.getnchannels(),stream.getframerate(),stream.getnframes())==(2,48000,2560)
        actual=np.frombuffer(stream.readframes(2560),dtype='<i2').reshape(-1,2)
    np.testing.assert_array_equal(actual,np.concatenate(expected))
    assert not engine.recording_stats()['dropped_samples']
    assert any(d['stream']=='audio' and d['event']=='start' for _,d in events)


def test_overlay_constant_rate_repeats_canvas_during_gaps(tmp_path):
    path=tmp_path/'overlay.avi'
    writer=OverlayRecorder(path)
    red=np.zeros((48,64,3),np.uint8)
    red[:,:,2]=255
    writer.submit(red,10.)
    writer.tick(10.5)
    writer.submit(np.zeros_like(red),10.6)
    writer.stop(11.)
    assert writer.stats()['frames']==30
    cap=cv2.VideoCapture(str(path))
    assert cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT)==30
    frames=[]
    while True:
        ok,image=cap.read()
        if not ok:
            break
        frames.append(image)
    cap.release()
    assert len(frames)==30 and frames[17][:,:,2].mean()>250 and frames[18].max()<3


def test_slow_disk_causes_logged_drop_and_silent_gap_without_blocking_producer(tmp_path,monkeypatch):
    entered,released=threading.Event(),threading.Event()
    original=wave.open
    class SlowWave:
        def __init__(self,path,mode):
            self.real=original(path,mode)
        def __getattr__(self,name):
            return getattr(self.real,name)
        def writeframesraw(self,data):
            entered.set()
            assert released.wait(2)
            self.real.writeframesraw(data)
    monkeypatch.setattr('reach.recording.wave.open',SlowWave)
    events=[]
    recorder=AudioRecorder(tmp_path/'slow.wav',48000,256,NS(emit=lambda k,d:events.append(d)),ring_s=.001)
    block=np.full((256,2),.25,np.float32)
    try:
        recorder.accept(block,10)
        assert entered.wait(2)
        recorder.accept(block,10.01)
        recorder.accept(block,10.02)
        assert recorder.stats()['dropped_samples']==256
    finally:
        released.set()
        recorder.stop()
    with original(str(tmp_path/'slow.wav'),'rb') as stream:
        actual=np.frombuffer(stream.readframes(768),'<i2').reshape(-1,2)
    assert len(actual)==768 and not actual[512:].any()
    assert any(row['event']=='overrun' for row in events)
