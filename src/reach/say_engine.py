"""Local macOS speech-to-samples engine; never plays through a second output."""

import subprocess
import tempfile
from pathlib import Path
import wave
import numpy as np


def trim_silence(samples, sample_rate, db=-40):
    samples=np.asarray(samples,dtype=np.float32)
    if samples.ndim!=1 or not np.isfinite(samples).all():
        raise ValueError('speech samples must be finite mono audio')
    indices=np.flatnonzero(np.abs(samples)>10**(db/20))
    if not len(indices):return samples[:max(1,round(sample_rate*.05))].copy()
    padding=round(sample_rate*.02)
    return samples[max(0,indices[0]-padding):min(len(samples),indices[-1]+padding+1)].copy()


class SayEngine:
    name='say'

    def __init__(self, cfg=None, *, run=subprocess.run):
        self.cfg=(cfg or {}).get('tts',cfg or {})
        self.voice=self.cfg.get('voice')
        self.rate=int(self.cfg.get('rate',210))
        self._run=run
        self._cache={}
        if not 50<=self.rate<=600:raise ValueError('speech rate must be 50 to 600 words per minute')

    def synthesize(self, text, sample_rate=48000):
        if not isinstance(text,str) or not text.strip():raise ValueError('speech text is empty')
        sample_rate=int(sample_rate)
        if not 8000<=sample_rate<=192000:raise ValueError('unsupported output sample rate')
        key=(self.voice,self.rate,sample_rate,float(self.cfg.get('trim_db',-40)),text)
        if key in self._cache:
            return self._cache[key].copy()
        with tempfile.TemporaryDirectory(prefix='reach-speech-') as directory:
            path=Path(directory)/'speech.wav'
            args=['/usr/bin/say','-o',str(path),'--file-format=WAVE',f'--data-format=LEI16@{sample_rate}','-r',str(self.rate)]
            if self.voice:args+=['-v',str(self.voice)]
            args+=['--',text]
            try:
                self._run(args,check=True,capture_output=True,timeout=30)
                with wave.open(str(path),'rb') as stream:
                    if stream.getsampwidth()!=2 or stream.getcomptype()!='NONE':
                        raise ValueError('say did not return PCM16')
                    channels=stream.getnchannels();rate=stream.getframerate()
                    data=np.frombuffer(stream.readframes(stream.getnframes()),dtype='<i2').astype(np.float32)/32768
                if channels not in (1,2):raise ValueError('unsupported speech channel count')
                if channels==2:data=data.reshape(-1,2).mean(axis=1)
                if rate!=sample_rate and len(data):
                    length=max(1,round(len(data)*sample_rate/rate))
                    data=np.interp(np.arange(length)*rate/sample_rate,np.arange(len(data)),data).astype(np.float32)
                if not len(data) or not np.any(np.abs(data)>10**(float(self.cfg.get('trim_db',-40))/20)):
                    raise ValueError('speech output is empty or silent')
                trimmed=trim_silence(data,sample_rate,float(self.cfg.get('trim_db',-40)))
                self._cache[key]=trimmed
                return trimmed.copy()
            except (OSError,ValueError,wave.Error,subprocess.SubprocessError):
                raise RuntimeError('macOS speech synthesis failed; check the installed voice and say command') from None

    def pre_render(self,texts,sample_rate=48000):
        for text in texts:
            self.synthesize(text,sample_rate)
        return len(self._cache)

    def render(self,text,sample_rate=48000):
        return self.synthesize(text,sample_rate)
