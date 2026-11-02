"""Energy endpoint estimate from capture-clocked 20 ms PCM subframes."""

import math

import numpy as np


class EnergyEndpoint:
    def __init__(self, rate=24000, floor=.008, hangover_s=.2):
        self.rate, self.floor, self.hangover_s = rate, floor, hangover_s
        self.noise, self.threshold = [], None
        self.last_voice = None
        self.last_time_valid = False
        self.last_capture = None

    def accept(self, pcm, capture_end, time_valid=True):
        if len(pcm)%2:
            self.last_time_valid = False
            return
        samples = np.frombuffer(pcm,dtype='<i2').astype(np.float64)/32768
        if not len(samples):
            return
        self.last_capture = capture_end
        if self.threshold is None:
            self.noise.append(float(np.sqrt(np.mean(samples*samples))))
            if len(self.noise) >= 8:
                self.threshold = max(self.floor,float(np.median(self.noise))*4)
            return
        step = self.rate*20//1000
        for offset in range(0,len(samples),step):
            part = samples[offset:offset+step]
            if float(np.sqrt(np.mean(part*part))) >= self.threshold:
                self.last_voice = capture_end-(len(samples)-offset-len(part))/self.rate
                self.last_time_valid = time_valid

    def end(self, since=None):
        if (self.last_voice is None or not self.last_time_valid or self.last_capture is None
                or self.last_capture-self.last_voice < self.hangover_s-1e-6
                or (since is not None and self.last_voice <= since)):
            return None
        return self.last_voice


def capture_end_time(now, timing, frames, rate=24000):
    try:
        stamp = now-(float(timing.currentTime)-float(timing.inputBufferAdcTime))+frames/rate
        valid = math.isfinite(stamp) and now-1 <= stamp <= now+.04 and timing.inputBufferAdcTime > 0
    except (AttributeError,TypeError,ValueError):
        return now,False
    return (stamp if valid else now),valid
