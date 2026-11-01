"""Local overlay and exact output-mix recording with bounded callback work."""

import math
from pathlib import Path
import threading
import time
import wave

import cv2
import numpy as np


class AudioRecorder:
    """One producer, one consumer. A full ring drops input rather than blocking."""
    def __init__(self,path,rate,blocksize,log=None,ring_s=4):
        self.path,self.rate,self.blocksize,self.log = Path(path),rate,blocksize,log
        self._slots = max(2,math.ceil(rate*ring_s/blocksize))
        self._ring = np.empty((self._slots,blocksize,2),np.float32)
        self._counts = [0]*self._slots
        self._offsets = [0]*self._slots
        self._write = self._read = self._received = self._dropped = self._written = 0
        self._first = None
        self._error = None
        self._stop = threading.Event()
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self._wave = wave.open(str(self.path),'wb')
        self._wave.setnchannels(2)
        self._wave.setsampwidth(2)
        self._wave.setframerate(rate)
        self._thread = threading.Thread(target=self._drain,name='output-recording',daemon=True)
        self._thread.start()

    def accept(self,block,stamp):
        if self._stop.is_set() or self._error:
            return
        if self._first is None:
            self._first = stamp
        offset = self._received
        self._received += len(block)
        if len(block)>self.blocksize or self._write-self._read>=self._slots:
            self._dropped += len(block)
            return
        slot = self._write%self._slots
        np.copyto(self._ring[slot,:len(block)],block)
        self._counts[slot],self._offsets[slot] = len(block),offset
        self._write += 1

    def _emit(self,data):
        if self.log is not None:
            self.log.emit('record',data)

    def _silence(self,count):
        while count>0:
            n = min(count,self.blocksize)
            self._wave.writeframesraw(bytes(n*4))
            self._written += n
            count -= n

    def _drain(self):
        announced,noted = False,0
        try:
            while not self._stop.is_set() or self._read<self._write:
                if self._first is not None and not announced:
                    self._emit({'event':'start','stream':'audio','file':self.path.name,
                                't_mono':self._first,'samplerate':self.rate})
                    announced = True
                if self._dropped != noted:
                    self._emit({'event':'overrun','stream':'audio','dropped_samples':self._dropped-noted})
                    noted = self._dropped
                if self._read>=self._write:
                    self._stop.wait(.005)
                    continue
                slot = self._read%self._slots
                count,offset = self._counts[slot],self._offsets[slot]
                # Keep ownership until this copy/convert/write completes.
                data = (np.clip(self._ring[slot,:count],-1,1)*32767).astype('<i2').tobytes()
                self._silence(max(0,offset-self._written))
                self._wave.writeframesraw(data)
                self._written += count
                self._read += 1
            self._silence(max(0,self._received-self._written))
            if self._dropped != noted:
                self._emit({'event':'overrun','stream':'audio','dropped_samples':self._dropped-noted})
        except Exception as exc:
            self._error = f'output recording failed ({type(exc).__name__})'
        finally:
            self._wave.close()

    def stats(self):
        return {'file':self.path.name,'samples_received':self._received,'samples_written':self._written,
                'dropped_samples':self._dropped,'first_sample_mono':self._first,'samplerate':self.rate,'error':self._error}

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            raise RuntimeError('Output recording writer did not stop')


class OverlayRecorder:
    def __init__(self,path,fps=30,log=None,*,writer_factory=cv2.VideoWriter):
        self.path,self.fps,self.log,self._factory = Path(path),fps,log,writer_factory
        if not 0 < fps <= 120:
            raise ValueError('Invalid recording frame rate')
        self._writer = self._image = self._first = None
        self._frames = 0
        self._closed = False

    def _advance(self,now):
        count = max(0,math.ceil((now-self._first)*self.fps-1e-8))
        while self._frames<count:
            self._writer.write(self._image)
            self._frames += 1

    def submit(self,image,now):
        if self._closed:
            raise RuntimeError('Overlay recording already closed')
        if self._writer is None:
            self.path.parent.mkdir(parents=True,exist_ok=True)
            self._first = float(now)
            size = (image.shape[1],image.shape[0])
            codec = 'MJPG' if self.path.suffix.lower()=='.avi' else 'mp4v'
            self._writer = self._factory(str(self.path),cv2.VideoWriter_fourcc(*codec),self.fps,size)
            if not self._writer.isOpened():
                self._writer.release()
                self._writer = None
                raise RuntimeError('Cannot open overlay recording writer')
            self._image = image.copy()
            if self.log is not None:
                self.log.emit('record',{'event':'start','stream':'video','file':self.path.name,
                    't_mono':self._first,'fps':self.fps,'codec':codec,'width':size[0],'height':size[1]})
        self._advance(now)
        self._image = cv2.resize(image,(self._image.shape[1],self._image.shape[0])).copy()

    def tick(self,now):
        if self._writer is not None and not self._closed:
            self._advance(now)

    def stop(self,now=None):
        if not self._closed:
            if self._writer is not None:
                self._advance(time.perf_counter() if now is None else now)
                self._writer.release()
            self._closed = True

    def stats(self):
        return {'file':self.path.name,'fps':self.fps,'frames':self._frames,'first_frame_mono':self._first}
