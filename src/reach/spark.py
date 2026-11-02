"""Single-flight, bounded read-back on standard Spark with a local image cache."""

import asyncio
import base64
from dataclasses import asdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import queue
import tempfile
import threading
import time

import cv2
import numpy as np

from .config import PROJECT_ROOT
from .sam_client import retry_seconds
from .types import ReadBack

SCHEMA = {"type": "object", "additionalProperties": False,
    "required": ["matches_request", "object_name", "confidence", "hint"],
    "properties": {"matches_request": {"type": "boolean"}, "object_name": {"type": "string"},
                   "confidence": {"type": "number"}, "hint": {"type": "string"}}}


def validate_answer(value):
    if not isinstance(value, dict) or set(value) != set(SCHEMA['required']):
        raise ValueError('Invalid read-back fields')
    if (type(value['matches_request']) is not bool or not isinstance(value['object_name'], str)
            or not 1 <= len(value['object_name'].split()) <= 3
            or not isinstance(value['hint'], str) or len(value['hint'].split()) > 6
            or type(value['confidence']) not in (float, int)
            or not math.isfinite(value['confidence']) or not 0 <= value['confidence'] <= 1):
        raise ValueError('Invalid read-back answer')
    return value


def make_crop(image, box, hull=(), long_side=512):
    """Target plus the contact-time hand hull, padded and bounded to the image."""
    if image is None or image.ndim != 3 or long_side < 32:
        raise ValueError('A color image and valid crop size are required')
    height, width = image.shape[:2]
    x0, y0, x1, y1 = map(float, box)
    if not all(math.isfinite(v) for v in (x0, y0, x1, y1)) or not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError('Target box is outside the source image')
    points = np.asarray(((x0,y0),(x1,y1), *hull), dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError('Invalid hand hull')
    lo, hi = points.min(axis=0), points.max(axis=0)
    pad = (hi-lo)*.25
    left, top = np.maximum(0, np.floor(lo-pad)).astype(int)
    right, bottom = np.minimum((width,height), np.ceil(hi+pad)).astype(int)
    canvas = image[top:bottom, left:right].copy()
    cv2.rectangle(canvas, (round(x0)-left,round(y0)-top), (round(x1)-left-1,round(y1)-top-1), (0,255,0), 2)
    scale = long_side/max(canvas.shape[:2])
    canvas = cv2.resize(canvas, (max(1,round(canvas.shape[1]*scale)),max(1,round(canvas.shape[0]*scale))),
                        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    ok, encoded = cv2.imencode('.jpg', canvas, [cv2.IMWRITE_JPEG_QUALITY,80])
    if not ok:
        raise ValueError('Cannot encode read-back crop')
    return encoded.tobytes()


class ReadBackClient:
    def __init__(self, cfg, log=None, *, api_key=None, client_factory=None, clock=time.perf_counter):
        self.cfg, self.log, self._key, self._factory, self._clock = cfg.get('spark', {}), log, api_key, client_factory, clock
        self.model = self.cfg.get('model','muse-spark-1.3')
        if not isinstance(self.model,str) or self.model.endswith('-contributor') or self.model != 'muse-spark-1.3':
            raise ValueError('Read-back requires standard muse-spark-1.3')
        self.effort, self.detail = self.cfg.get('reasoning_effort','minimal'), self.cfg.get('detail','low')
        self.deadline = float(self.cfg.get('deadline_s',6))
        if self.effort not in ('minimal','low') or self.detail not in ('low','auto') or not 0 < self.deadline <= 6:
            raise ValueError('Invalid read-back effort, detail or deadline')
        mode = self.cfg.get('cache','auto')
        self.mode = ('write' if cfg.get('camera',{}).get('source') == 'file' else 'off') if mode == 'auto' else mode
        if self.mode not in ('off','read','write'):
            raise ValueError('Invalid read-back cache mode')
        self.root = Path(self.cfg.get('cache_dir','runs/spark-cache'))
        self.prices = cfg.get('costs',{})
        self._results = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._busy = self._closed = False
        self._thread = None
        self._retry_at = 0.
        self._last = None
        self._calls = self._hits = self._input_tokens = self._output_tokens = 0
        self._cost = 0.

    def _emit(self, kind, data):
        if self.log is not None:
            self.log.emit(kind,data)

    def _identity(self, jpeg, phrase):
        return {'jpeg_sha256':sha256(jpeg).hexdigest(),'phrase':phrase,'model':self.model,
                'effort':self.effort,'detail':self.detail,'prompt_version':str(self.cfg.get('prompt_version','1'))}

    def _path(self, identity):
        return self.root/(sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()+'.json')

    def body(self, jpeg, phrase):
        prompt = (f"The person asked for: '{phrase}'. Their fingers are touching the object inside the green box. "
                  f"Is that object a {phrase}? object_name: what the object in the box is, 1 to 3 words. "
                  "hint: at most 6 words about how it is oriented relative to their hand, or empty. confidence: 0 to 1.")
        return {'model':self.model,'store':False,'reasoning':{'effort':self.effort},
            'max_output_tokens':int(self.cfg.get('max_output_tokens',600)),
            'input':[{'role':'user','content':[{'type':'input_text','text':prompt},
                {'type':'input_image','image_url':'data:image/jpeg;base64,'+base64.b64encode(jpeg).decode('ascii'),'detail':self.detail}]}],
            'text':{'format':{'type':'json_schema','name':'readback','strict':True,'schema':SCHEMA}}}

    async def _call(self, jpeg, phrase, *, count_tokens=False):
        return await asyncio.wait_for(self._perform(jpeg,phrase,count_tokens=count_tokens),self.deadline)

    async def _perform(self, jpeg, phrase, *, count_tokens=False):
        from openai import AsyncOpenAI
        from dotenv import dotenv_values
        key = self._key or os.environ.get('MODEL_API_KEY') or dotenv_values(PROJECT_ROOT/'.env').get('MODEL_API_KEY')
        if not key and self._factory is None:
            raise RuntimeError('Model API key missing')
        factory = self._factory or AsyncOpenAI
        async with factory(api_key=key or 'synthetic',base_url='https://api.meta.ai/v1',max_retries=0,timeout=self.deadline) as client:
            body = self.body(jpeg,phrase)
            if count_tokens:
                # The token-count endpoint accepts the same model/input description.
                response = await client.post('/responses/input_tokens',cast_to=dict,
                    body={k:v for k,v in body.items() if k in ('model','input','text','reasoning')})
                return response
            return await client.responses.create(**body)

    def count_tokens(self, jpeg, phrase):
        return asyncio.run(self._call(jpeg,phrase,count_tokens=True))

    def request(self, request_id, jpeg, phrase, target_frame_id):
        if not jpeg or not isinstance(phrase,str) or not phrase.strip():
            self.fail(request_id, target_frame_id, phrase, 'image_missing')
            return False
        with self._lock:
            if self._closed or self._busy or self._clock() < self._retry_at:
                self.fail(request_id,target_frame_id,phrase,'closed' if self._closed else 'busy' if self._busy else 'http_429')
                return False
            self._busy = True
        self._emit('readback_req',{'request_id':request_id,'phrase':phrase,'frame_id':target_frame_id,
            'image_bytes':len(jpeg),'image_sha256':sha256(jpeg).hexdigest(),'effort':self.effort,'detail':self.detail,
            'model':self.model,'store':False,'cache_mode':self.mode})
        self._thread = threading.Thread(target=self._work,args=(request_id,bytes(jpeg),phrase,target_frame_id),name='readback',daemon=True)
        self._thread.start()
        return True

    def fail(self, request_id, target_frame_id, phrase, error):
        self._finish(ReadBack(request_id,target_frame_id,phrase,None,None,None,None,0.,error),{},0.,False)

    def _finish(self, result, usage, cost, cached):
        self._last = asdict(result)
        self._emit('readback_resp',{**self._last,'status':result.error or 'ok','usage':usage,'cost_usd':cost,'cached':cached})
        self._results.put(result)

    def _work(self, ident, jpeg, phrase, frame_id):
        start, answer, error, usage, cost, cached = self._clock(), None, None, {}, 0., False
        try:
            identity = self._identity(jpeg,phrase)
            if self.mode != 'off':
                try:
                    saved = json.loads(self._path(identity).read_text(encoding='utf-8'))
                    if saved['identity'] == identity:
                        answer = validate_answer(saved['answer'])
                        cached = True
                        self._hits += 1
                except (OSError,ValueError,TypeError,KeyError):
                    pass
            if answer is None:
                if self.mode == 'read':
                    error = 'cache_miss'
                else:
                    self._calls += 1
                    response = asyncio.run(self._call(jpeg,phrase))
                    raw_usage = getattr(response,'usage',None)
                    usage = raw_usage.model_dump() if hasattr(raw_usage,'model_dump') else dict(raw_usage or {})
                    inp, out = max(0,int(usage.get('input_tokens',0))), max(0,int(usage.get('output_tokens',0)))
                    self._input_tokens += inp
                    self._output_tokens += out
                    cost = (inp*float(self.prices.get('spark_input_per_million',1.25))+out*float(self.prices.get('spark_output_per_million',4.25)))/1e6
                    self._cost += cost
                    if getattr(response,'status','completed') != 'completed':
                        raise ValueError('Incomplete response')
                    answer = validate_answer(json.loads(response.output_text))
                    if self.mode == 'write':
                        self.root.mkdir(parents=True,exist_ok=True)
                        temporary = None
                        try:
                            with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=self.root,delete=False,suffix='.tmp') as stream:
                                temporary = Path(stream.name)
                                json.dump({'identity':identity,'answer':answer},stream,allow_nan=False)
                            temporary.replace(self._path(identity))
                        finally:
                            if temporary:
                                temporary.unlink(missing_ok=True)
        except (ValueError,KeyError,TypeError):
            error = 'schema'
        except TimeoutError:
            error = 'timeout'
        except Exception as exc:
            status = getattr(exc,'status_code',None)
            if status == 429:
                headers = getattr(getattr(exc,'response',None),'headers',{})
                self._retry_at = self._clock()+retry_seconds(headers.get('Retry-After'))
                error = 'http_429'
            elif status is not None:
                error = f'http_{int(status)}'
            else:
                error = 'timeout' if type(exc).__name__ == 'APITimeoutError' else 'network'
        finally:
            duration = (self._clock()-start)*1000
            result = ReadBack(ident,frame_id,phrase,answer['matches_request'] if answer and not error else None,
                answer['object_name'] if answer and not error else None, answer['hint'] if answer and not error else None,
                float(answer['confidence']) if answer and not error else None,duration,error)
            try:
                self._finish(result,usage,cost,cached)
            finally:
                with self._lock:
                    self._busy = False

    def poll(self):
        results = []
        while True:
            try:
                results.append(self._results.get_nowait())
            except queue.Empty:
                return results

    def stats(self):
        return {'busy':self._busy,'requests':self._calls,'cache_hits':self._hits,'input_tokens':self._input_tokens,
                'output_tokens':self._output_tokens,'cost_usd':self._cost,'last':self._last}

    def stop(self):
        self._closed = True
        if self._thread:
            self._thread.join(timeout=self.deadline+2)
            if self._thread.is_alive():
                raise RuntimeError('Read-back worker did not stop')
