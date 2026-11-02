import asyncio
import json
import time
from types import SimpleNamespace as NS

import cv2
import httpx2 as httpx
import numpy as np
from openai import AsyncOpenAI
import pytest

from reach.spark import ReadBackClient, make_crop, validate_answer


ANSWER = {'matches_request':True,'same_object':True,'object_name':'wallet','confidence':.9,'hint':''}


def factory(handler):
    return lambda **kw: AsyncOpenAI(**kw,http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def finish(client):
    limit = time.perf_counter()+3
    while client.stats()['busy'] and time.perf_counter()<limit:
        time.sleep(.002)
    assert not client.stats()['busy']
    return client.poll()


def test_request_schema_privacy_cache_and_usage(tmp_path):
    bodies=[]
    async def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200,json={'id':'synthetic','object':'response','created_at':0,'status':'completed',
            'output':[{'type':'message','id':'msg','role':'assistant','status':'completed',
                       'content':[{'type':'output_text','text':json.dumps(ANSWER),'annotations':[]}]}],
            'usage':{'input_tokens':100,'output_tokens':50,'total_tokens':150}})
    cfg={'camera':{'source':'file'},'spark':{'cache_dir':str(tmp_path)}}
    events=[]
    client=ReadBackClient(cfg,NS(emit=lambda k,d:events.append((k,d))),client_factory=factory(handler))
    assert client.request(1,b'jpeg','wallet',7)
    result,=finish(client)
    assert result.matches is True and result.target_frame_id==7 and result.error is None
    assert bodies[0]['store'] is False and bodies[0]['model']=='muse-spark-1.3'
    assert bodies[0]['text']['format']['strict'] is True and 'response_format' not in bodies[0]
    assert client.stats()['cost_usd']==pytest.approx((100*1.25+50*4.25)/1e6)
    assert client.request(2,b'jpeg','wallet',8)
    cached,=finish(client)
    assert cached.request_id==2 and cached.target_frame_id==8 and len(bodies)==1
    client.stop()
    read=ReadBackClient({'spark':{'cache':'read','cache_dir':str(tmp_path)}},client_factory=lambda **kw:pytest.fail('network on miss'))
    read.request(3,b'new','wallet',9)
    assert finish(read)[0].error=='cache_miss'
    # Image bytes, phrase, effort, detail and prompt version all affect cache identity.
    original=read._path(read._identity(b'jpeg','wallet'))
    for key,value in [('reasoning_effort','low'),('detail','auto'),('prompt_version','3')]:
        changed=ReadBackClient({'spark':{key:value,'cache_dir':str(tmp_path)}})
        assert changed._path(changed._identity(b'jpeg','wallet'))!=original
    assert read._path(read._identity(b'jpeg','pen'))!=original
    assert next(d for k,d in events if k=='readback_req')['image_bytes']==4


@pytest.mark.parametrize('change',[{'confidence':float('nan')},{'confidence':True},{'confidence':1.1},
    {'matches_request':'yes'},{'hint':'one two three four five six seven'},{'object_name':''},{'extra':1}])
def test_invalid_schema_rejected(change):
    with pytest.raises(ValueError):
        validate_answer({**ANSWER,**change})


@pytest.mark.parametrize('model',['muse-spark-1.3-contributor','muse-spark-1.2-contributor','arbitrary'])
def test_standard_model_only(model):
    with pytest.raises(ValueError):
        ReadBackClient({'spark':{'model':model}})


def test_rate_limit_honors_retry_after_without_retry():
    calls=[]
    async def handler(request):
        calls.append(1)
        return httpx.Response(429,headers={'Retry-After':'10'},json={'error':{'message':'rate limit'}})
    client=ReadBackClient({},client_factory=factory(handler))
    client.request(1,b'jpeg','pen',1)
    assert finish(client)[0].error=='http_429'
    assert not client.request(2,b'jpeg','pen',2)
    assert client.poll()[0].error=='http_429' and len(calls)==1


def test_total_timeout_single_flight_and_sanitized_errors():
    cancelled,started=[],[]
    async def handler(request):
        started.append(True)
        try:
            await asyncio.sleep(1)
        finally:
            cancelled.append(True)
        return httpx.Response(200,json={})
    client=ReadBackClient({'spark':{'deadline_s':.03}},client_factory=factory(handler))
    start=time.perf_counter()
    assert client.request(1,b'jpeg','wallet',1)
    assert not client.request(2,b'jpeg','wallet',2)
    results=finish(client)
    assert {r.error for r in results}=={'busy','timeout'}
    # A short deadline may expire in SDK setup before reaching the transport.
    assert not started or cancelled
    assert time.perf_counter()-start<.8
    client.stop()


def test_crop_union_clamping_green_box_and_resize():
    image=np.zeros((480,640,3),np.uint8)
    jpeg=make_crop(image,(10,20,110,120),((0,0),(200,200)))
    decoded=cv2.imdecode(np.frombuffer(jpeg,np.uint8),cv2.IMREAD_COLOR)
    assert decoded.shape[:2]==(512,512) and image.max()==0
    assert np.count_nonzero((decoded[:,:,1]>100)&(decoded[:,:,0]<80))>100
    edge=make_crop(image,(600,430,640,480),((639,479),))
    assert max(cv2.imdecode(np.frombuffer(edge,np.uint8),1).shape[:2])==512
    with pytest.raises(ValueError):
        make_crop(image,(-1,0,10,10))


def test_incomplete_answer_keeps_billed_usage_and_returns_schema_error():
    async def handler(request):
        return httpx.Response(200,json={'id':'synthetic','object':'response','created_at':0,'status':'incomplete',
            'output':[],'usage':{'input_tokens':100,'output_tokens':600,'total_tokens':700}})
    client=ReadBackClient({},client_factory=factory(handler))
    client.request(1,b'jpeg','wallet',1)
    result,=finish(client)
    assert result.error=='schema' and result.matches is None
    assert client.stats()['output_tokens']==600 and client.stats()['cost_usd']>0
