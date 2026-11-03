import json
import time
from types import SimpleNamespace as NS

import httpx2
from openai import AsyncOpenAI
import numpy as np
import pytest

from reach.spark import ReadBackClient,SceneClient
from reach.reference import capture_reference,clean_target_box
from reach.types import HandObs,Instance,TargetObs
from reach.types import Effect


def finish(client):
    end=time.perf_counter()+5
    while client.stats()['busy'] and time.perf_counter()<end:time.sleep(.003)
    assert not client.stats()['busy']
    result,=client.poll()
    return result


def factory(value,bodies):
    async def handler(request):
        bodies.append(json.loads(request.content))
        return httpx2.Response(200,json={'id':'synthetic','object':'response','created_at':0,'status':'completed',
            'output':[{'type':'message','id':'m','role':'assistant','status':'completed',
                'content':[{'type':'output_text','text':json.dumps(value),'annotations':[]}]}]})
    return lambda **kw:AsyncOpenAI(**kw,http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))


def test_pair_identity_overrides_name_disagreement_and_keys_both_images(tmp_path):
    bodies=[]
    value={'same_object':True,'matches_request':False,'object_name':'stylus','hint':'','confidence':.95}
    client=ReadBackClient({'camera':{'source':'file'},'spark':{'cache_dir':str(tmp_path)}},client_factory=factory(value,bodies))
    client.request(1,b'contact','pen',1,reference_jpeg=b'reference')
    row=finish(client)
    assert row.matches is True and row.same_object is True and row.mode=='pair'
    content=bodies[0]['input'][0]['content']
    assert len(content)==3 and content[1]['image_url'].endswith('cmVmZXJlbmNl')
    assert client._identity(b'contact','pen',b'reference')!=client._identity(b'contact','pen',b'other')
    client.request(2,b'contact','pen',2,reference_jpeg=b'reference')
    assert finish(client).matches and len(bodies)==1
    client.request(3,b'contact','pen',3)
    assert finish(client).matches is False and client.stats()['last']['mode']=='single'
    client.stop()


def test_scene_truncation_and_separate_cache(tmp_path):
    bodies=[]
    values={'objects':[{'name':f'item {i}','position':'left'} for i in range(8)]}
    client=SceneClient({'camera':{'source':'file'},'scene':{'cache_dir':str(tmp_path)}},client_factory=factory(values,bodies))
    client.request(1,b'frame',9)
    row=finish(client)
    assert row.frame_id==9 and len(row.objects)==6 and row.error is None
    assert bodies[0]['store'] is False and bodies[0]['text']['format']['strict'] is True
    client.request(2,b'frame',10)
    assert finish(client).frame_id==10 and len(bodies)==1
    client.stop()


def test_clean_reference_rate_limit_and_occlusion():
    camera=NS(get=lambda frame:np.zeros((100,100,3),np.uint8))
    target=TargetObs(1,'wallet',(Instance('a',(20,20,60,60),.9,None),),0,'sam',0)
    clear=HandObs(1,True,(90,90),None,((80,80),(95,95)),(0,0),1)
    first=capture_reference(camera,target,clear,1)
    assert first and first['sam_id']=='a'
    assert capture_reference(camera,target,clear,1.2,first) is first
    covered=HandObs(1,True,(40,40),None,((20,20),(70,70)),(0,0),1)
    assert not clean_target_box(covered,(20,20,60,60))
    assert capture_reference(camera,target,covered,2,None) is None
    assert capture_reference(camera,target,covered,2,first) is first


def test_reference_requires_an_observation_of_the_same_image():
    camera=NS(get=lambda frame:np.zeros((100,100,3),np.uint8))
    target=TargetObs(1,'wallet',(Instance('a',(20,20,60,60),.9,None),),0,'sam',0)
    newer=HandObs(2,False,None,None,(),(0,0),0)
    assert capture_reference(camera,target,newer,1) is None
    assert capture_reference(camera,target,None,1) is None
    observed_absent=HandObs(1,False,None,None,(),(0,0),0)
    assert capture_reference(camera,target,observed_absent,1) is not None


def test_query_id_change_preserves_clean_reference_through_occlusion():
    from dataclasses import replace
    camera=NS(get=lambda frame:np.zeros((100,100,3),np.uint8))
    target=TargetObs(1,'bottle',(Instance('0',(20,20,60,60),.9,None),),0,'sam',0)
    clear=HandObs(1,False,None,None,(),(0,0),0)
    first=capture_reference(camera,target,clear,1)
    refreshed=replace(target,frame_id=2,instances=(replace(target.instances[0],sam_id='1'),))
    covered=HandObs(2,True,(40,40),None,((20,20),(70,70)),(0,0),1)
    assert capture_reference(camera,refreshed,covered,2,first) is first
    assert capture_reference(camera,replace(refreshed,phrase='mug'),covered,2,first) is None
    clear=replace(clear,frame_id=2)
    latest=capture_reference(camera,refreshed,clear,2,first)
    assert latest['frame_id']==2 and latest['sam_id']=='1' and latest is not first


@pytest.mark.parametrize('effect',[Effect('set_phrase',text='pen'),
    Effect('choose_instance',payload={'sam_id':'other'}),
    Effect('exclude_instance',payload={'frame_id':1,'box_xyxy':(1,2,3,4)})])
def test_target_changes_reset_reference(effect):
    from reach.runner import Runner
    runner=Runner.__new__(Runner)
    runner._reference={'jpeg':b'old'}
    runner._click_mode=False
    runner._excluded=[]
    runner.perception=NS(set_phrase=lambda p:None,choose=lambda p:None,exclude=lambda *p:None)
    runner._effects([effect],1)
    assert runner._reference is None


def test_scene_deadline_and_offline_cache_miss(tmp_path):
    async def handler(request):
        import asyncio
        await asyncio.sleep(.2)
        raise AssertionError('Deadline must cancel the call')
    make=lambda **kw:AsyncOpenAI(**kw,http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))
    client=SceneClient({'scene':{'deadline_s':.03}},client_factory=make)
    client.request(1,b'image',1)
    assert finish(client).error=='timeout'
    client.stop()
    client=SceneClient({'scene':{'cache':'read','cache_dir':str(tmp_path)}},client_factory=make)
    client.request(1,b'image',1)
    assert finish(client).error=='cache_miss' and client.stats()['requests']==0
    client.stop()


def test_scene_does_not_queue_behind_readback():
    import asyncio
    import threading
    release=threading.Event()
    async def slow(request):
        while not release.is_set():await asyncio.sleep(.005)
        return httpx2.Response(500,json={})
    make=lambda **kw:AsyncOpenAI(**kw,http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(slow)))
    readback=ReadBackClient({},client_factory=make)
    scene=SceneClient({},client_factory=factory({'objects':[]},[]))
    try:
        readback.request(1,b'contact','pen',1)
        scene.request(1,b'frame',2)
        assert finish(scene).error is None and readback.stats()['busy']
    finally:
        release.set();readback.stop();scene.stop()


def test_scene_rate_limit_is_not_retried():
    calls=[]
    async def handler(request):
        calls.append(request)
        return httpx2.Response(429,headers={'Retry-After':'60'},json={'error':{'message':'rate limit'}})
    make=lambda **kw:AsyncOpenAI(**kw,http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))
    client=SceneClient({},client_factory=make)
    client.request(1,b'image',1)
    assert finish(client).error=='http_429'
    assert not client.request(2,b'image',2)
    assert finish(client).error=='http_429' and len(calls)==1
    client.stop()
