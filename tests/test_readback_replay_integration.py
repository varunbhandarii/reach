"""The real reducer, contact rule, SDK adapter and runner replay together."""

import json

import cv2
import httpx2
import numpy as np
from openai import AsyncOpenAI
import pytest

from reach.cli import parser,configure
from reach.runner import Runner
from reach.spark import ReadBackClient
from reach.types import Instance,TargetObs
from test_runner import FakeHand,FakeAudio


def test_voice_contact_readback_runner_replays_exactly(tmp_path):
    replay=pytest.importorskip('reach.replay',reason='Requires the shared confirming reducer and replay implementation')
    from reach.session_io import make_command
    path=tmp_path/'table.avi'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),20,(160,120))
    assert writer.isOpened()
    for i in range(45):
        image=np.zeros((120,160,3),np.uint8)
        image[i%100,100]=255
        writer.write(image)
    writer.release()
    requests=[]
    async def handle(request):
        requests.append(json.loads(request.content))
        return httpx2.Response(200,json={'id':'synthetic','object':'response','created_at':0,'status':'completed',
            'output':[{'type':'message','id':'msg','role':'assistant','status':'completed','content':[
                {'type':'output_text','annotations':[],'text':json.dumps({'matches_request':True,
                    'object_name':'wallet','same_object':True,'hint':'','confidence':.9})}]}],
            'usage':{'input_tokens':100,'output_tokens':50,'total_tokens':150}})
    def client(cfg,log):
        cfg={**cfg,'spark':{**cfg['spark'],'cache':'off'}}
        return ReadBackClient(cfg,log,client_factory=lambda **kw:AsyncOpenAI(**kw,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handle))))
    class Perception:
        def __init__(self,cfg,camera,log): self.camera,self.phrase=camera,None
        def start(self): pass
        def stop(self): pass
        def set_phrase(self,phrase): self.phrase=phrase
        def set_state(self,state): pass
        def force_query(self): pass
        def on_hand(self,hand): pass
        def stats(self): return {'error':None,'stale':False,'images_sent':0}
        def latest(self):
            value=self.camera.latest()
            if value is None or self.phrase is None: return None
            frame,_=value
            return TargetObs(frame.id,self.phrase,(Instance('wallet',(10,10,80,80),.9,None),),0,'tracker',0)
    class Voice:
        def __init__(self,cfg,log,post,**kwargs): self.post=post
        def start(self): self.post(make_command('find','wallet',source='voice',transcript='synthetic find wallet'))
        def stop(self): pass
        def stats(self): return {'connected':True,'stereo_ok':True}
    args=parser().parse_args(['run','--mount','handheld','--source',str(path),'--headless','--mute','--voice',
                            '--log-dir',str(tmp_path/'runs')])
    cfg=configure(args)
    runner=Runner(cfg,args,hand_factory=FakeHand,audio_factory=FakeAudio,
        perception_factory=Perception,voice_factory=Voice,readback_factory=client)
    assert runner.run()==0
    rows=[json.loads(s) for s in (runner.run_dir/'events.jsonl').read_text().splitlines()]
    assert len(requests)==1 and requests[0]['store'] is False
    assert any(r['type']=='state' and r['data'].get('reason')=='readback_match' for r in rows)
    result=replay.replay_session(runner.run_dir)
    assert result['status']=='ok' and result['readbacks']==1
    inputs=sum(r['type']=='brain_in' for r in rows)
    assert result['steps']==inputs==sum(r['type']=='brain_out' for r in rows)
