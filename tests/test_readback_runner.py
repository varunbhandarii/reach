from dataclasses import asdict
import json

import cv2
import numpy as np

from reach.cli import parser,configure
from reach.runner import Runner
from reach.types import Decision,Effect,ReadBack
from test_runner import FakeHand,FakeAudio,video


def test_runner_builds_contact_frame_crop_polls_results_and_logs_exact_effects(tmp_path):
    class Readback:
        def __init__(self,cfg,log):
            self.pending=[]
            self.calls=[]
        def request(self,ident,jpeg,phrase,frame_id):
            image=cv2.imdecode(np.frombuffer(jpeg,np.uint8),1)
            assert image is not None and max(image.shape[:2])==512
            self.calls.append((ident,phrase,frame_id))
            self.pending.append(ReadBack(ident,frame_id,phrase,False,'pen','',.9,100))
        def poll(self):
            rows,self.pending=self.pending,[]
            return rows
        def stats(self):
            return {'cost_usd':.002,'input_tokens':100,'output_tokens':50}
        def stop(self): pass
    class Brain:
        def __init__(self,cfg,objects):
            self.seq=0
            self.received=[]
        def step(self,now,*,hand,results=(),**kwargs):
            self.seq+=1
            self.received.extend(results)
            effects=[Effect('log',key='brain_in',payload={'now':now,'data':{'nested':'retained'},'results':[asdict(r) for r in results]})]
            if self.seq==1:
                effects.append(Effect('readback',payload={'request_id':1,'frame_id':hand.frame_id,'phrase':'wallet','box_xyxy':[10,10,80,80]}))
            decision=Decision(self.seq,now,'CONFIRMING',None,None,None,None,None,None)
            effects.append(Effect('log',key='brain_out',payload={'decision':asdict(decision),'effects':[asdict(e) for e in effects]}))
            return decision,effects
    args=parser().parse_args(['run','--mount','handheld','--source',str(video(tmp_path)),
        '--headless','--mute','--no-voice','--log-dir',str(tmp_path/'runs')])
    runner=Runner(configure(args),args,hand_factory=FakeHand,audio_factory=FakeAudio,
                  brain_factory=Brain,readback_factory=Readback)
    assert runner.run()==0
    assert len(runner.readback.calls)==1
    assert runner.brain.received[0].object_name=='pen'
    rows=[json.loads(line) for line in runner.log.path.read_text().splitlines()]
    assert next(r['data'] for r in rows if r['type']=='brain_in')['data']=={'nested':'retained'}
    outputs=[r['data'] for r in rows if r['type']=='brain_out']
    assert outputs[0]['effects'][0]['payload']['data']=={'nested':'retained'}
    assert outputs[0]['decision']['state']=='CONFIRMING'
    summary=json.loads((runner.run_dir/'summary.json').read_text())
    assert summary['cost']['spark_usd']==.002


def test_choose_exclude_call_perception_and_keep_payload_unchanged():
    from types import SimpleNamespace as NS
    calls=[]
    runner=Runner.__new__(Runner)
    runner.perception=NS(choose=lambda ident:calls.append(('choose',ident)),
        exclude=lambda frame,box:calls.append(('exclude',frame,box)))
    runner._excluded=[]
    payload={'frame_id':7,'box_xyxy':[1,2,3,4]}
    runner._effects([Effect('choose_instance',payload={'sam_id':'second'}),Effect('exclude_instance',payload=payload)],0)
    assert calls==[('choose','second'),('exclude',7,[1,2,3,4])]
    assert runner._excluded==[payload] and payload=={'frame_id':7,'box_xyxy':[1,2,3,4]}


def test_safety_request_is_not_silently_ignored_by_trial_phrase_guard():
    from types import SimpleNamespace as NS
    from reach.types import Command
    runner = Runner.__new__(Runner)
    runner._trial = NS(prepared=True, phrase='wallet')
    runner.cfg = {}; runner.voice = None
    posted = []
    runner.inbox = NS(post=posted.append)
    runner.log = NS(emit=lambda *args: None)
    unsafe = Command('find', 'knife', 'find knife', 1, 0)
    runner.post_command(unsafe)
    assert posted == [unsafe]


def test_missing_scene_image_error_is_delivered_with_readback_enabled(tmp_path):
    class Brain:
        def __init__(self, cfg, objects): self.sent=False; self.received=[]
        def step(self, now, *, hand, results=(), **kwargs):
            self.received.extend(results)
            effects=[]
            if hand and not self.sent:
                self.sent=True
                effects=[Effect('describe', payload={'request_id': 4, 'frame_id': hand.frame_id})]
            return Decision(0, now, 'IDLE', None, None, None, None, None, None), effects
    args=parser().parse_args(['run','--mount','handheld','--source',str(video(tmp_path)),
        '--headless','--mute','--no-voice','--log-dir',str(tmp_path/'runs')])
    runner=Runner(configure(args),args,hand_factory=FakeHand,audio_factory=FakeAudio,brain_factory=Brain)
    runner.camera.get=lambda ident:None
    assert runner.run()==0
    assert any(r.request_id == 4 and r.error == 'image_missing' for r in runner.brain.received)
