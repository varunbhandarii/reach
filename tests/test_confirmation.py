from dataclasses import asdict, replace
import json

import numpy as np
import pytest

from reach.brain import Brain
from reach.contact import ContactState
from reach.grammar import parse
from reach.types import Command, HandObs, Instance, ReadBack, TargetObs


def hand(i=1, point=(150,150)):
    return HandObs(i,True,point,None,(),(0,0),1)


def target(two=False):
    items=[Instance('a',(100,100,200,200),.95,None)]
    if two: items.append(Instance('b',(400,100,500,200),.9,None))
    return TargetObs(10,'wallet',tuple(items),0,'sam',0)


def command(kind,phrase=None):
    return Command(kind,phrase,'script',None,0,'key')


class Contact:
    def __init__(self,fire=True): self.fire=fire
    def reset(self): pass
    def update(self,now,h,t):
        return ContactState(fired=self.fire,inside=True,still=self.fire) if h and t else ContactState()


def confirmation(two=False,cfg=None):
    brain=Brain(cfg);brain.contact=Contact()
    brain.step(0,commands=[command('find','wallet')])
    d,e=brain.step(.1,hand=hand(),target=target(two))
    if two: d,e=brain.step(.2,hand=hand(),target=target(two),commands=[command('choose','left')])
    assert d.state=='CONFIRMING'
    request=next(e.payload for e in e if e.kind=='readback')
    return brain,request


def answer(request,match=True,**changes):
    return ReadBack(**({'request_id':request['request_id'],'target_frame_id':request['frame_id'],
        'phrase':'wallet','matches':match,'object_name':'wallet' if match else 'pen',
        'hint':'','confidence':.9,'latency_ms':100}|changes))


def spoken(effects): return [e.text for e in effects if e.kind=='speak']


def test_contact_is_immediate_and_correct_match_then_hint_are_ordered():
    b,r=confirmation()
    d,e=b.step(.3,results=[answer(r,hint='On your left')])
    assert d.state=='DONE' and d.cue is None
    assert spoken(e)==['You have the wallet.','On your left']
    assert [e.priority for e in e if e.kind=='speak']==[2,0]
    assert e[0].key=='brain_in' and e[-1].key=='brain_out'


def test_wait_boundary_fallback_once_then_late_match_is_silent():
    b,r=confirmation()
    assert spoken(b.step(2.5999)[1])==[]
    assert spoken(b.step(2.6)[1])==["You're touching the wallet."]
    assert spoken(b.step(2.7)[1])==[]
    d,e=b.step(3,results=[answer(r)])
    assert d.state=='DONE' and spoken(e)==[]


def test_late_mismatch_excludes_and_searches_without_reusing_rejected_frame():
    b,r=confirmation();b.step(2.6)
    d,e=b.step(3,target=target(),hand=hand(2),results=[answer(r,False)])
    assert d.state=='SEARCHING' and d.cue is None
    assert spoken(e)==["That's not the wallet.",'It looks like a pen.']
    assert next(x for x in e if x.kind=='exclude_instance').payload=={'frame_id':10,'box_xyxy':[100,100,200,200]}
    assert any(x.kind=='set_phrase' and x.text=='wallet' for x in e)


def test_mismatch_selects_alternative_then_second_mismatch_stops():
    b,r=confirmation(True)
    d,e=b.step(.3,results=[answer(r,False)])
    assert d.state=='GUIDING' and d.cue is None
    assert next(x.payload['sam_id'] for x in e if x.kind=='choose_instance')=='b'
    chosen=replace(target(True),chosen=1)
    d,e=b.step(.4,hand=hand(3),target=chosen)
    request=next(x.payload for x in e if x.kind=='readback')
    d,e=b.step(.5,results=[answer(request,False)])
    assert d.state=='IDLE' and b.phrase is None
    assert spoken(e)==["I can't confirm the wallet."]


@pytest.mark.parametrize('error',['timeout','schema','http_429'])
def test_readback_error_finishes_with_fallback(error):
    b,r=confirmation()
    d,e=b.step(.3,results=[answer(r,matches=None,confidence=None,error=error)])
    assert d.state=='DONE' and spoken(e)==["You're touching the wallet."]


def test_deadline_does_not_accept_expired_mismatch():
    b,r=confirmation()
    d,e=b.step(6.1001,results=[answer(r,False)])
    assert d.state=='DONE' and spoken(e)==["You're touching the wallet."]
    assert not any(x.kind=='exclude_instance' for x in e)


def test_stop_ignores_same_step_and_later_results_then_ids_are_not_reused():
    b,r=confirmation()
    d,e=b.step(.3,commands=[command('stop')],results=[answer(r,False)])
    assert d.state=='IDLE' and not any(x.kind=='exclude_instance' for x in e)
    assert spoken(b.step(.4,results=[answer(r)])[1])==[]
    b.step(.5,commands=[command('find','wallet')])
    d,e=b.step(.6,hand=hand(4),target=target())
    r2=next(x.payload for x in e if x.kind=='readback')
    assert r2['request_id']>r['request_id']
    d,e=b.step(.7,results=[answer(r)])
    assert d.state=='CONFIRMING' and spoken(e)==[]


@pytest.mark.parametrize('changes',[{'request_id':99},{'target_frame_id':99},{'phrase':'pen'}])
def test_result_identity_must_match_every_field(changes):
    b,r=confirmation()
    d,e=b.step(.3,results=[answer(r,**changes)])
    assert d.state=='CONFIRMING' and spoken(e)==[]
    assert any(x.key=='readback_ignore' for x in e)


def disambiguation():
    b=Brain({'contact':{'enabled':False}})
    b.step(0,commands=[command('find','wallet')])
    d,e=b.step(.1,target=target(True))
    assert d.state=='DISAMBIGUATING' and d.cue is None
    return b


def test_spatial_choice_and_single_instance_are_safe():
    b=disambiguation()
    d,e=b.step(.2,target=target(True),hand=hand(),commands=[command('choose','right')])
    assert d.state=='GUIDING' and next(x.payload['sam_id'] for x in e if x.kind=='choose_instance')=='b'
    assert d.dx>0
    b=disambiguation()
    d,e=b.step(.2,target=target(),hand=hand(),commands=[command('choose','right')])
    assert d.state=='GUIDING'


def test_timeout_exact_boundary_and_invalid_choice_prompt_once():
    b=disambiguation()
    assert spoken(b.step(.2,commands=[command('choose','blue')])[1])==['Say left or right.']
    assert spoken(b.step(.3,commands=[command('choose','blue')])[1])==[]
    assert b.step(8.0999)[0].state=='DISAMBIGUATING'
    d,e=b.step(8.1,hand=hand(point=(450,150)),target=target(True))
    assert d.state=='GUIDING' and next(x.payload['sam_id'] for x in e if x.kind=='choose_instance')=='b'
    assert spoken(e)==['Guiding to the nearest one.']


def test_ahead_prompt_has_matching_grammar_and_flipped_choices():
    assert (parse('ahead').kind,parse('ahead').phrase)==('choose','farther')
    b=disambiguation()
    d,e=b.step(.2,target=target(True),commands=[command('choose','left')],flipped=True)
    assert next(x.payload['sam_id'] for x in e if x.kind=='choose_instance')=='b'


def test_optional_touch_question_and_yes_confirmation():
    b=Brain({'brain':{'ask_touch_enabled':True}});b.contact=Contact(False)
    b.step(0,commands=[command('find','wallet')])
    b.step(.1,hand=hand(),target=target())
    assert spoken(b.step(1.6,hand=hand(2),target=target())[1])==['Are you touching it?']
    d,e=b.step(1.7,hand=hand(3),target=target(),commands=[command('yes')])
    assert d.state=='CONFIRMING' and any(x.kind=='readback' for x in e)
    assert any(x.key=='state' and x.payload['reason']=='user_yes' for x in e)


def test_plain_snapshot_has_masks_and_is_deterministic_with_results():
    def run():
        b,r=confirmation()
        observation=replace(hand(2),score=np.float32(.9))
        d,e=b.step(.3,hand=observation,target=target(),results=[answer(r)])
        return json.loads(json.dumps({'decision':asdict(d),'effects':[asdict(x) for x in e]},allow_nan=False))
    assert run()==run()
    assert run()['effects'][0]['payload']['results'][0]['request_id']==1


def test_uncertain_response_never_claims_confirmed_identity():
    b,r=confirmation()
    d,e=b.step(.3,results=[answer(r,confidence=.4)])
    assert d.state=='DONE' and 'You have the wallet.' not in spoken(e)



def test_word_without_spatial_separation_does_not_select_arbitrarily():
    b=disambiguation()
    d,e=b.step(.2,target=target(True),commands=[command('choose','closer')])
    assert d.state=='DISAMBIGUATING' and not any(x.kind=='choose_instance' for x in e)


def test_match_exactly_at_wait_boundary_is_confirmed():
    b,r=confirmation()
    d,e=b.step(2.6,results=[answer(r)])
    assert d.state=='DONE' and spoken(e)==['You have the wallet.']


def test_malformed_match_flag_cannot_claim_success():
    b,r=confirmation()
    d,e=b.step(.3,results=[answer(r,matches='true')])
    assert d.state=='DONE' and spoken(e)==["You're touching the wallet."]
