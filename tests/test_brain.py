from dataclasses import asdict
import pytest
from reach.brain import Brain
from reach.types import HandObs,TargetObs,Instance,Command


def h(i=1, present=True, point=(40,40)):
    return HandObs(i,present,point,None,(),(0,0),1)


def t(phrase='wallet', box=(100,100,200,200)):
    return TargetObs(1,phrase,(Instance('1',box,.9,None),),0,'sam',0)


def cmd(kind,phrase=None):
    return Command(kind,phrase,'script',1,0,'key')


def step(b,now,commands=(),hand=None,target=None,**stats):
    return b.step(now,commands=commands,hand=hand,target=target,perception_stats=stats)


def keys(effects,kind):return [e.key for e in effects if e.kind==kind]


def guiding():
    b=Brain();step(b,0,[cmd('find','wallet')]);step(b,.1,hand=h(),target=t());return b


def test_find_found_and_semantic_earcons():
    b=Brain();d,e=step(b,0,[cmd('find','wallet')])
    assert d.state=='SEARCHING' and keys(e,'perception_state')==['SEARCHING']
    d,e=step(b,.2,hand=h(),target=t(),ambiguous=True)
    assert d.state=='GUIDING' and d.cue and keys(e,'earcon')==['found']
    assert 'more than one' in d.speech
    d,e=step(b,.3,hand=h(2),target=t(),ambiguous=True)
    assert d.speech is None and keys(e,'earcon')==[]


def test_exact_search_prompt_and_timeout_boundaries():
    b=Brain();step(b,0,[cmd('find','wallet')])
    assert step(b,3.999)[0].speech is None
    assert 'Move the phone' in step(b,4)[0].speech
    assert step(b,9.999)[0].speech is None
    assert 'Move the phone' in step(b,10)[0].speech
    assert step(b,20)[0].state=='IDLE'


def test_hand_loss_silence_and_return_no_target_earcons():
    b=guiding();d,e=step(b,.2,hand=h(present=False),target=t())
    assert d.state=='HAND_LOST' and d.cue is None and keys(e,'earcon')==[]
    assert step(b,1.699,target=t())[0].speech is None
    assert 'Bring your hand' in step(b,1.7,target=t())[0].speech
    d,e=step(b,1.8,hand=h(4),target=t())
    assert d.state=='GUIDING' and keys(e,'earcon')==[]


def test_target_loss_thresholds_reacquisition_and_give_up():
    b=guiding();assert step(b,1,hand=h())[0].cue is None
    assert step(b,1.999,hand=h())[0].state=='GUIDING'
    d,e=step(b,2,hand=h());assert d.state=='TARGET_LOST' and keys(e,'earcon')==['lost']
    assert 'Hold the phone' in step(b,3,hand=h())[0].speech
    d,e=step(b,4,hand=h(),target=t());assert d.state=='GUIDING' and keys(e,'earcon')==['reacquired']
    step(b,5,hand=h());step(b,6,hand=h())
    assert step(b,20,hand=h())[0].state=='IDLE'


def test_same_find_rechecks_new_find_discards_old_target():
    b=guiding();d,e=step(b,.2,[cmd('find','wallet')],h(),t())
    assert d.state=='GUIDING' and 'force_query' in [x.kind for x in e]
    d,e=step(b,.3,[cmd('find','pen')],h(),t())
    assert d.state=='SEARCHING' and d.cue is None


@pytest.mark.parametrize('kind,key',[('stop','stopped'),('cancel','stopped'),('start_over','ready')])
def test_interrupts_speech_and_stop_dominates_batch(kind,key):
    b=guiding();d,e=step(b,.2,[cmd('find','apple'),cmd(kind)],h(),t())
    assert d.state=='IDLE' and d.cue is None and keys(e,'speak')==[key]
    assert e[0].kind=='cancel_speech' and not any(x.kind=='set_phrase' and x.text for x in e)


def test_veto_repeat_no_choose_and_offline_network_notices():
    b=Brain();step(b,0,[cmd('find','wallet')]);d,e=step(b,1.2,[cmd('no')])
    assert d.state=='IDLE' and keys(e,'speak')==['cancelled']
    assert keys(step(b,1.3,[cmd('repeat')])[1],'speak')==['cancelled']
    d,e=step(b,1.4,[cmd('yes'),cmd('choose','right')],voice_online=False)
    assert keys(e,'speak')==['yes_ack','voice_off'] and d.state=='IDLE'
    assert step(b,1.5,voice_online=False)[0].speech is None
    step(b,2,[cmd('find','wallet')])
    assert 'network' in keys(step(b,3,consecutive_errors=2)[1],'speak')
    assert 'network' not in keys(step(b,17.999,consecutive_errors=2)[1],'speak')
    assert 'network' in keys(step(b,18,consecutive_errors=2)[1],'speak')


def test_contact_only_guiding_then_done_and_idle():
    b=guiding()
    for i in range(1,8):d,e=step(b,.1+i*.05,hand=h(i,(True),(150,150)),target=t())
    assert d.state=='CONTACT' and keys(e,'earcon')==['contact'] and d.cue is None
    assert keys(e,'perception_state')==['IDLE']
    d,e=b.step(.6,hand=h(9),target=t(),speaking=True)
    assert d.state=='CONTACT'
    assert b.step(2.5,speaking=True)[0].state=='DONE'
    assert b.step(3.5)[0].state=='IDLE'
    b=guiding();step(b,.2,hand=h(present=False),target=t())
    assert step(b,1,hand=h(present=False),target=t())[0].state=='HAND_LOST'


def test_click_and_deterministic_sequences():
    def run():
        b=Brain();out=[]
        for i in range(5):
            d,e=step(b,i*.03,hand=h(i),target=t('',(100,100,200,200)))
            out.append((asdict(d),[asdict(x) for x in e]))
        return out
    assert run()==run()
    assert run()[0][0]['state']=='GUIDING'


@pytest.mark.parametrize('state',['SEARCHING','GUIDING','TARGET_LOST'])
def test_recheck_in_each_active_state(state):
    b=Brain();b.state=state;b.phrase='wallet';b._search_start=0
    d,e=step(b,.1,[cmd('recheck')],hand=h(),target=t())
    assert 'force_query' in [x.kind for x in e] and 'checking' in keys(e,'speak')


@pytest.mark.parametrize('elapsed,key',[(1.2,'cancelled'),(1.20001,'stopped')])
def test_cancel_veto_boundary(elapsed,key):
    b=Brain();step(b,0,[cmd('find','wallet')])
    assert keys(step(b,elapsed,[cmd('cancel')])[1],'speak')==[key]


def test_contact_speech_completion_and_passive_commands():
    b=Brain();b.state='CONTACT';b.phrase='wallet'
    d,e=b.step(.2,speaking=False)
    assert d.state=='DONE' and any(x.kind=='set_phrase' and x.text is None for x in e)
    b=Brain();d,e=step(b,0,[cmd('choose','left'),cmd('unknown')])
    assert d.state=='IDLE' and e==[]


def test_disabled_contact_keeps_guiding_without_chimes():
    b=Brain({'contact': {'enabled': False}})
    step(b,0,[cmd('find','wallet')])
    for i in range(1,20):
        d,e=step(b,i*.05,hand=h(i,point=(150,150)),target=t())
        assert d.state=='GUIDING' and d.cue is not None
        assert 'contact' not in keys(e,'earcon')


def test_prediction_setting_from_session_config_is_used():
    from dataclasses import replace
    moving=replace(h(),velocity_px_s=(100,0))
    results=[]
    for prediction in (0,.5):
        b=Brain({'brain': {'prediction': prediction}})
        step(b,0,[cmd('find','wallet')])
        results.append(step(b,.1,hand=moving,target=t())[0])
    assert results[0].dx == pytest.approx((150-40)/640)
    assert results[1].dx < results[0].dx


@pytest.mark.parametrize('kind',['yes','no'])
@pytest.mark.parametrize('active',[False,True])
def test_yes_no_acknowledges_without_changing_search_or_guidance(kind,active):
    b=guiding() if active else Brain()
    previous_state,previous_phrase=b.state,b.phrase
    d,e=step(b,.3,[cmd(kind)],hand=h(3) if active else None,target=t() if active else None)
    assert (d.state,b.phrase)==(previous_state,previous_phrase)
    assert keys(e,'speak')==[f'{kind}_ack']
    assert d.speech==f'I heard {kind}.'
    assert next(x.priority for x in e if x.kind=='speak')==0
    assert not any(x.kind in ('set_phrase','force_query','earcon') for x in e)
