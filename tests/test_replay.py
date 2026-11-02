from dataclasses import asdict
import json
from pathlib import Path

import pytest

from reach.brain import Brain
from reach.replay import replay_session, ReplayError
from reach.session_codec import plain, replay_config
from reach.types import Command, HandObs, Instance, ReadBack, TargetObs


def trace(root):
    root.mkdir()
    cfg={'objects':['wallet'],'voice':{'input_device':'private microphone'},
         'audio':{'output_device':'private headphones','gain':.2},'brain':{'device_name':'not used'}}
    manifest={'config':cfg,'objects':['wallet'],'input_device':'private microphone'}
    (root/'manifest.json').write_text(json.dumps(manifest))
    brain=Brain(cfg,['wallet'])
    target=TargetObs(1,'wallet',(Instance('a',(100,100,200,200),.9,None),),0,'sam',0)
    rows=[]
    def step(now,**kwargs):
        decision,effects=brain.step(now,**kwargs)
        rows.extend(dict(type=e.key,data=e.payload,t_mono=now) for e in effects if e.kind=='log')
        rows.append(dict(type='decision',data=plain(decision),t_mono=now))
        return effects
    step(0,commands=[Command('find','wallet','private spoken sentence',1,0,'voice')])
    for i in range(1,15):
        hand=HandObs(i,True,(150,150),None,(),(0,0),1)
        effects=step(i*.05,hand=hand,target=target)
        request=next((e.payload for e in effects if e.kind=='readback'),None)
        if request:
            step((i+1)*.05,results=[ReadBack(request['request_id'],request['frame_id'],'wallet',True,'wallet','',.9,50)])
            break
    path=root/'events.jsonl'
    path.write_text('\n'.join(json.dumps(plain(dict(seq=i,t_wall=0,trial=None,**row)),allow_nan=False) for i,row in enumerate(rows))+'\n')
    return root


def test_synthetic_voice_contact_readback_replays_without_devices(tmp_path):
    result=replay_session(trace(tmp_path/'run'))
    assert result['status']=='ok' and result['steps']>=8
    assert result['state_changes']>=4 and result['readbacks']==1


def test_old_optional_wire_fields_restore_without_rewriting_evidence(tmp_path):
    root=trace(tmp_path/'run')
    manifest=json.loads((root/'manifest.json').read_text());manifest['contract_version']='4'
    (root/'manifest.json').write_text(json.dumps(manifest))
    rows=[json.loads(s) for s in (root/'events.jsonl').read_text().splitlines()]
    for row in rows:
        if row['type']=='brain_in':
            for command in row['data']['commands']:command.pop('language',None)
            for result in row['data']['results']:
                result.pop('same_object',None);result.pop('mode',None)
    (root/'events.jsonl').write_text('\n'.join(json.dumps(row) for row in rows)+'\n')
    before=(root/'events.jsonl').read_bytes()
    assert replay_session(root)['status']=='ok'
    assert (root/'events.jsonl').read_bytes()==before


@pytest.mark.parametrize('kind',['decision','brain_out'])
def test_first_divergence_identifies_edited_field(tmp_path,kind):
    root=trace(tmp_path/'run')
    rows=[json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
    row=next(row for row in rows if row['type']==kind)
    if kind=='decision': row['data']['state']='edited'
    else: row['data']['decision']['state']='edited'
    (root/'events.jsonl').write_text('\n'.join(json.dumps(row) for row in rows)+'\n')
    result=replay_session(root)
    assert result['status']=='diverged' and result['step']==0 and result['field'].endswith('.state')


def test_strip_removes_text_devices_and_replays_without_mutating_source(tmp_path):
    root=trace(tmp_path/'run')
    before={p.name:p.read_bytes() for p in root.iterdir()}
    result=replay_session(root,strip=tmp_path/'public')
    assert result['status']=='ok' and replay_session(tmp_path/'public')['status']=='ok'
    contents=''.join(path.read_text() for path in (tmp_path/'public').iterdir())
    assert 'private spoken sentence' not in contents
    assert 'private microphone' not in contents and 'private headphones' not in contents
    assert 'device_name' not in contents
    assert before=={p.name:p.read_bytes() for p in root.iterdir()}
    assert {p.name for p in (tmp_path/'public').iterdir()}=={'manifest.json','events.jsonl'}
    with pytest.raises(ReplayError,match='new strip directory'):
        replay_session(root,strip=tmp_path/'public')


def test_legacy_logs_have_no_replay_inputs_and_do_not_construct_brain(tmp_path,monkeypatch):
    (tmp_path/'events.jsonl').write_text(json.dumps({'type':'decision','data':{'seq':0}})+'\n')
    monkeypatch.setattr('reach.replay.Brain',lambda *a:pytest.fail('legacy log cannot replay'))
    assert replay_session(tmp_path)=={'status':'no_inputs','message':'no replay inputs','steps':0}


def test_incomplete_trace_is_not_a_pass(tmp_path):
    (tmp_path/'events.jsonl').write_text(json.dumps({'type':'brain_in','data':{'now':0}})+'\n')
    with pytest.raises(ReplayError,match='without output'): replay_session(tmp_path)


def test_cli_reports_success_and_no_inputs(tmp_path,capsys):
    from reach.cli import main
    assert main(['replay',str(trace(tmp_path/'run'))])==0
    assert 'replay OK' in capsys.readouterr().out
    legacy=tmp_path/'legacy';legacy.mkdir()
    (legacy/'events.jsonl').write_text('')
    assert main(['replay',str(legacy)])==0
    assert capsys.readouterr().out.strip()=='no replay inputs'



def test_effect_edit_is_detected_even_when_decision_is_unchanged(tmp_path):
    root=trace(tmp_path/'run')
    rows=[json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
    output=next(row['data'] for row in rows if row['type']=='brain_out')
    next(effect for effect in output['effects'] if effect['kind']=='set_phrase')['text']='edited'
    (root/'events.jsonl').write_text('\n'.join(json.dumps(row) for row in rows)+'\n')
    result=replay_session(root)
    assert result['status']=='diverged' and result['step']==0
    assert result['field'].startswith('effects') and result['field'].endswith('.text')
