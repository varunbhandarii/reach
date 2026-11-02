"""Replay recorded reducer inputs and outputs; no camera, model or network calls."""

from copy import deepcopy
import json
from pathlib import Path

from .brain import Brain
from .session_codec import plain, replay_config, restore_inputs
from .types import CONTRACT_VERSION, Command, Effect, ReadBack


class ReplayError(ValueError):
    pass


def first_difference(logged, replayed, path='$'):
    if type(logged) is not type(replayed):
        return dict(field=path, logged=logged, replayed=replayed)
    if isinstance(logged,dict):
        for key in sorted(set(logged)|set(replayed)):
            if key not in logged or key not in replayed:
                return dict(field=f'{path}.{key}', logged=logged.get(key), replayed=replayed.get(key))
            difference=first_difference(logged[key],replayed[key],f'{path}.{key}')
            if difference: return difference
    elif isinstance(logged,list):
        if len(logged)!=len(replayed):
            return dict(field=f'{path}.length',logged=len(logged),replayed=len(replayed))
        for index,(a,b) in enumerate(zip(logged,replayed)):
            difference=first_difference(a,b,f'{path}[{index}]')
            if difference: return difference
    elif logged!=replayed:
        return dict(field=path,logged=logged,replayed=replayed)
    return None


def _load(root):
    try:
        events=[json.loads(line) for line in (root/'events.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    except (OSError,ValueError):
        raise ReplayError('A saved events.jsonl with complete JSON records is required') from None
    pairs=[]
    pending=None
    decisions={}
    for event in events:
        if not isinstance(event,dict) or not isinstance(event.get('data'),dict):
            raise ReplayError('Invalid event envelope')
        kind,data=event.get('type'),event['data']
        if kind=='brain_in':
            if pending is not None: raise ReplayError('Incomplete replay trace: input without output')
            pending=data
        elif kind=='brain_out':
            if pending is None: raise ReplayError('Incomplete replay trace: output without input')
            pairs.append((pending,data));pending=None
        elif kind=='decision':
            seq=data.get('seq')
            if seq in decisions: raise ReplayError('Duplicate decision sequence')
            decisions[seq]=data
    if pending is not None: raise ReplayError('Incomplete replay trace: final input without output')
    if not pairs: return None,[],decisions
    try:
        manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    except (OSError,ValueError):
        raise ReplayError('Replay inputs need a saved manifest.json') from None
    if not isinstance(manifest,dict) or not isinstance(manifest.get('config'),dict):
        raise ReplayError('Replay inputs need manifest.config with the reducer settings')
    if manifest.get('contract_version') in ('3','4'):
        # Old snapshots omit newly optional message fields. Restore their defaults
        # in memory so wrapper-effect comparisons stay meaningful across versions.
        for incoming,_ in pairs:
            incoming['commands']=[plain(Command(**row)) for row in incoming.get('commands',())]
            incoming['results']=[plain(ReadBack(**row)) for row in incoming.get('results',())]
    return manifest,pairs,decisions


def _emit_rows(inputs,outputs,decision):
    now=inputs['now']
    return [dict(type='brain_in',data=inputs,t_mono=now),
            dict(type='brain_out',data=outputs,t_mono=now),
            dict(type='decision',data=decision,t_mono=now)]


def _strip(output,manifest,pairs):
    output=Path(output).expanduser().resolve()
    if output.exists(): raise ReplayError('Choose a new strip directory; existing evidence is preserved')
    cfg=replay_config(manifest['config'],manifest.get('objects'))
    objects=cfg['objects']
    brain=Brain(cfg,objects)
    rows=[]
    for inputs,expected in pairs:
        inputs=deepcopy(inputs)
        for command in inputs.get('commands',[]):
            command['transcript']=' '.join(str(value) for value in (command['kind'],command.get('phrase')) if value)
        decision,effects=brain.step(inputs['now'],**restore_inputs(inputs))
        actual=plain(decision)
        if first_difference(expected['decision'],actual):
            raise ReplayError('Transcript stripping unexpectedly changed a decision')
        incoming=effects[0].payload
        outgoing=effects[-1].payload
        rows.extend(_emit_rows(incoming,outgoing,actual))
    output.mkdir(parents=True)
    safe=dict(config=cfg,objects=objects,contract_version=CONTRACT_VERSION,source='events_only',replay_only=True)
    (output/'manifest.json').write_text(json.dumps(safe,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    with (output/'events.jsonl').open('w',encoding='utf-8') as stream:
        for seq,row in enumerate(rows):
            stream.write(json.dumps(dict(seq=seq,t_wall=0,trial=None,**row),allow_nan=False)+'\n')
    return str(output)


def replay_session(run_dir, *, strip=None):
    root=Path(run_dir).expanduser().resolve()
    manifest,pairs,decisions=_load(root)
    if not pairs:
        return dict(status='no_inputs',message='no replay inputs',steps=0)
    cfg=manifest['config']
    objects=manifest.get('objects',cfg.get('objects',['wallet','pen','ball','apple']))
    brain=Brain(cfg,objects)
    total_effects=state_changes=readbacks=0
    for index,(inputs,expected) in enumerate(pairs):
        try:
            decision,effects=brain.step(inputs['now'],**restore_inputs(inputs))
            actual=plain(decision)
            expected_effects=[plain(Effect('log',key='brain_in',payload=inputs)),
                              *expected['effects'],plain(Effect('log',key='brain_out',payload=expected))]
        except (ValueError,TypeError,KeyError,IndexError):
            raise ReplayError(f'Invalid replay record at step {index}') from None
        comparisons=[('decision',expected['decision'],actual),
                     ('logged_decision',decisions.get(expected.get('seq')),actual),
                     ('effects',expected_effects,plain(effects))]
        for field,logged,replayed in comparisons:
            difference=first_difference(logged,replayed,field)
            if difference:
                return dict(status='diverged',step=index,**difference)
        total_effects+=len(effects)
        state_changes+=sum(e.kind=='log' and e.key=='state' for e in effects)
        readbacks+=sum(e.kind=='readback' for e in effects)
    result=dict(status='ok',steps=len(pairs),effects=total_effects,state_changes=state_changes,readbacks=readbacks)
    if strip is not None:
        result['stripped']=_strip(strip,manifest,pairs)
    return result
