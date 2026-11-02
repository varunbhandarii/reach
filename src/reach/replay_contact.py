"""Replay the contact heuristic from local events only; never opens devices."""

import csv
import json
from pathlib import Path
import shutil
from .contact import ContactRule
from .types import HandObs, Instance, TargetObs


def _snapshot_firings(root, events, cfg):
    """Prefer complete logged reducer inputs, including the actual local masks."""
    from .brain import Brain
    from .session_codec import restore_inputs
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    brain = Brain(cfg if cfg is not None else manifest['config'], manifest.get('objects', ('wallet','pen','ball','apple')))
    firings, ends = [], {}
    started = None
    for event in events:
        kind, data = event['type'], event['data']
        if kind == 'trial_start':
            started = event['t_mono']
        elif kind == 'trial_end':
            ends[event['trial']] = {'time': event['t_mono'], 'success': data.get('success', False)}
            started = None
        elif kind == 'brain_in':
            decision, effects = brain.step(data['now'], **restore_inputs(data))
            if any(e.kind == 'earcon' and e.key == 'contact' for e in effects):
                state = brain.contact_state
                firings.append({'trial': event.get('trial'), 'phrase': brain.phrase or 'clicked point',
                    'frame_id': decision.hand_frame_id, 'target_frame_id': decision.target_frame_id,
                    't_mono': data['now'], 'trial_elapsed_s': data['now']-started if started is not None else None,
                    'reported_touch_after_s': None, 'inside': state.inside, 'still': state.still,
                    'overlap': state.overlap, 'rel_speed': state.rel_speed, 'review': ''})
    return firings, ends


def replay_contact(run_dir, *, frames=False, force=False, cfg=None):
    root=Path(run_dir)
    destination=root/'contact-replay.csv'
    if destination.exists() and not force:
        raise ValueError('contact-replay.csv exists; use --force or preserve the earlier result')
    events=[]
    with (root/'events.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip(): events.append(json.loads(line))
    rule=ContactRule(cfg)
    current=None;trial=None;started=None;phrase=None;firings=[];ends={}
    snapshots = any(event['type'] == 'brain_in' for event in events)
    for event in (() if snapshots else events):
        kind,data,now=event['type'],event['data'],event['t_mono']
        if kind=='trial_start':
            trial=event['trial'];started=now;rule.reset()
        elif kind=='trial_end':
            ends[event['trial']]={'time':now,'success':data.get('success',False)}
            trial=None;started=None;rule.reset()
        elif kind=='ui' and 'click' in data:
            phrase='';rule.reset()
        elif kind in ('command','ui') and ('phrase' in data):
            if data['phrase'] != phrase:
                phrase=data['phrase'];rule.reset()
        elif kind=='target':
            box=data.get('box_xyxy')
            if box is None or data.get('stale') or data.get('source')=='none':
                current=None
            else:
                current=TargetObs(data['frame_id'],phrase or '',(Instance('replay',tuple(box),None,None),),0,
                    data.get('source','tracker'),data.get('age_ms') or 0)
        elif kind=='hand':
            points=tuple(tuple(p) for p in data.get('landmarks_px',()))
            hull=()
            if points:
                from .hand import forearm_hull
                hull=forearm_hull(points)
            fingertip=data.get('fingertip_px')
            hand=HandObs(data['frame_id'],data['present'],tuple(fingertip) if fingertip else None,None,
                         hull,(0,0),data.get('score',0),points,data.get('handedness'))
            state=rule.update(now,hand,current)
            if state.fired:
                firings.append({'trial':trial or event['trial'],'phrase':phrase or 'clicked point','frame_id':hand.frame_id,
                    'target_frame_id':current.frame_id,'t_mono':now,
                    'trial_elapsed_s':now-started if started is not None else None,
                    'reported_touch_after_s':None,'inside':state.inside,'still':state.still,
                    'overlap':state.overlap,'rel_speed':state.rel_speed,'review':''})
    if snapshots:
        firings, ends = _snapshot_firings(root, events, cfg)
    for row in firings:
        end=ends.get(row['trial'])
        if end and end['success'] and end['time']>=row['t_mono']:
            row['reported_touch_after_s']=end['time']-row['t_mono']
    columns=['trial','phrase','frame_id','target_frame_id','t_mono','trial_elapsed_s','reported_touch_after_s',
             'inside','still','overlap','rel_speed','review']
    with destination.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(firings)
    if frames:
        folder=root/'contact-replay-frames';folder.mkdir(exist_ok=True)
        for i,row in enumerate(firings,1):
            source=root/'frames'/f"{int(row['frame_id'])}.jpg"
            if source.exists():shutil.copyfile(source,folder/f"{i:03d}-frame-{row['frame_id']}.jpg")
    return {'firings':firings,'csv':str(destination),
            'mode': 'saved_reducer_inputs' if snapshots else 'legacy_box_geometry',
            'limitation': ('Offline reducer replay with the logged masks and contact arming. Physical touch truth still requires frame/owner review.'
                          if snapshots else 'Offline box geometry, not physical contact truth. Target-box events do not reconstruct propagated masks; overlap is unavailable. Legacy logs also lack landmarks. Spot-check every saved firing frame.')}


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir',type=Path)
    parser.add_argument('--frames',action='store_true')
    parser.add_argument('--force',action='store_true')
    args=parser.parse_args(argv)
    report=replay_contact(args.run_dir,frames=args.frames,force=args.force)
    for row in report['firings']:
        print(f"{row['trial']} frame {row['frame_id']}: {row['trial_elapsed_s']} s after start; reported touch {row['reported_touch_after_s']} s later")
    print(json.dumps({'count':len(report['firings']),'csv':report['csv'],'limitation':report['limitation']},indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
