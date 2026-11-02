import json
from pathlib import Path
from reach.replay_contact import replay_contact


def test_legacy_contact_replay_pairs_manual_touch_and_preserves_logs(tmp_path):
    rows=[]
    def event(kind,t,data): rows.append({'seq':len(rows),'t_mono':t,'trial':'t1','type':kind,'data':data})
    event('trial_start',0,{})
    event('target',0,{'frame_id':1,'box_xyxy':[100,100,200,200],'source':'tracker','age_ms':0})
    for i in range(8): event('hand',i*.05,{'frame_id':i,'present':True,'fingertip_px':[150,150],'score':1})
    event('trial_end',1,{'success':True})
    path=tmp_path/'events.jsonl';path.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    original=path.read_bytes()
    report=replay_contact(tmp_path)
    assert len(report['firings'])==1
    row=report['firings'][0]
    assert row['overlap'] is None and row['trial_elapsed_s']<1 and row['reported_touch_after_s']>0
    assert path.read_bytes()==original


def test_snapshot_contact_replay_keeps_masks_and_trial_arming(tmp_path):
    from dataclasses import asdict
    import numpy as np
    from reach.brain import Brain
    from reach.masks import encode_rle
    from reach.types import Command, HandObs, TargetObs, Instance
    cfg = {'spark': {'enabled': False}}
    (tmp_path/'manifest.json').write_text(json.dumps({'config': cfg, 'objects': ['ball'], 'contract_version': '5'}))
    b = Brain(cfg, ['ball']); rows = []
    for i in range(10):
        hand = HandObs(i, True, (150,150), None, ((100,100),(200,100),(200,200),(100,200)), (0,0), 1)
        target = TargetObs(i, 'ball', (Instance('ball', (100,100,200,200), 1, encode_rle(np.ones((100,100), np.uint8))),), 0, 'sam', 0)
        d, effects = b.step(i*.05, hand=hand, target=target,
                            commands=[Command('find', 'ball', '', 1, 0)] if i == 0 else ())
        rows += [{'type': kind, 'data': data, 'trial': 't1', 't_mono': i*.05} for kind, data in (
            ('brain_in', effects[0].payload), ('brain_out', effects[-1].payload), ('decision', asdict(d)))]
    (tmp_path/'events.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    report = replay_contact(tmp_path)
    assert report['mode'] == 'saved_reducer_inputs'
    assert len(report['firings']) == 1 and report['firings'][0]['overlap'] == 1
