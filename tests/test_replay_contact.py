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
