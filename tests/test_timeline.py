import json

from reach.timeline import write_html


def test_timeline_escapes_content_and_omits_transcripts_and_external_resources(tmp_path):
    root=tmp_path/'run'
    root.mkdir()
    (root/'manifest.json').write_text('{}')
    events=[{'type':'state','t_mono':1,'data':{'to':'GUIDING'}},
        {'type':'decision','t_mono':2,'data':{'d':.1,'cue':{'pan':.2,'freq_hz':440}}},
        {'type':'voice_turn','t_mono':2,'data':{'transcript':'PRIVATE RAW SENTENCE','t_complete':2,'t_local_speech_end':1.5,'local_latency_valid':True,'latency_local_ms':500}},
        {'type':'speech','t_mono':2,'data':{'id':1,'event':'start','text':'<script>alert(1)</script>'}},
        {'type':'speech','t_mono':3,'data':{'id':1,'event':'end'}}]
    (root/'events.jsonl').write_text('\n'.join(json.dumps(r) for r in events))
    path=tmp_path/'timeline.html'
    write_html(root,path)
    text=path.read_text()
    assert '<svg' in text and 'local energy EOS to complete: 500.0 ms' in text
    assert 'PRIVATE RAW SENTENCE' not in text and '<script>' not in text
    assert '&lt;script&gt;' in text and 'http://' not in text and 'https://' not in text


def test_events_only_timeline_needs_no_live_summary_or_devices(tmp_path):
    (tmp_path/'manifest.json').write_text(json.dumps({'source':'events_only','replay_only':True}))
    row={'type':'brain_out','t_mono':1,'data':{'effects':[
        {'kind':'log','key':'state','payload':{'to':'DONE'}},
        {'kind':'speak','text':'You have the wallet.'}]}}
    (tmp_path/'events.jsonl').write_text(json.dumps(row))
    out=tmp_path/'out.html'
    assert write_html(tmp_path,out)['stripped']
    assert 'You have the wallet.' in out.read_text()


def test_stripped_snapshot_restores_measurement_lanes_without_raw_text(tmp_path):
    (tmp_path/'manifest.json').write_text(json.dumps({'source':'events_only','replay_only':True}))
    rows=[{'type':'brain_in','t_mono':1,'data':{'commands':[{'kind':'find','source':'voice','t_mono':1,
        'transcript':'PRIVATE RAW SENTENCE'}],'results':[]}},
        {'type':'brain_out','t_mono':1,'data':{'decision':{'d':.2,'cue':{'pan':.3,'freq_hz':440}},
            'effects':[{'kind':'readback','payload':{'request_id':1}}]}},
        {'type':'brain_in','t_mono':2,'data':{'results':[{'request_id':1,'latency_ms':1000,'error':None}]}}]
    (tmp_path/'events.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    out=tmp_path/'out.html'
    result=write_html(tmp_path,out);text=out.read_text()
    assert result['decisions']==1 and 'readback answer; client 1000.0 ms' in text
    assert 'voice command accepted: find' in text and 'PRIVATE RAW SENTENCE' not in text
