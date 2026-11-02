import json

import pytest

from reach.session_report import summarize


def test_report_separates_latency_origins_contacts_and_active_window(tmp_path):
    (tmp_path/'manifest.json').write_text(json.dumps(dict(
        mount='handheld', source='android_mjpeg', objects=['wallet', 'pen'], prediction=.5,
        trial_mode='voice', active_start_mono=10, active_end_mono=20)))
    (tmp_path/'summary.json').write_text(json.dumps(dict(
        completed=True, duration_s=10, audio_measured=True,
        audio=dict(underflows=0, callback_errors=0))))
    rows = [
        (8, 'voice_session', dict(event='ack', mode='ENDPOINTING', session_id='one')),
        (9, 'state', {'to': 'IDLE'}),
        (11, 'state', {'to': 'GUIDING'}),
        (12, 'voice_turn', dict(mode='ENDPOINTING', latency_ms=500,latency_local_ms=700,local_latency_valid=True)),
        (13, 'voice_turn', dict(mode='ENDPOINTING', latency_ms=900, ignored=True)),
        (14, 'voice_turn', dict(mode='PUSH_TO_TALK', latency_ms=80)),
        (14, 'speech', dict(id=1, event='ready', cached=True)),
        (14.1, 'speech', dict(id=1, event='start', first_audio_ms=15)),
        (15, 'trial_start', dict(index=1)),
        (16, 'contact', dict(event='fire', inside=True)),
        (17, 'trial_end', dict(index=1, success=True, contact_fired_s=1)),
        (18, 'readback_resp', dict(status='ok',matches=True,latency_ms=900,cached=False)),
        (19, 'readback_resp', dict(status='ok',matches=False,latency_ms=1,cached=True)),
        (21, 'voice_session', dict(event='close', mode='ENDPOINTING', session_id='one')),
        (23, 'session_end', {}),
    ]
    (tmp_path/'events.jsonl').write_text('\n'.join(json.dumps(dict(t_mono=t, type=k, data=d)) for t,k,d in rows))
    report = summarize(tmp_path)
    assert report['voice_latency_ms'] == dict(count=1, p50=500, p95=500)
    assert report['ptt_latency_ms']['p50'] == 80
    assert report['voice_local_latency_ms']['p50']==700
    assert report['readback_latency_ms']['p50']==900
    assert report['readback_matches']==report['readback_mismatches']==1
    assert report['cached_speech_first_audio_ms']['p50'] == 15
    assert report['endpointing_connected_s'] == pytest.approx(10)
    assert report['time_in_state_s'] == {'IDLE': 1, 'GUIDING': 9}
    assert report['contacts_with_reported_success'] == 1
    assert len(report['contact_firings']) == 1
