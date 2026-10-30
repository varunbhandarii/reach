"""Summaries of measured loop metrics and explicitly reported trial outcomes."""

import json
from pathlib import Path

import numpy as np

from .run_metrics import summarize as summarize_loop


def summarize(run_dir):
    root = Path(run_dir)
    summary = summarize_loop(root)
    manifest = json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    saved = json.loads((root/"summary.json").read_text(encoding="utf-8")) if (root/"summary.json").exists() else {}
    events = [json.loads(line) for line in (root/"events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    trials = []
    active = None
    path = []
    for event in events:
        if event["type"] == "trial_start":
            active, path = event["data"], []
        elif event["type"] == "reach_sample" and active:
            path.append(event["data"]["relative_xy"])
        elif event["type"] == "trial_end":
            ratio = None
            if len(path) >= 2:
                initial = float(np.linalg.norm(path[0]))
                if initial > .001:
                    ratio = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()/initial)
            trials.append({**event["data"], "relative_path_length_ratio": ratio})
            active, path = None, []
    summary.update(mount=manifest["mount"], source=manifest["source"], objects=manifest["objects"],
        prediction=manifest["prediction"], trial_mode=manifest.get("trial_mode"), completed=saved.get("completed", False),
        duration_s=saved.get("duration_s"), audio_measured=saved.get("audio_measured", False),
        audio_underflows=saved.get("audio", {}).get("underflows"),
        audio_callback_errors=saved.get("audio", {}).get("callback_errors"),
        capture=saved.get("capture"), trials=trials,
        trial_successes=sum(t["success"] for t in trials),
        motion_to_sound_estimate_ms=saved.get("motion_to_sound_estimate_ms"),
        latency_note=manifest.get("latency_note"),
        path_note="Relative source-image path proxy includes camera/target estimation error; not physical hand distance.")
    def percentiles(values):
        values = [float(v) for v in values if isinstance(v, (int, float)) and np.isfinite(v)]
        return {'count': len(values), 'p50': float(np.median(values)) if values else None,
                'p95': float(np.percentile(values, 95)) if values else None}
    voice_turns = [r['data'] for r in events if r['type'] == 'voice_turn' and not r['data'].get('ignored')]
    speech = [r['data'] for r in events if r['type'] == 'speech']
    cached_ids = {r['id'] for r in speech if r.get('event') == 'ready' and r.get('cached')}
    state_times = {}
    states = [r for r in events if r['type'] == 'state']
    connected = {}
    connected_s = 0.0
    start_window = manifest.get('active_start_mono') or (events[0]['t_mono'] if events else 0)
    end_window = manifest.get('active_end_mono') or (events[-1]['t_mono'] if events else start_window)
    for event in events:
        data = event['data']
        if event['type'] != 'voice_session' or data.get('mode') != 'ENDPOINTING':
            continue
        session = data.get('session_id')
        if data.get('event') == 'ack':
            connected[session] = event['t_mono']
        elif data.get('event') == 'close' and session in connected:
            connected_s += max(0, min(end_window, event['t_mono'])-max(start_window, connected.pop(session)))
    connected_s += sum(max(0, end_window-max(start_window, stamp)) for stamp in connected.values())
    for i,event in enumerate(states):
        state = event['data'].get('to')
        end = states[i+1]['t_mono'] if i+1 < len(states) else end_window
        if state:
            state_times[state] = state_times.get(state, 0)+max(0, min(end, end_window)-max(event['t_mono'], start_window))
    summary.update(
        voice=saved.get('voice'), speech=saved.get('speech'),
        voice_latency_ms=percentiles([r.get('latency_ms') for r in voice_turns if r.get('mode') == 'ENDPOINTING']),
        ptt_latency_ms=percentiles([r.get('latency_ms') for r in voice_turns if r.get('mode') == 'PUSH_TO_TALK']),
        speech_first_audio_ms=percentiles([r.get('first_audio_ms') for r in speech if r.get('event') == 'start']),
        cached_speech_first_audio_ms=percentiles([r.get('first_audio_ms') for r in speech if r.get('event') == 'start' and r.get('id') in cached_ids]),
        voice_acknowledgements=sum(r['type'] == 'voice_session' and r['data'].get('event') == 'ack' for r in events),
        voice_reconnects=sum(r['type'] == 'voice_session' and r['data'].get('reason') == 'reconnecting' for r in events),
        endpointing_connected_s=connected_s,
        commands=[r['data'] for r in events if r['type'] == 'command'],
        time_in_state_s=state_times,
        contact_firings=[r['data'] for r in events if r['type'] == 'contact' and r['data'].get('event') == 'fire'],
        contacts_with_reported_success=sum(t.get('success', False) and t.get('contact_fired_s') is not None for t in trials))
    return summary
