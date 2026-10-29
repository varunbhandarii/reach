"""Summarize saved loop measurements without inventing missing hardware results."""

import argparse
import json
from pathlib import Path
import numpy as np


def percentiles(values):
    return {"count": len(values), "p50": float(np.median(values)) if values else None,
            "p95": float(np.percentile(values, 95)) if values else None}


def summarize(run_dir):
    events = [json.loads(line) for line in (Path(run_dir)/"events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    groups = {}
    for event in events:
        groups.setdefault(event['type'], []).append(event)
    frames = groups.get('frame', [])
    decisions = groups.get('decision', [])
    rate = lambda rows: (len(rows)-1)/(rows[-1]['t_mono']-rows[0]['t_mono']) if len(rows)>1 and rows[-1]['t_mono']>rows[0]['t_mono'] else None
    values = lambda kind, key: [r['data'][key] for r in groups.get(kind, []) if isinstance(r['data'].get(key), (int, float))]
    duplicates = max(values('frame', 'duplicates_dropped'), default=0)
    return {'capture_fps': rate(frames), 'saved_frame_events': len(frames), 'duplicates_dropped': duplicates,
        'duplicate_fraction': duplicates/(duplicates+len(frames)) if frames else None,
        'hand_ms': percentiles(values('hand', 'detect_ms')), 'motion_ms': percentiles(values('motion', 'ms')),
        'sam_ms': percentiles([r['data']['latency_ms'] for r in groups.get('sam_resp', []) if r['data'].get('status') == 'ok']),
        'controller_hz': rate(decisions), 'audio_cpu_load': percentiles(values('audio', 'cpu_load')),
        'audio_underflows': max(values('audio', 'underflows'), default=None),
        'errors': [r['data'] for r in groups.get('error', [])],
        'motion_to_sound_ms': None, 'latency_note': 'Requires this device camera and audio output measurements; loop times alone are insufficient.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.run_dir), indent=2))


if __name__ == '__main__':
    main()
