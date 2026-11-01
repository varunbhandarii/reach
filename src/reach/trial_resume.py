"""Read an interrupted trial sequence without changing its source evidence."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import random


class TrialResumeError(ValueError):
    pass


def _source(root):
    root = Path(root).expanduser().resolve()
    try:
        contents = {name:(root/name).read_bytes() for name in
                    ('manifest.json', 'summary.json', 'events.jsonl')}
        manifest = json.loads(contents['manifest.json'])
        saved = json.loads(contents['summary.json'])
        events = [json.loads(line) for line in contents['events.jsonl'].splitlines() if line.strip()]
    except (OSError, ValueError, TypeError):
        raise TrialResumeError('Resume source must contain a saved manifest, summary and valid events.') from None
    if (not isinstance(manifest, dict) or not isinstance(saved, dict) or
            any(not isinstance(row, dict) or not isinstance(row.get('data'), dict) or
                not isinstance(row.get('type'), str) for row in events)):
        raise TrialResumeError('Resume source has invalid event or metadata records.')
    if manifest.get('trial_resume'):
        raise TrialResumeError('Choose the original interrupted run, not an earlier continuation.')
    ends = [row['data'] for row in events if row['type'] == 'trial_end']
    if (saved.get('completed') is not False or not ends or
            ends[-1].get('reason') != 'interrupted' or ends[-1].get('success') is not False):
        raise TrialResumeError('Resume requires an incomplete run ending in an interrupted trial.')
    if ([row.get('index') for row in ends] != list(range(1, len(ends)+1)) or
            any(row.get('reason') == 'interrupted' for row in ends[:-1]) or
            saved.get('trials') != ends):
        raise TrialResumeError('Resume source has inconsistent or nonconsecutive trial outcomes.')
    starts = [row['data'] for row in events if row['type'] == 'trial_start']
    if [row.get('index') for row in starts] != list(range(1, len(ends)+1)):
        raise TrialResumeError('Each saved outcome needs its original timed trial start.')
    hashes = {name:sha256(value).hexdigest() for name,value in contents.items()}
    return root, manifest, saved, events, ends, starts, hashes


def load_trial_resume(root, *, count, mode, objects, seed, limit, mount, prediction, source):
    root, manifest, saved, events, ends, starts, hashes = _source(root)
    expected = dict(trial_mode=mode, objects=list(objects), trial_seed=seed,
                    mount=mount, prediction=prediction, source=source)
    if any(manifest.get(key) != value for key,value in expected.items()):
        raise TrialResumeError('Resume must keep the original mode, objects, seed, mount, prediction and camera source.')
    if count < len(ends) or (manifest.get('trial_count') is not None and manifest['trial_count'] != count):
        raise TrialResumeError('Use the original total trial count when resuming.')
    if any(row.get('limit_s') != limit for row in starts):
        raise TrialResumeError('Resume must keep the original per-trial time limit.')
    if manifest.get('contract_version', '3') != '3':
        raise TrialResumeError('Resume requires the same shared session contract version.')
    if mode not in ('phrase', 'voice'):
        raise TrialResumeError('Resume currently supports scripted phrase or voice trials.')
    rng = random.Random(seed)
    if any(row.get('phrase') != rng.choice(objects) for row in ends):
        raise TrialResumeError('Saved trial targets do not match the original scripted sequence.')
    return dict(source_run=str(root), files_sha256=hashes, completed_prefix=len(ends)-1,
                interrupted_trial=len(ends), trial_count=count)


def saved_trial_prefix(descriptor):
    root, manifest, saved, events, ends, starts, hashes = _source(descriptor['source_run'])
    if (hashes != descriptor.get('files_sha256') or
            len(ends)-1 != descriptor.get('completed_prefix') or
            len(ends) != descriptor.get('interrupted_trial')):
        raise TrialResumeError('Resume source changed after it was linked; original evidence must be preserved.')
    prefix = [{**deepcopy(row), 'source_run':str(root), 'retained_from_source':True}
              for row in ends[:-1]]
    recovery = dict(source_run=str(root), source_completed=saved['completed'],
                    source_error=saved.get('error'), retained_trials=len(prefix),
                    interrupted_attempt=deepcopy(ends[-1]),
                    source_audio_underflows=saved.get('audio', {}).get('underflows'),
                    source_audio_callback_errors=saved.get('audio', {}).get('callback_errors'),
                    source_audio_measured=saved.get('audio_measured', False),
                    note='Completed trial outcomes retained; interrupted attempt preserved separately. This is a recovered sequence across processes.')
    return prefix, recovery
