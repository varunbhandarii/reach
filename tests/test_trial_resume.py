import json
import random
from types import SimpleNamespace as NS

import pytest

from reach.session_report import summarize
from reach.trial_resume import load_trial_resume, saved_trial_prefix, TrialResumeError
from reach.trials import Trials


def source_run(root, earlier_failure=False):
    root.mkdir()
    manifest = dict(mount='handheld', source='android_mjpeg', objects=['wallet','pen'],
        prediction=.5, trial_mode='voice', trial_seed=1, contact={'enabled':True})
    (root/'manifest.json').write_text(json.dumps(manifest))
    rng = random.Random(1)
    rows, outcomes = [], []
    for index in range(1, 6):
        phrase = rng.choice(manifest['objects'])
        result = dict(index=index, phrase=phrase, success=index<5, elapsed_s=10 if index<5 else 2,
            reason='reported_touch' if index<5 else 'interrupted',
            outcome_source='manual_touch_key' if index<5 else 'interrupted',
            contact_fired_s=8 if index<5 else None)
        if index == 2 and earlier_failure:
            result.update(success=False, reason='reported_failure', outcome_source='reported_failure',
                          contact_fired_s=None)
        outcomes.append(result)
        rows.extend([dict(type='trial_start',t_mono=index*20,data=dict(index=index,phrase=phrase,limit_s=45)),
                     dict(type='trial_end',t_mono=index*20+result['elapsed_s'],data=result)])
    saved = dict(completed=False, error='hand/controller stopped (TypeError)', audio_measured=True,
                 audio=dict(underflows=0,callback_errors=0),trials=outcomes)
    (root/'summary.json').write_text(json.dumps(saved))
    (root/'events.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))
    return root


def descriptor(root, **changes):
    kwargs = dict(count=5,mode='voice',objects=['wallet','pen'],seed=1,limit=45,
                  mount='handheld',prediction=.5,source='android_mjpeg')
    return load_trial_resume(root, **(kwargs|changes))


def test_continuation_restarts_only_interrupted_target_and_keeps_source(tmp_path):
    root = source_run(tmp_path/'old')
    before = {path.name:path.read_bytes() for path in root.iterdir()}
    linked = descriptor(root)
    prefix, recovery = saved_trial_prefix(linked)
    events, announcements = [], []
    now = [200.0]
    trial = Trials(5,'voice',['wallet','pen'],NS(emit=lambda k,d:events.append((k,d))),
                   retained=prefix,clock=lambda:now[0])
    actions = NS(announce_trial=lambda n,p:announcements.append((n,p)),set_phrase=lambda p:None)
    trial.prepare(actions,None)
    assert announcements == [(5,'pen')]
    assert trial.index == 4 and len(trial.results) == 4
    assert trial.start(True)
    now[0] += 1
    trial.note_contact(now[0])
    now[0] += 2
    trial.finish(True,'reported_touch',actions)
    assert trial.finished and len(trial.results) == 5
    assert [data['index'] for kind,data in events if kind == 'trial_end'] == [5]
    assert recovery['source_completed'] is False
    assert recovery['interrupted_attempt']['reason'] == 'interrupted'
    assert before == {path.name:path.read_bytes() for path in root.iterdir()}


@pytest.mark.parametrize('changes',[
    {'objects':['pen','wallet']}, {'mode':'phrase'}, {'seed':2},
    {'limit':60}, {'count':4}, {'prediction':0}, {'source':'file'}])
def test_resume_rejects_changed_protocol(tmp_path,changes):
    with pytest.raises(TrialResumeError):
        descriptor(source_run(tmp_path/'old'),**changes)


def test_retained_prefix_includes_previous_failed_trials(tmp_path):
    prefix,_ = saved_trial_prefix(descriptor(source_run(tmp_path/'old',earlier_failure=True)))
    assert len(prefix) == 4 and prefix[1]['success'] is False
    assert [row['index'] for row in prefix] == [1,2,3,4]


def test_report_detects_modified_source_evidence(tmp_path):
    root = source_run(tmp_path/'old')
    linked = descriptor(root)
    with (root/'events.jsonl').open('a') as stream:
        stream.write('\n'+json.dumps(dict(type='ui',t_mono=120,data={})))
    with pytest.raises(TrialResumeError,match='changed'):
        saved_trial_prefix(linked)


def test_recovered_report_separates_original_crash_and_current_completion(tmp_path):
    root = source_run(tmp_path/'old')
    linked = descriptor(root)
    current = tmp_path/'new'
    current.mkdir()
    manifest = json.loads((root/'manifest.json').read_text())
    manifest.update(trial_resume=linked,trial_count=5)
    (current/'manifest.json').write_text(json.dumps(manifest))
    (current/'summary.json').write_text(json.dumps(dict(
        completed=True,audio_measured=True,audio=dict(underflows=0,callback_errors=0))))
    result = dict(index=5,phrase='pen',success=True,elapsed_s=3,contact_fired_s=1)
    (current/'events.jsonl').write_text('\n'.join(json.dumps(dict(type=k,t_mono=t,data=d))
        for k,t,d in [('trial_start',200,dict(index=5)),('trial_end',203,result)]))
    report = summarize(current)
    assert report['completed'] and report['trial_successes'] == 5
    assert report['contacts_with_reported_success'] == 5
    assert len(report['current_session_trials']) == 1
    assert report['trial_recovery']['source_completed'] is False
    assert 'TypeError' in report['trial_recovery']['source_error']
    assert report['trial_recovery']['attempt_count'] == 6
    assert report['trial_recovery']['interrupted_attempt']['success'] is False
    assert all(row['retained_from_source'] for row in report['trials'][:4])


def test_successful_source_cannot_be_replayed_as_an_interruption(tmp_path):
    root = source_run(tmp_path/'old')
    saved = json.loads((root/'summary.json').read_text())
    saved['completed'] = True
    (root/'summary.json').write_text(json.dumps(saved))
    with pytest.raises(TrialResumeError,match='incomplete'):
        descriptor(root)


def test_cli_resume_needs_original_total_before_opening_devices(tmp_path,capsys):
    from reach.cli import main
    assert main(['run','--mount','handheld','--resume-trials',str(tmp_path)]) == 2
    assert 'original total' in capsys.readouterr().err


def test_runner_initializes_only_remaining_trial_and_links_manifest(tmp_path):
    import time
    from reach.cli import parser, configure
    from reach.runner import Runner
    old = source_run(tmp_path/'old')
    before = (old/'events.jsonl').read_bytes()
    args = parser().parse_args(['run','--mount','handheld','--source','http://synthetic/video',
        '--objects','wallet,pen','--trials','5','--trial-mode','voice','--speech','off',
        '--resume-trials',str(old),'--log-dir',str(tmp_path/'new')])
    runner = Runner(configure(args),args,
        camera_factory=lambda cfg,log:NS(source='android_mjpeg',log=log),
        hand_factory=lambda cfg:NS(),audio_factory=lambda cfg:NS(),
        perception_factory=lambda *a:NS(),
        voice_factory=lambda *a,**kw:NS())
    try:
        manifest = runner._manifest()
        runner.log.write_manifest(manifest)
        runner._trial.prepare(runner._trial_actions,None)
        assert runner._trial.index == 4 and runner._trial.phrase == 'pen'
        assert runner._trial.start(True)
        runner._trial.note_contact(time.perf_counter())
        runner._trial.finish(True,'reported_touch',runner._trial_actions)
        assert runner._trial.finished and len(runner._trial.results) == 5
    finally:
        runner.log.close()
    assert manifest['trial_count'] == 5
    assert manifest['trial_resume']['source_run'] == str(old.resolve())
    rows = [json.loads(line) for line in runner.log.path.read_text().splitlines()]
    assert [r['data']['index'] for r in rows if r['type']=='trial_end'] == [5]
    assert (old/'events.jsonl').read_bytes() == before



def test_resume_accepts_current_contract_without_rewriting_source(tmp_path):
    root=source_run(tmp_path/'old')
    p=root/'manifest.json';metadata=json.loads(p.read_text());metadata['contract_version']='4'
    p.write_text(json.dumps(metadata))
    before=p.read_bytes()
    assert descriptor(root)['completed_prefix']==4
    assert p.read_bytes()==before
