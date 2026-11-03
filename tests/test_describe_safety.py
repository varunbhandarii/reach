from dataclasses import asdict
import json

import pytest

from reach.brain import Brain
from reach.grammar import parse
from reach.replay import replay_session
from reach.safety import BLOCKED_TERMS, blocked_term
from reach.session_codec import plain, replay_config
from reach.types import Command, SceneList, CONTRACT_VERSION


@pytest.mark.parametrize('text', ["what's on the table", 'what do you see', "what's in front of me",
                                "what's here", 'describe', 'look around'])
def test_scene_grammar(text):
    assert parse(text).kind == 'describe'


@pytest.mark.parametrize('source', ['voice', 'spark_parse', 'key', 'trial'])
@pytest.mark.parametrize('term', BLOCKED_TERMS)
def test_unsafe_find_clears_existing_guidance_and_never_sets_a_search(term, source):
    brain = Brain()
    brain.state, brain.phrase = 'GUIDING', 'wallet'
    decision, effects = brain.step(0, commands=[Command('find', 'blue '+term, '', 1, 0, source)])
    assert decision.state == 'IDLE' and decision.cue is None
    assert any(e.key == 'safety_block' for e in effects)
    assert any(e.key == 'safety_refusal' and e.priority == 2 for e in effects)
    assert not any(e.kind == 'set_phrase' and e.text for e in effects)


def test_whole_word_matching_and_attribute_preservation():
    for phrase in ('matchbox toy', 'panther figurine', 'potato', 'ironman toy'):
        assert blocked_term(phrase) is None
    assert blocked_term('BROKEN-glass') == 'broken glass'
    for phrase in ('blue pen', 'black pen', 'red bottle'):
        assert parse('find the '+phrase, vocabulary=['pen', 'bottle']).phrase == phrase


def test_describe_supersedes_guidance_correlates_results_and_flips_positions():
    brain = Brain()
    brain.state, brain.phrase = 'GUIDING', 'wallet'
    d, effects = brain.step(0, commands=[parse('describe')], frame_id=10)
    request = next(e.payload for e in effects if e.kind == 'describe')
    assert d.state == 'DESCRIBING' and d.cue is None and brain.phrase is None
    assert request == {'request_id': 1, 'frame_id': 10}
    assert brain.step(1, results=[SceneList(99, 10, (('pen', 'left'),), 1)])[0].state == 'DESCRIBING'
    assert brain.step(2, results=[SceneList(1, 11, (('pen', 'left'),), 1)])[0].state == 'DESCRIBING'
    d, effects = brain.step(3, flipped=True, results=[SceneList(1, 10, (('pen', 'left'), ('knife', 'right')), 1)])
    assert d.state == 'IDLE' and brain.last_scene == (('pen', 'right'), ('knife', 'left'))
    assert 'pen on your right' in d.speech
    assert 'Careful, there is a knife.' in d.speech
    assert not any(e.kind == 'set_phrase' and e.text for e in effects)


@pytest.mark.parametrize('error', [None, 'timeout'])
def test_empty_and_error_scene_return_idle(error):
    b = Brain(); b.step(0, commands=[parse('describe')], frame_id=3)
    d, _ = b.step(1, results=[SceneList(1, 3, (), 1, error)])
    assert d.state == 'IDLE'
    assert ('anything I can name' if error is None else "couldn't look around") in d.speech


def test_accepted_scene_cannot_reactivate_an_old_click_target():
    from reach.types import TargetObs, Instance
    b = Brain(); b.step(0, commands=[parse('describe')], frame_id=3)
    old = TargetObs(3, '', (Instance('click', (1,1,5,5), 1, None),), 0, 'tracker', 0)
    d, _ = b.step(1, target=old, results=[SceneList(1, 3, (), 1)])
    assert d.state == 'IDLE' and d.cue is None


def test_describe_timeout_stop_new_find_and_confirming_guards():
    b = Brain(); b.step(0, commands=[parse('describe')])
    assert b.step(6)[0].state == 'IDLE'
    b.step(7, commands=[parse('describe')])
    assert b.step(7.1, commands=[parse('stop')], results=[SceneList(2, 3, (('pen', 'left'),), 1)])[0].state == 'IDLE'
    b.step(8, commands=[parse('describe')])
    assert b.step(8.1, commands=[parse('find wallet')], results=[SceneList(3, 3, (('pen', 'left'),), 1)])[0].state == 'SEARCHING'
    b.state = 'CONFIRMING'
    _, effects = b.step(8.2, commands=[parse('describe')])
    assert not any(e.kind == 'describe' for e in effects)


def test_describe_exact_replay_and_events_only_copy(tmp_path):
    b = Brain(); rows = []
    for now, kwargs in [(0, {'commands': [parse('describe')], 'frame_id': 10}),
                        (1, {'results': [SceneList(1, 10, (('pen', 'left'),), 10)]})]:
        d, effects = b.step(now, **kwargs)
        for kind, data in [('brain_in', effects[0].payload), ('brain_out', effects[-1].payload), ('decision', asdict(d))]:
            rows.append({'type': kind, 'data': data, 't_mono': now})
    (tmp_path/'events.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    (tmp_path/'manifest.json').write_text(json.dumps({'config': replay_config({}), 'contract_version': CONTRACT_VERSION}))
    assert replay_session(tmp_path, strip=tmp_path/'stripped')['status'] == 'ok'
    assert replay_session(tmp_path/'stripped')['status'] == 'ok'


def test_scene_response_between_default_and_configured_deadline():
    result = SceneList(1, 10, (('mug', 'middle'),), 6890)
    b = Brain({'scene': {'deadline_s': 10}})
    b.step(0, commands=[parse('describe')], frame_id=10)
    assert b.step(6.1)[0].state == 'DESCRIBING'
    d, _ = b.step(6.89, results=[result])
    assert d.state == 'IDLE' and 'mug in the middle' in d.speech
    assert b.last_scene == result.objects

    default = Brain()
    default.step(0, commands=[parse('describe')], frame_id=10)
    assert default.step(6)[0].state == 'IDLE'
    default.step(6.89, results=[result])
    assert default.last_scene == ()


def test_configured_scene_deadline_is_still_bounded():
    b = Brain({'scene': {'deadline_s': 10}})
    b.step(0, commands=[parse('describe')], frame_id=10)
    assert b.step(9.999)[0].state == 'DESCRIBING'
    assert b.step(10)[0].state == 'IDLE'
    b.step(10.1, results=[SceneList(1, 10, (('pen', 'left'),), 10100)])
    assert b.last_scene == ()


@pytest.mark.parametrize('command,state', [('stop', 'IDLE'), ('cancel', 'IDLE'), ('find wallet', 'SEARCHING')])
def test_slow_scene_cannot_undo_a_new_command(command, state):
    b = Brain({'scene': {'deadline_s': 10}})
    b.step(0, commands=[parse('describe')], frame_id=10)
    d, _ = b.step(6.89, commands=[parse(command)], results=[SceneList(1, 10, (('pen', 'left'),), 6890)])
    assert d.state == state and b.last_scene == ()


def test_long_scene_deadline_survives_events_only_replay(tmp_path):
    cfg = {'scene': {'deadline_s': 10, 'cache_dir': 'private-local-path'}}
    b = Brain(cfg); rows = []
    for now, kwargs in [(0, {'commands': [parse('describe')], 'frame_id': 10}),
                        (6.89, {'results': [SceneList(1, 10, (('pen', 'left'),), 6890)]})]:
        d, effects = b.step(now, **kwargs)
        for kind, data in [('brain_in', effects[0].payload), ('brain_out', effects[-1].payload), ('decision', asdict(d))]:
            rows.append({'type': kind, 'data': data, 't_mono': now})
    clean = replay_config(cfg)
    assert clean['scene'] == {'deadline_s': 10}
    (tmp_path/'events.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    (tmp_path/'manifest.json').write_text(json.dumps({'config': clean, 'contract_version': CONTRACT_VERSION}))
    assert replay_session(tmp_path, strip=tmp_path/'stripped')['status'] == 'ok'
    assert replay_session(tmp_path/'stripped')['status'] == 'ok'
