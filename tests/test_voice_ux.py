import numpy as np
import pytest

from reach.brain import Brain, GREETING, HELP
from reach.grammar import parse
from reach.types import Command, HandObs, Instance, TargetObs

ON = {'brain': {'direction_words': 'plain', 'cancel_stale_prompts': True, 'greeting': True}}


def command(kind, phrase=None):
    return Command(kind, phrase, kind, None, 0.0, 'key')


def spoken(effects):
    return [(e.key, e.text) for e in effects if e.kind == 'speak']


def target(phrase='wallet', frame=1):
    return TargetObs(frame, phrase, (Instance('a', (300, 200, 340, 260), .9, None),), 0, 'sam', 0)


@pytest.mark.parametrize('text,kind', [
    ('help', 'help'), ('What can I say?', 'help'), ('commands', 'help'),
    ('louder', 'louder'), ('volume up', 'louder'), ('quieter', 'quieter'), ('softer', 'quieter'),
    ('less talking', 'brief'), ('more talking', 'verbose')])
def test_grammar_maps_spoken_controls(text, kind):
    assert parse(text).kind == kind


def test_greeting_waits_for_speech_and_plays_once():
    brain = Brain(ON)
    _, effects = brain.step(0, perception_stats={'speech_ready': False})
    assert not spoken(effects)
    _, effects = brain.step(1, perception_stats={'speech_ready': True})
    assert spoken(effects) == [('greeting', GREETING)]
    _, effects = brain.step(2, perception_stats={'speech_ready': True})
    assert not spoken(effects)


def test_greeting_and_new_wording_are_off_without_configuration():
    # Saved runs predate these settings and must replay with their original speech.
    brain = Brain({})
    _, effects = brain.step(0, perception_stats={'speech_ready': True})
    assert not spoken(effects)
    hand = HandObs(1, True, (320, 240), (320, 240), (), (0, 0), 1)
    assert brain._where(hand, (200, 400), (640, 480), False) == 'back on your left'
    assert Brain(ON)._where(hand, (200, 400), (640, 480), False) == 'on your left, closer to you'
    assert Brain(ON)._where(hand, (320, 240), (640, 480), False) == 'right by your hand'


def test_help_keeps_the_current_state():
    brain = Brain(ON)
    brain.step(0, commands=[command('find', 'wallet')])
    state = brain.state
    _, effects = brain.step(1, commands=[command('help')])
    assert brain.state == state and ('help', HELP) in spoken(effects)


def test_volume_steps_and_limits():
    brain = Brain(ON)
    levels = []
    for i in range(3):
        _, effects = brain.step(i, commands=[command('louder')])
        levels += [e.payload['db'] for e in effects if e.kind == 'volume']
    assert levels == [3, 6]
    assert spoken(effects) == [('loudest', "That's the loudest.")]
    for i in range(3, 10):
        _, effects = brain.step(i, commands=[command('quieter')])
    assert brain.volume_db == -12
    assert spoken(effects) == [('quietest', "That's the quietest.")]


def test_brief_mode_drops_chatter_but_answers_requests():
    brain = Brain(ON)
    brain.step(0, commands=[command('brief')])
    brain.step(1, commands=[command('find', 'wallet')])
    _, effects = brain.step(2, commands=[command('recheck')])
    assert not [k for k, _ in spoken(effects) if k == 'checking']
    _, effects = brain.step(3, commands=[command('help')])
    assert spoken(effects)[0][0] == 'help'
    hand = HandObs(5, True, (100, 100), (100, 100), (), (0, 0), 1)
    _, effects = brain.step(4, hand=hand, target=target())
    assert ('found_brief', 'Found the wallet.') in spoken(effects)


def test_finding_the_target_drops_queued_search_prompts():
    brain = Brain(ON)
    brain.step(0, commands=[command('find', 'wallet')])
    hand = HandObs(5, True, (100, 100), (100, 100), (), (0, 0), 1)
    _, effects = brain.step(1, hand=hand, target=target())
    drops = [e.payload['keys'] for e in effects if e.kind == 'cancel_speech' and e.payload]
    assert drops == [['sweep', 'looking']]
    # Without the setting no extra effect appears, so old traces replay unchanged.
    brain = Brain({})
    brain.step(0, commands=[command('find', 'wallet')])
    _, effects = brain.step(1, hand=hand, target=target())
    assert not [e for e in effects if e.kind == 'cancel_speech' and e.payload]


def test_speaker_drops_only_queued_outdated_phrases():
    from reach.tts import NullEngine, Speaker

    class Audio:
        sample_rate = 48000
        def speak(self, samples, ident): pass
        def cancel_speech(self): pass
        def poll_speech_events(self): return []
    speaker = Speaker({'objects': [], 'tts': {'preload': False}}, Audio(), engine_factory=NullEngine)
    try:
        speaker.say('Looking for the wallet.', key='looking', priority=2)
        speaker.say('Bring your hand into view.', key='hand_lost', priority=1)
        speaker.say('Checking.', key='checking', priority=1)
        speaker.drop_waiting(['hand_lost'])
        with speaker._condition:
            assert speaker._current.key == 'looking'
            assert [job.key for job in speaker._waiting] == ['checking']
    finally:
        speaker.stop()


def test_master_level_scales_rendered_output():
    from reach.audio import CueEngine

    class Status:
        output_underflow = False
    engine = CueEngine({'audio': {}})
    from reach.audio import Tone
    engine._tone = Tone(48000, 256, .2)
    engine.set(__import__('reach.types', fromlist=['CueParams']).CueParams(0, 440, 4, True, 0, .2))
    loud = np.zeros((256, 2), np.float32)
    for _ in range(40):
        engine._callback(loud, 256, None, Status())
    engine2 = CueEngine({'audio': {}})
    engine2._tone = Tone(48000, 256, .2)
    engine2.set(__import__('reach.types', fromlist=['CueParams']).CueParams(0, 440, 4, True, 0, .2))
    engine2.set_master_db(-6)
    quiet = np.zeros((256, 2), np.float32)
    for _ in range(40):
        engine2._callback(quiet, 256, None, Status())
    ratio = np.abs(quiet).max()/np.abs(loud).max()
    assert ratio == pytest.approx(10**(-6/20), rel=.02)
    with pytest.raises(ValueError):
        engine2.set_master_db(40)


@pytest.mark.parametrize('message,spoken', [
    ('Hand model missing; set hand.model_path to hand_landmarker.task', 'The hand model is missing.'),
    ('Cannot open camera. Check source, permission and backend', "I can't open the camera."),
    ('No usable frames received', "I can't open the camera."),
    ('Model API key missing', 'The API key is missing.'),
    ('Input name must identify one microphone on the selected host API', 'Microphone unavailable. Use the keys.'),
    ('No output device; use reach devices', None)])
def test_startup_failures_get_a_plain_fix(message, spoken):
    from reach.startup import explain, speak_once
    said, printed = explain(ValueError(message))
    assert said == spoken and printed
    assert not speak_once(said)  # never plays sound during tests
