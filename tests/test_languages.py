from string import Formatter
import pytest

from reach.brain import Brain
from reach.phrases import table, translate, speech_language
from reach.session_codec import plain, restore_inputs
from reach.types import Command


def test_hindi_table_completeness_and_template_fields():
    en, hi = table('en'), table('hi')
    assert en.keys() == hi.keys()
    for section in en:
        assert en[section].keys() == hi[section].keys()
    for key in en['phrases']:
        fields = lambda text: {f for _, f, _, _ in Formatter().parse(text) if f}
        assert fields(en['phrases'][key]) == fields(hi['phrases'][key]), key


def test_search_language_and_spoken_object_follow_command_and_reset_to_english():
    b = Brain()
    command = Command('find', 'wallet', 'मेरा बटुआ ढूँढो', 1, 0, 'spark_parse', 'hi', 'बटुआ')
    d, effects = b.step(0, commands=[command])
    speak = next(e for e in effects if e.kind == 'speak')
    assert speak.payload == {'language': 'hi', 'spoken_phrase': 'बटुआ'}
    assert translate(speak.text, speak.key, **speak.payload) == 'बटुआ खोज रही हूँ।'
    assert restore_inputs({'commands': [plain(command)]})['commands'] == (command,)
    _, effects = b.step(1, commands=[Command('find', 'pen', '', 2, 1)])
    assert next(e for e in effects if e.kind == 'speak').payload is None


def test_windows_keeps_english_and_hindi_uses_fixed_templates():
    assert speech_language('hi-IN', 'darwin') == 'hi'
    assert speech_language('hi', 'win32') == 'en'
    assert translate('Found the pen, to your right.', 'found', language='hi') == 'कलम की जगह पता चल गई है: आपके दाएँ।'
    assert translate('You have the wallet.', language='hi', spoken_phrase='मेरा बटुआ') == 'आपका हाथ मेरा बटुआ तक पहुँच गया है।'
    assert translate('I can see a mug in the middle and a pen on your right.', language='hi') == 'मुझे मग बीच में और कलम आपके दाएँ दिख रहे हैं।'
    assert translate('arbitrary generated hint', 'hint', language='hi') is None


def test_mixed_hindi_request_is_not_silently_reduced_to_english_noun():
    from reach.grammar import parse
    assert parse('मेरी pen ढूँढो').kind == 'unknown'


def test_hindi_speaker_selects_voice_and_preloads_both_languages(monkeypatch):
    import numpy as np
    from reach.tts import Speaker, NullEngine
    from test_tts import engine
    monkeypatch.setattr('reach.tts.sys.platform', 'darwin')
    rendered = []
    class Fake(NullEngine):
        def __init__(self, cfg): self.voice = cfg.get('voice')
        def synthesize(self, text, sample_rate):
            rendered.append((self.voice, text))
            return np.ones(20, np.float32)*.2
    audio = engine()
    speaker = Speaker({'objects': [], 'tts': {'voices': {'en': 'English', 'hi': 'Lekha'}}}, audio, engine_factory=Fake)
    try:
        assert speaker.wait_ready(timeout=2)
        assert ('English', 'Ready.') in rendered and ('Lekha', 'तैयार हूँ।') in rendered
        speaker.say('You have the wallet.', key='confirmed', priority=2, language='hi', spoken_phrase='बटुआ')
        assert speaker._current.language == 'hi'
        assert speaker._current.text == 'आपका हाथ बटुआ तक पहुँच गया है।'
    finally:
        speaker.stop()


def test_stripped_replay_removes_spoken_alias_as_well_as_transcript(tmp_path):
    from dataclasses import asdict
    import json
    from reach.replay import replay_session
    from reach.session_codec import replay_config
    command = Command('find', 'wallet', 'private recording words', 1, 0, 'spark_parse', 'hi', 'private spoken alias')
    b = Brain(); d, effects = b.step(0, commands=[command])
    rows = [{'type': kind, 'data': data, 't_mono': 0} for kind, data in (
        ('brain_in', effects[0].payload), ('brain_out', effects[-1].payload), ('decision', asdict(d)))]
    (tmp_path/'events.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (tmp_path/'manifest.json').write_text(json.dumps({'contract_version': '5', 'config': replay_config({})}))
    assert replay_session(tmp_path, strip=tmp_path/'stripped')['status'] == 'ok'
    content = (tmp_path/'stripped/events.jsonl').read_text()
    assert 'private spoken alias' not in content and 'private recording words' not in content
    assert replay_session(tmp_path/'stripped')['status'] == 'ok'


@pytest.mark.parametrize('english,expected',[(None,'LegacyVoice'),('NamedEnglish','NamedEnglish')])
def test_english_voice_map_keeps_the_existing_voice_preference(english,expected):
    from reach.tts import Speaker,NullEngine
    from test_tts import engine
    selected=[]
    class Fake(NullEngine):
        def __init__(self,cfg):selected.append(cfg.get('voice'))
    speaker=Speaker({'objects':[],'tts':{'voice':'LegacyVoice','voices':{'en':english}}},
                    engine(),engine_factory=Fake)
    try:
        assert speaker.wait_ready(timeout=2)
        assert selected==[expected]
    finally:
        speaker.stop()
