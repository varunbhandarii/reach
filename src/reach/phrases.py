"""Checked-in speech templates; no translation request is made at runtime."""

from functools import lru_cache
import re
from string import Formatter
import sys
import yaml

from .config import PROJECT_ROOT


@lru_cache
def table(language):
    with (PROJECT_ROOT/'config'/'phrases'/f'{language}.yaml').open(encoding='utf-8') as stream:
        return yaml.safe_load(stream)


def speech_language(language, platform=None):
    code = (language or 'en').split('-')[0].casefold()
    return 'hi' if (platform or sys.platform) == 'darwin' and code == 'hi' else 'en'


def _pattern(template):
    pieces = []
    for literal, field, _, _ in Formatter().parse(template):
        pieces.append(re.escape(literal))
        if field:
            pieces.append(f'(?P<{field}>.+?)')
    return re.compile(''.join(pieces))


def translate(text, key=None, *, language='en', spoken_phrase=None):
    if language == 'en':
        return text
    english, target = table('en'), table(language)
    ordered = sorted(english['phrases'], key=lambda name: name != key)
    for name in ordered:
        match = _pattern(english['phrases'][name]).fullmatch(text)
        if match is None:
            continue
        values = match.groupdict()
        for field in ('object', 'name'):
            if field in values:
                noun = re.sub(r'^(?:the|a|an) ', '', values[field])
                values[field] = (spoken_phrase if field == 'object' and spoken_phrase
                                 else target['objects'].get(noun, noun))
        for field, section in [('where', 'directions'), ('count', 'counts'), ('choice_word', 'choices')]:
            if field in values:
                values[field] = target[section].get(values[field], values[field])
        if 'prompt' in values:
            values['prompt'] = translate(values['prompt'], language=language)
        if 'items' in values:
            items = []
            for part in re.split(r', | and ', values['items']):
                item = re.fullmatch(r'a (.+) (on your left|on your right|in the middle)', part)
                if item:
                    noun, position = item.groups()
                    items.append(target['objects'].get(noun, noun)+' '+target['directions'][position])
            values['items'] = ' और '.join(items) if items else values['items']
        return target['phrases'][name].format(**values)
    # Model hints have no checked translation. Do not speak arbitrary English
    # hints in a Hindi session; requested-object confirmation is still spoken.
    return None if key == 'hint' else text
