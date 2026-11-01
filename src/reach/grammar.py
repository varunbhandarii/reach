"""Pure local command grammar. Unknown conversation produces no action."""

import difflib
import re
from .types import Command

DEFAULT_OBJECTS = ("wallet", "pen", "ball", "apple")
EXACT = {"stop": "stop", "cancel": "cancel", "never mind": "cancel", "nevermind": "cancel",
         "start over": "start_over", "reset": "start_over", "repeat": "repeat", "say again": "repeat",
         "say that again": "repeat", "recheck": "recheck", "check again": "recheck", "look again": "recheck",
         "yes": "yes", "yeah": "yes", "yep": "yes", "correct": "yes", "no": "no", "nope": "no", "wrong": "no"}
FILLERS = ("hey reach", "can you", "could you", "please", "okay", "um", "uh")


def normalize(text):
    text = text.lower().replace("’", "'")
    text = re.sub(r"\bwhere's\b", "where is", text)
    return " ".join(re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text))


def clean_phrase(phrase, vocabulary):
    words = phrase.split()
    while words and words[0] in ("the", "my", "a", "an"):
        words.pop(0)
    phrase = ' '.join(words)
    if not phrase or len(phrase) > 120:
        return None
    matches = difflib.get_close_matches(phrase, vocabulary, n=1, cutoff=.8)
    if matches:
        return matches[0]
    matches = difflib.get_close_matches(words[-1], vocabulary, n=1, cutoff=.8)
    if matches:
        words[-1] = matches[0]
    phrase = ' '.join(words)
    return phrase if len(phrase) <= 120 else None


def parse(transcript, objects=DEFAULT_OBJECTS, vocabulary=(), *, t_mono=0.0, turn_id=None, source="voice"):
    if not isinstance(transcript, str):
        raise TypeError("transcript must be text")
    text = normalize(transcript)
    while True:
        prefix = next((p for p in FILLERS if text == p or text.startswith(p+' ')), None)
        if prefix is None:
            break
        text = text[len(prefix):].strip()
    text = re.sub(r'(?: please)+$', '', text).strip()
    # Recognition sometimes repeats a single short command.
    if text and len(set(text.split())) == 1 and text.split()[0] in EXACT:
        text = text.split()[0]
    command = lambda kind, phrase=None: Command(kind, phrase, transcript, turn_id, t_mono, source)
    if text in EXACT:
        return command(EXACT[text])
    choice = re.fullmatch(r'(?:the )?(left|right|closer|farther|ahead|other)(?: one)?', text)
    if choice:
        return command('choose', 'farther' if choice[1]=='ahead' else choice[1])
    find = re.fullmatch(r'(?:find|look for|where is|locate|search for|show me) (.+)', text)
    known = tuple(normalize(word) for word in objects)
    vocab = sorted(set(known) | {normalize(word) for word in vocabulary})
    candidate = find[1] if find else text if text in known or re.sub(r'^(?:the|my|a|an) ', '', text) in known else None
    if candidate is None or ';' in transcript or re.search(r'\w,\s*\w', transcript):
        return command('unknown')
    phrase = clean_phrase(candidate, vocab)
    return command('find', phrase) if phrase else command('unknown')
