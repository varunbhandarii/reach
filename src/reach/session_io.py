"""Command ingress and trial actions, leaving state ownership in the hand worker."""

from dataclasses import asdict
import queue
import time

from .types import Command

TESTED_OBJECTS = frozenset(('wallet', 'pen', 'ball', 'apple'))


def make_command(kind, phrase=None, *, source='key', transcript='', now=None):
    kwargs = dict(kind=kind, phrase=phrase, transcript=transcript, turn_id=None,
                  t_mono=time.perf_counter() if now is None else now)
    if 'source' in Command.__dataclass_fields__:
        kwargs['source'] = source
    return Command(**kwargs)


class TrialActions:
    def __init__(self, runner):
        self.runner = runner
    def set_phrase(self, phrase):
        self.runner.post_command(make_command('find' if phrase else 'cancel', phrase, source='trial'))
    def select_point(self, x, y):
        self.runner.select_point(x, y)
    def announce_trial(self, number, phrase):
        self.runner.speaker.say(f'Trial {number}. Find the {phrase}.', key='trial', priority=1)


class CommandInbox:
    def __init__(self, log):
        self.queue = queue.SimpleQueue()
        self.log = log
    def post(self, command):
        data = asdict(command)
        data.setdefault('source', 'key')
        data['untested_noun'] = command.kind == 'find' and command.phrase not in TESTED_OBJECTS and command.phrase != 'clicked point'
        self.log.emit('command', data)
        self.queue.put(command)
    def drain(self):
        result = []
        while True:
            try:
                result.append(self.queue.get_nowait())
            except queue.Empty:
                return result
