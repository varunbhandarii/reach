from types import SimpleNamespace as NS

from reach.session_io import CommandInbox, make_command
from reach.trials import Trials


def test_all_input_sources_use_one_logged_queue():
    rows = []
    inbox = CommandInbox(NS(emit=lambda kind, data: rows.append((kind, data))))
    for source in ('voice', 'key', 'trial', 'ptt'):
        inbox.post(make_command('find', 'wallet', source=source))
    assert [c.kind for c in inbox.drain()] == ['find']*4
    assert all(kind == 'command' and not data['untested_noun'] for kind,data in rows)
    inbox.post(make_command('find', 'cup'))
    assert rows[-1][1]['untested_noun']


def test_voice_trial_announces_but_does_not_search_until_spoken():
    now, actions, events = [1.0], [], []
    adapter = NS(announce_trial=lambda n,p: actions.append(('announce', n, p)),
                 set_phrase=lambda p: actions.append(('find', p)))
    log = NS(emit=lambda kind,data: events.append((kind,data)))
    trials = Trials(1, 'voice', ['wallet'], log, clock=lambda: now[0])
    trials.prepare(adapter, None)
    assert actions == [('announce', 1, 'wallet')]
    assert not trials.start(False)
    assert trials.start(True)
    now[0] = 3
    trials.note_contact(now[0])
    now[0] = 4
    trials.note_contact(now[0])
    trials.finish(True, 'reported_touch', adapter)
    assert trials.results[0]['contact_fired_s'] == 2
    assert trials.results[0]['outcome_source'] == 'manual_touch_key'
