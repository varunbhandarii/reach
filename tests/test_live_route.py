from types import SimpleNamespace as NS

from reach.runner import Runner
from reach.speech_devices import AudioRouteMonitor, route_ok


def test_live_route_loss_recovery_device_change_and_unknown_do_not_latch():
    routes = iter([{'device_id': 7, 'rate': 48000}, {'device_id': 7, 'rate': 24000},
                   None, {'device_id': 7, 'rate': 48000},
                   {'device_id': 9, 'rate': 48000}, {'device_id': 7, 'rate': 48000}])
    monitor = AudioRouteMonitor(lambda: next(routes), platform='darwin')
    assert monitor.poll(0)['ok']
    assert monitor.poll(1.999) is None
    assert monitor.poll(2)['transition'] == 'lost'
    assert monitor.poll(4)['ok'] is False  # failed metadata is not recovery
    assert monitor.poll(6)['transition'] == 'recovered'
    assert monitor.poll(8)['transition'] == 'lost'
    assert monitor.poll(10)['transition'] == 'recovered'


def test_windows_wired_route_does_not_probe_coreaudio():
    monitor = AudioRouteMonitor(lambda: 1/0, platform='win32')
    assert monitor.poll(0) == {'rate': None, 'device_id': None, 'start_device_id': None,
                             'ok': True, 'status': 'not_required', 'transition': None, 'error': None}


def test_unknown_query_is_labeled_and_invalid_rates_are_rejected():
    def unavailable(): raise OSError('synthetic')
    monitor = AudioRouteMonitor(unavailable, platform='darwin')
    assert monitor.poll(0)['status'] == 'unknown'
    for rate in (0, 16000, 24000, float('nan'), True, None):
        assert not route_ok({'device_id': 1, 'rate': rate}, {'device_id': 1})


def test_runner_announces_each_loss_once_and_blocks_trial_start():
    calls, logs = [], []
    runner = Runner.__new__(Runner)
    routes = iter([{'device_id': 7, 'rate': 48000}, {'device_id': 7, 'rate': 24000},
                   {'device_id': 7, 'rate': 24000}, {'device_id': 7, 'rate': 48000}])
    runner._route = AudioRouteMonitor(lambda: next(routes), platform='darwin')
    runner._route_ok = True
    runner.log = NS(emit=lambda kind, data: logs.append((kind, data)))
    runner.audio = NS(set=lambda cue: calls.append(('cue', cue)))
    runner.speaker = NS(say=lambda text, **kwargs: calls.append(('speech', kwargs['key'])))
    runner._trial = NS(start=lambda has_target: calls.append(('trial', has_target)))
    runner.perception = NS(latest=lambda: None)
    runner._check_audio_route(0)
    runner._check_audio_route(2)
    runner._key(ord(' '), None)
    runner._check_audio_route(4)
    assert not any(kind == 'trial' for kind, _ in calls)
    runner._check_audio_route(6)
    runner._key(ord(' '), None)
    assert [value for kind, value in calls if kind == 'speech'] == ['stereo_lost', 'stereo_back']
    assert ('trial', False) in calls
    assert len([1 for kind, _ in logs if kind == 'audio_route']) == 4
    assert any(kind == 'trial_block' for kind, _ in logs)
