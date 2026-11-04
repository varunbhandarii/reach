from pathlib import Path

from reach.replay import replay_session

SAMPLE = Path(__file__).resolve().parents[1]/'sample_trial'


def test_published_sample_trial_replays_exactly():
    result = replay_session(SAMPLE)
    assert result['status'] == 'ok', result
    assert result['steps'] == 964
