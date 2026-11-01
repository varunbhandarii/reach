from dataclasses import replace
from hashlib import sha256
import json
from types import SimpleNamespace as NS

import cv2
import numpy as np
import pytest

from reach.cli import parser, configure
from reach.runner import Runner
from reach.session_report import summarize
from reach.trials import Trials
from reach.types import HandObs, Frame, Decision
from reach.controller import decide


class FakeBrain:
    """Only tests runner wiring; the actual reducer has its own owner and tests."""
    def __init__(self, cfg, objects):
        self.phrase = None
        self.seq = 0
    def step(self, now, *, hand, target, perception_stats, commands, speaking, frame_size, flipped):
        effects = []
        def effect(kind, text=None, key=None, payload=None):
            return NS(kind=kind, text=text, key=key, payload=payload, priority=0)
        for command in commands:
            if command.kind == 'find':
                self.phrase = command.phrase
                effects.extend([effect('set_phrase', self.phrase), effect('perception_state', key='SEARCHING')])
            elif command.kind in ('cancel', 'stop'):
                self.phrase = None
                effects.extend([effect('set_phrase', None), effect('perception_state', key='IDLE')])
        frame = Frame(hand.frame_id if hand else 0, now, 0, *frame_size, 'synthetic', 'file')
        result = decide(hand, target if self.phrase else None, frame, now=now, seq=self.seq, flipped=flipped)
        self.seq += 1
        return result, effects


class FakeHand:
    upside_down = False
    fingertip_raw_px = (20, 40)
    last_detect_ms = 2.0
    held = False
    def __init__(self, cfg):
        pass
    def observe(self, frame, image, target):
        return HandObs(frame.id, True, (20, 40), (20, 50), (), (0, 0), 1)
    def stats(self):
        return {"hand_ms_p50": 2, "hand_ms_p95": 2, "upside_down": False}
    def close(self):
        pass


class FakeAudio:
    def __init__(self, cfg):
        self.running = False
        self.cue = None
    def start(self):
        self.running = True
    def set(self, cue):
        self.cue = cue
    def earcon(self, name):
        pass
    def stats(self):
        return {"underflows": 0, "cpu_load": 0, "reported_output_latency_ms": None, "error": None}
    def stop(self):
        self.running = False


def video(tmp_path):
    path = tmp_path/"input.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20, (160, 120))
    assert writer.isOpened()
    rng = np.random.default_rng(4)
    base = rng.integers(0, 255, (120, 160, 3), np.uint8)
    for i in range(12):
        image = base.copy()
        cv2.rectangle(image, (i*2, 5), (i*2+10, 15), (0, 255, 0), -1)
        writer.write(image)
    writer.release()
    return path


def test_file_loop_click_no_network_and_saved_inputs(tmp_path, monkeypatch):
    from reach.sam_client import SamClient
    monkeypatch.setattr(SamClient, "segment", lambda *a: pytest.fail("click mode must not use SAM"))
    args = parser().parse_args(["run", "--mount", "handheld", "--source", str(video(tmp_path)),
        "--headless", "--mute", "--click", "100", "50", "--log-dir", str(tmp_path/"runs")])
    cfg = configure(args)
    runner = Runner(cfg, args, hand_factory=FakeHand, audio_factory=FakeAudio, brain_factory=FakeBrain)
    assert runner.run() == 0
    result = summarize(runner.run_dir)
    assert result["completed"] and not result["audio_measured"]
    assert result["saved_frame_events"] == 12 and result["errors"] == []
    rows = [json.loads(line) for line in runner.log.path.read_text().splitlines()]
    decisions = [r["data"] for r in rows if r["type"] == "decision"]
    assert any(d["cue"] for d in decisions)
    assert all(d["hand_frame_id"] is not None for d in decisions)
    assert [r["seq"] for r in rows] == list(range(len(rows)))
    for row in (r["data"] for r in rows if r["type"] == "frame"):
        jpeg = (runner.run_dir/"frames"/f"{row['frame_id']}.jpg").read_bytes()
        assert sha256(jpeg).hexdigest() == row["jpeg_sha256"]


def test_hand_thread_failure_is_recorded_and_cleanup_finishes(tmp_path):
    class BrokenHand(FakeHand):
        def observe(self, *args):
            raise RuntimeError("test failure")
    args = parser().parse_args(["run", "--mount", "fixed", "--source", str(video(tmp_path)),
        "--headless", "--mute", "--log-dir", str(tmp_path/"runs")])
    runner = Runner(configure(args), args, hand_factory=BrokenHand, audio_factory=FakeAudio, brain_factory=FakeBrain)
    assert runner.run() == 1
    assert not runner._hand_thread.is_alive()
    assert summarize(runner.run_dir)["errors"][0]["where"] == "hand_controller"


def test_trials_need_explicit_touch_and_time_out():
    now, events = [0], []
    log = NS(emit=lambda kind, data: events.append((kind, data)))
    perception = NS(set_phrase=lambda phrase: None, select_point=lambda *a: None)
    trial = Trials(2, "phrase", ["wallet", "pen"], log, clock=lambda: now[0])
    trial.prepare(perception, NS(width=640, height=480))
    assert not trial.start(False)
    assert trial.start(True)
    now[0] = 10
    trial.finish(True, "reported_touch", perception)
    assert trial.results[0]["success"] and trial.results[0]["outcome_source"] == "manual_touch_key"
    trial.prepare(perception, NS(width=640, height=480))
    assert trial.start(True)
    now[0] = 56
    trial.tick(perception)
    assert trial.finished and not trial.results[1]["success"]


def test_mount_is_required_and_headless_trials_rejected():
    with pytest.raises(SystemExit):
        parser().parse_args(["run"])
    args = parser().parse_args(["run", "--mount", "handheld", "--headless", "--trials", "5"])
    with pytest.raises(ValueError):
        configure(args)


def test_trial_target_controls_preserve_scripted_phrases():
    calls = []
    runner = Runner.__new__(Runner)
    runner.log = NS(emit=lambda *args: None)
    runner.audio = FakeAudio({})
    runner.perception = NS(select_point=lambda *args: calls.append(("point", args)),
                           set_phrase=lambda value: calls.append(("phrase", value)),
                           force_query=lambda: calls.append(("recheck",)))
    runner.cfg = {"objects": ["wallet", "pen"]}
    runner._trial = NS(mode="phrase", started=None, prepared=True, phrase='wallet')
    runner.inbox = NS(post=lambda command: calls.append((command.kind, command.phrase)))
    runner._mouse(cv2.EVENT_LBUTTONDOWN, 50, 60, None, None)
    runner._key(ord("1"), None)
    assert calls == []
    runner._key(ord("r"), None)
    assert calls == [("recheck", None)]
    runner._trial.mode = "click"
    runner._trial.phrase = 'clicked point'
    runner._mouse(cv2.EVENT_LBUTTONDOWN, 50, 60, None, None)
    assert calls[-1] == ("find", 'clicked point')
    runner._trial.started = 1
    before = len(calls)
    runner._mouse(cv2.EVENT_LBUTTONDOWN, 80, 90, None, None)
    assert len(calls) == before


def test_no_frame_intervals_still_tick_session_timers(tmp_path):
    import time
    class TickingBrain(FakeBrain):
        def step(self, *args, **kwargs):
            result = super().step(*args, **kwargs)
            if self.seq >= 3:
                runner._stop.set()
            return result
    class PausedCamera:
        source = 'file'
        def __init__(self, cfg, log=None):
            self.log = log
        def open(self):
            pass
        def start(self):
            pass
        def latest(self):
            return None
        def wait_newer(self, last, timeout):
            time.sleep(timeout)
            return None
        def stats(self):
            return {'error': None, 'eof': False, 'capture_fps': 0}
        def stop(self):
            pass
    args = parser().parse_args(['run','--mount','fixed','--headless','--mute','--duration','3',
                               '--log-dir',str(tmp_path/'runs')])
    runner = Runner(configure(args), args, camera_factory=PausedCamera, hand_factory=FakeHand,
                    audio_factory=FakeAudio, brain_factory=TickingBrain)
    assert runner.run() == 0
    assert runner.brain.seq >= 3
    assert not runner._hand_thread.is_alive()


def test_injected_voice_posts_to_same_session_queue(tmp_path):
    from reach.session_io import make_command
    class Voice:
        def __init__(self, cfg, log, post, **kwargs):
            self.post = post
        def start(self):
            self.post(make_command('find', 'wallet', source='voice', transcript='find the wallet'))
        def stop(self):
            pass
        def stats(self):
            return {'connected': True, 'stereo_ok': True, 'input_name': 'synthetic microphone'}
    class Perception:
        def __init__(self, *args):
            self.phrase = None
        def start(self):
            pass
        def stop(self):
            pass
        def latest(self):
            return None
        def on_hand(self, hand):
            pass
        def set_phrase(self, phrase):
            self.phrase = phrase
        def set_state(self, state):
            pass
        def stats(self):
            return {'error': None}
    args = parser().parse_args(['run','--mount','fixed','--source',str(video(tmp_path)),
        '--headless','--mute','--voice','--log-dir',str(tmp_path/'runs')])
    runner = Runner(configure(args), args, hand_factory=FakeHand, audio_factory=FakeAudio,
        perception_factory=Perception, voice_factory=Voice, brain_factory=FakeBrain)
    assert runner.run() == 0
    assert runner.brain.phrase == 'wallet'
    commands = summarize(runner.run_dir)['commands']
    assert commands[0]['kind'] == 'find' and commands[0]['phrase'] == 'wallet'


def test_real_shared_reducer_file_integration_when_available(tmp_path, monkeypatch):
    pytest.importorskip('reach.brain', reason='Shared reducer dependency has not been integrated yet')
    from reach.sam_client import SamClient
    monkeypatch.setattr(SamClient, 'segment', lambda *a: pytest.fail('Click mode cannot send images'))
    args = parser().parse_args(['run','--mount','fixed','--source',str(video(tmp_path)),
        '--headless','--mute','--no-voice','--contact','off','--click','100','50',
        '--log-dir',str(tmp_path/'runs')])
    runner = Runner(configure(args), args, hand_factory=FakeHand, audio_factory=FakeAudio)
    assert runner.run() == 0
    rows = [json.loads(line) for line in runner.log.path.read_text().splitlines()]
    assert any(r['type']=='decision' and r['data'].get('cue') for r in rows)



def test_instance_effects_reach_perception_with_source_geometry():
    from reach.types import Effect
    calls=[]
    runner=Runner.__new__(Runner)
    runner.perception=NS(choose=lambda ident:calls.append(('choose',ident)),
                         exclude=lambda frame,box:calls.append(('exclude',frame,box)))
    runner._effects([Effect('exclude_instance',payload={'frame_id':12,'box_xyxy':[1,2,3,4]}),
                     Effect('choose_instance',payload={'sam_id':'next'})],0)
    assert calls==[('exclude',12,[1,2,3,4]),('choose','next')]
