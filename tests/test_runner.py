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
from reach.types import HandObs


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
    runner = Runner(cfg, args, hand_factory=FakeHand, audio_factory=FakeAudio)
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
    runner = Runner(configure(args), args, hand_factory=BrokenHand, audio_factory=FakeAudio)
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
