"""Checks for structured event recording."""

import json
from datetime import datetime

import pytest

from reach.log import EventLog


def test_writes_utf8_json_lines_with_ordered_sequence(tmp_path):
    with EventLog(tmp_path, trial="trial-42") as event_log:
        assert event_log.emit("session_start", {"label": "café"}) == 0
        assert event_log.emit("frame", {"frame_id": 3, "instances": []}) == 1

    lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]

    assert len(records) == 2
    assert records[0]["seq"] == 0
    assert records[0]["trial"] == "trial-42"
    assert records[0]["type"] == "session_start"
    assert records[0]["data"] == {"label": "café"}
    assert isinstance(records[0]["t_mono"], float)
    assert records[1]["seq"] == 1
    assert records[1]["type"] == "frame"
    assert records[1]["data"] == {"frame_id": 3, "instances": []}
    datetime.fromisoformat(records[0]["t_wall"].replace("Z", "+00:00"))


def test_reopens_and_continues_sequence(tmp_path):
    first = EventLog(tmp_path)
    first.emit("session_start", {})
    first.close()

    second = EventLog(tmp_path)
    assert second.emit("trial_end", {}) == 1
    second.close()

    records = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [record["seq"] for record in records] == [0, 1]


def test_rejects_invalid_data_and_writes_nothing(tmp_path):
    event_log = EventLog(tmp_path)
    with pytest.raises(TypeError, match="mapping"):
        event_log.emit("frame", [])
    event_log.close()

    assert (tmp_path / "events.jsonl").read_text(encoding="utf-8") == ""


def test_concurrent_producers_and_pending_writes_drain(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    with EventLog(tmp_path) as log:
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(lambda i: log.emit("test", {"i": i}), range(300)))
        log.write_manifest({"models": {"sam": "sam-3.1"}})
        log.save_frame(7, b"jpeg")
    records = [json.loads(line) for line in log.path.read_text().splitlines()]
    assert [r["seq"] for r in records] == list(range(300))
    assert len({r["data"]["i"] for r in records}) == 300
    assert (tmp_path / "frames/7.jpg").read_bytes() == b"jpeg"
    assert json.loads((tmp_path / "manifest.json").read_text())["models"]["sam"] == "sam-3.1"
    log.close()
    with pytest.raises(ValueError):
        log.emit("late", {})


def test_frame_saving_can_be_disabled(tmp_path):
    with EventLog(tmp_path, save_frames=False) as log:
        log.save_frame(2, b"jpeg")
    assert not (tmp_path / "frames").exists()
