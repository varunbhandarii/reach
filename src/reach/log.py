"""Ordered event, manifest and frame writes on a dedicated writer thread."""

from __future__ import annotations

import json
import queue
import threading
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class EventLog:
    """Queue writes; close drains them and surfaces any writer failure."""

    def __init__(self, run_dir: str | Path, trial: str = "t0001", *, save_frames: bool = True) -> None:
        if not trial:
            raise ValueError("trial must be a non-empty string")
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "events.jsonl"
        self.trial = trial
        self.save_frames = save_frames
        self._lock = threading.Lock()
        self._queue = queue.SimpleQueue()
        self._error = None
        self._closed = False
        self._seq = self._last_sequence()
        self._stream = self.path.open("a", encoding="utf-8", newline="\n")
        self._thread = threading.Thread(target=self._write, name="event-writer", daemon=True)
        self._thread.start()

    def _last_sequence(self) -> int:
        if not self.path.exists():
            return 0
        last = ""
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                last = line
        if not last:
            return 0
        try:
            return int(json.loads(last)["seq"]) + 1
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError("cannot continue event log with an invalid last record") from exc

    def _check(self):
        if self._error is not None:
            raise RuntimeError("event writer failed; run data may be incomplete") from self._error
        if self._closed:
            raise ValueError("cannot write to a closed event log")

    def emit(self, event_type: str, data: Mapping[str, Any]) -> int:
        if not isinstance(event_type, str) or not event_type:
            raise ValueError("event type must be a non-empty string")
        if not isinstance(data, Mapping):
            raise TypeError("event data must be a mapping")
        # Serialize now so later mutation by a producer cannot change the event.
        payload = json.dumps(dict(data), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        with self._lock:
            self._check()
            seq = self._seq
            envelope = {"seq": seq, "t_mono": time.perf_counter(),
                        "t_wall": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
                        "trial": self.trial, "type": event_type}
            line = json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))[:-1] + ',"data":' + payload + '}\n'
            self._queue.put(("event", line))
            self._seq += 1
            return seq

    def write_manifest(self, data: Mapping[str, Any]) -> None:
        value = json.dumps(dict(data), ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        with self._lock:
            self._check()
            self._queue.put(("manifest", value))

    def save_frame(self, frame_id: int, jpeg: bytes) -> None:
        if not isinstance(frame_id, int) or frame_id < 0:
            raise ValueError("frame_id must be a nonnegative integer")
        with self._lock:
            self._check()
            if self.save_frames:
                self._queue.put(("frame", (frame_id, bytes(jpeg))))

    def _write(self):
        last_flush = time.monotonic()
        try:
            while True:
                try:
                    item = self._queue.get(timeout=max(0.001, .5 - (time.monotonic() - last_flush)))
                except queue.Empty:
                    item = ("flush", None)
                if item is None:
                    break
                kind, value = item
                if kind == "event":
                    self._stream.write(value)
                elif kind == "manifest":
                    path = self.run_dir / "manifest.json"
                    temporary = path.with_suffix(".json.tmp")
                    temporary.write_text(value, encoding="utf-8")
                    temporary.replace(path)
                elif kind == "frame":
                    frame_id, jpeg = value
                    folder = self.run_dir / "frames"
                    folder.mkdir(exist_ok=True)
                    (folder / f"{frame_id}.jpg").write_bytes(jpeg)
                if time.monotonic() - last_flush >= .5:
                    self._stream.flush()
                    last_flush = time.monotonic()
        except Exception as exc:
            self._error = exc
        finally:
            self._stream.close()

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                self._queue.put(None)
        self._thread.join()
        if self._error is not None:
            raise RuntimeError("event writer failed; run data may be incomplete") from self._error

    def __enter__(self) -> EventLog:
        return self

    def __exit__(self, *args) -> None:
        self.close()
