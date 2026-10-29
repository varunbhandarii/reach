"""Append structured events to a UTF-8 JSON Lines file."""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any


class EventLog:
    """Write timestamped events to ``events.jsonl`` under a run directory."""

    def __init__(self, run_dir: str | Path, trial: str = "t0001") -> None:
        if not trial:
            raise ValueError("trial must be a non-empty string")

        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "events.jsonl"
        self.trial = trial
        self._lock = threading.Lock()
        self._seq = self._last_sequence()
        self._stream = self.path.open("a", encoding="utf-8", newline="\n")
        self._closed = False

    def _last_sequence(self) -> int:
        if not self.path.exists():
            return 0

        last_line = ""
        with self.path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    last_line = line
        if not last_line:
            return 0

        try:
            previous = json.loads(last_line)
            return int(previous["seq"]) + 1
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"cannot continue event log with an invalid last record: {self.path}") from exc

    def emit(self, event_type: str, data: Mapping[str, Any]) -> int:
        """Append one event and return its sequence number."""
        if not isinstance(event_type, str) or not event_type:
            raise ValueError("event type must be a non-empty string")
        if not isinstance(data, Mapping):
            raise TypeError("event data must be a mapping")

        with self._lock:
            if self._closed:
                raise ValueError("cannot emit to a closed event log")

            sequence = self._seq
            record = {
                "seq": sequence,
                "t_mono": perf_counter(),
                "t_wall": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
                "trial": self.trial,
                "type": event_type,
                "data": dict(data),
            }
            line = json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            self._stream.write(line + "\n")
            self._stream.flush()
            self._seq += 1
            return sequence

    def close(self) -> None:
        """Flush and close the event file; repeated calls are safe."""
        with self._lock:
            if not self._closed:
                self._stream.flush()
                self._stream.close()
                self._closed = True

    def __enter__(self) -> EventLog:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
