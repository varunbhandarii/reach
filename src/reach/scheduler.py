"""State-aware query rates with one-request and rate-limit guards."""

import math


class QueryScheduler:
    def __init__(self, cfg=None):
        self.rates = {"searching": 2.0, "guiding": .5, "guiding_near": 2.0}
        self.rates.update((cfg or {}).get("sam", {}).get("rates_hz", {}))
        if any(not math.isfinite(float(r)) or float(r) < 0 for r in self.rates.values()):
            raise ValueError("query rates must be finite and nonnegative")
        self._last = None
        self._backoff_until = 0.0
        self._force = False

    def reset(self):
        self._last = None
        self._force = False

    def force(self):
        self._force = True

    def backoff(self, now, seconds):
        self._backoff_until = max(self._backoff_until, now + max(0, seconds))

    def should_query(self, state, tracker_ok, now, *, in_flight=False, near=False):
        state = state.casefold()
        if state not in ("searching", "guiding", "guiding_near", "target_lost", "hand_lost"):
            return False
        if in_flight or now < self._backoff_until:
            return False
        name = "guiding_near" if near or not tracker_ok else state
        if name in ("target_lost", "hand_lost"):
            name = "searching" if not tracker_ok else "guiding"
        rate = float(self.rates.get(name, 0))
        if rate <= 0:
            return False
        if self._force or self._last is None or now-self._last >= 1/rate:
            self._last, self._force = now, False
            return True
        return False
