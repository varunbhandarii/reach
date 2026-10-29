"""Rate-limited SAM query scheduler."""

from collections.abc import Mapping
from typing import Any


class QueryScheduler:
    """Schedule queries using configured rates, with safe defaults."""

    def __init__(self, cfg: Mapping[str, Any] | None = None) -> None:
        sam_cfg = cfg.get("sam", {}) if isinstance(cfg, Mapping) else {}
        rates = sam_cfg.get("rates_hz", {}) if isinstance(sam_cfg, Mapping) else {}
        self._rates_hz = {"searching": 2.0, "guiding": 0.5, "guiding_near": 2.0}
        if isinstance(rates, Mapping):
            self._rates_hz.update(rates)
        self._last_query: float | None = None

    def should_query(self, state: str, tracker_ok: bool, now: float) -> bool:
        """Return whether a query is due at ``now`` and record scheduled queries."""
        rate_name = state.casefold()
        if not tracker_ok:
            rate_name = "searching"
        rate = float(self._rates_hz.get(rate_name, 2.0))
        if rate <= 0:
            return False

        interval = 1.0 / rate
        if self._last_query is None or now - self._last_query >= interval:
            self._last_query = now
            return True
        return False
