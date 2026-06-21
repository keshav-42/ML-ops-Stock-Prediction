"""Redis cache with graceful degradation.

Predictions are stable within a trading day, so we key on (ticker, date) and set
a TTL that expires at the next NSE close (15:30 IST = 10:00 UTC). If Redis is
unreachable the API keeps serving (cache simply reports 'unavailable').
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, time, timedelta

NSE_CLOSE_UTC = time(10, 0)  # 15:30 IST


def seconds_to_next_close(now: datetime | None = None) -> int:
    """Seconds from now until the next 10:00 UTC (next NSE close). Min 60s."""
    now = now or datetime.now(UTC)
    target = datetime.combine(now.date(), NSE_CLOSE_UTC, tzinfo=UTC)
    if now >= target:
        target += timedelta(days=1)
    return max(60, int((target - now).total_seconds()))


def cache_key(ticker: str, date: str) -> str:
    return f"pred:{ticker}:{date}"


class PredictionCache:
    """Thin wrapper around redis-py; never raises on a backend outage."""

    def __init__(self, url: str | None = None, enabled: bool = True):
        self.enabled = enabled
        self._client = None
        self._status = "disabled"
        if not enabled:
            return
        url = url or os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        try:
            import redis

            self._client = redis.Redis.from_url(url, socket_connect_timeout=1,
                                                socket_timeout=1, decode_responses=True)
            self._client.ping()
            self._status = "connected"
        except Exception:  # noqa: BLE001 - degrade gracefully
            self._client = None
            self._status = "unavailable"

    @property
    def status(self) -> str:
        return self._status

    def get(self, ticker: str, date: str) -> dict | None:
        if self._client is None:
            return None
        try:
            raw = self._client.get(cache_key(ticker, date))
            return json.loads(raw) if raw else None
        except Exception:  # noqa: BLE001
            self._status = "unavailable"
            return None

    def set(self, ticker: str, date: str, value: dict, ttl: int | None = None) -> None:
        if self._client is None:
            return
        try:
            self._client.set(cache_key(ticker, date), json.dumps(value),
                             ex=ttl or seconds_to_next_close())
        except Exception:  # noqa: BLE001
            self._status = "unavailable"
