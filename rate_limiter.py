"""Simple in-memory token bucket rate limiter."""
from __future__ import annotations

import asyncio
import time


class RateLimiter:
    def __init__(self, rate_per_minute: int) -> None:
        self._rate = max(1, rate_per_minute)
        self._allowance = float(self._rate)
        self._last_check = time.monotonic()
        self._lock = asyncio.Lock()

    async def allow(self) -> bool:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_check
            self._last_check = now
            self._allowance += elapsed * (self._rate / 60.0)
            if self._allowance > self._rate:
                self._allowance = float(self._rate)
            if self._allowance < 1.0:
                return False
            self._allowance -= 1.0
            return True
