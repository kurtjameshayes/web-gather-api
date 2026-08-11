"""Edge-case regression tests for SimpleLRUCache and RateLimiter clamps."""
from __future__ import annotations

import asyncio
import time
from typing import Any

from cache import SimpleLRUCache
from rate_limiter import RateLimiter


def test_simple_lru_cache_clamps_non_positive_constructor_args() -> None:
    """Zero/negative max_size and ttl are clamped to 1 to keep cache usable."""
    cache = SimpleLRUCache(max_size=0, ttl_seconds=0)
    assert cache._max_size == 1
    assert cache._ttl_seconds == 1

    cache.set("only", "value")
    assert cache.get("only") == "value"
    cache.set("next", "evicts")
    assert cache.get("only") is None
    assert cache.get("next") == "evicts"


def test_simple_lru_cache_set_existing_key_refreshes_lru_order(monkeypatch: Any) -> None:
    """Updating an existing key must move it to MRU so older keys evict first."""
    now = 1000.0
    monkeypatch.setattr(time, "time", lambda: now)

    cache = SimpleLRUCache(max_size=2, ttl_seconds=60)
    cache.set("a", "1")
    cache.set("b", "2")
    # Refresh "a" so "b" becomes the LRU victim on the next insert.
    cache.set("a", "1-updated")
    cache.set("c", "3")

    assert cache.get("a") == "1-updated"
    assert cache.get("b") is None
    assert cache.get("c") == "3"


def test_rate_limiter_clamps_zero_rate(monkeypatch: Any) -> None:
    """rate_per_minute <= 0 still allows one token (clamped to 1)."""
    now = 1000.0
    monkeypatch.setattr(time, "monotonic", lambda: now)

    limiter = RateLimiter(rate_per_minute=0)
    assert limiter._rate == 1
    assert asyncio.run(limiter.allow()) is True
    assert asyncio.run(limiter.allow()) is False
    now = 1060.0
    assert asyncio.run(limiter.allow()) is True
