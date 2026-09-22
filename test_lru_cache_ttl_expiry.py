"""SimpleLRUCache TTL expiry: expired entries are dropped on get.

#205 / #225 cover constructor clamps and MRU refresh on set. Expiry on get
(when expires_at < now) was still unpinned: a regression that returned stale
values would poison embedding/model caches used by search and indexing.
"""
from __future__ import annotations

import time
from typing import Any

from cache import SimpleLRUCache


def test_lru_cache_get_drops_expired_entries(monkeypatch: Any) -> None:
    """After TTL elapses, get() removes the key and returns None."""
    now = 1_000.0
    monkeypatch.setattr(time, "time", lambda: now)

    cache = SimpleLRUCache(max_size=4, ttl_seconds=10)
    cache.set("model", "v1")
    assert cache.get("model") == "v1"

    now = 1_010.5  # past expires_at = 1010.0
    assert cache.get("model") is None
    # Expired key is popped; a later get stays a miss without resurrecting it.
    assert cache.get("model") is None


def test_lru_cache_expired_slot_does_not_block_new_inserts(monkeypatch: Any) -> None:
    """Expired entries freed on get do not keep max_size occupancy forever."""
    now = 5_000.0
    monkeypatch.setattr(time, "time", lambda: now)

    cache = SimpleLRUCache(max_size=1, ttl_seconds=5)
    cache.set("old", "a")
    now = 5_010.0
    assert cache.get("old") is None

    cache.set("new", "b")
    assert cache.get("new") == "b"
    assert cache.get("old") is None
