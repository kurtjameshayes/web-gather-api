"""Simple in-memory LRU cache with TTL support."""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Optional


class SimpleLRUCache:
    def __init__(self, max_size: int = 512, ttl_seconds: int = 3600) -> None:
        self._max_size = max(1, max_size)
        self._ttl_seconds = max(1, ttl_seconds)
        self._lock = threading.Lock()
        self._data: OrderedDict[str, tuple[Any, float]] = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        now = time.time()
        with self._lock:
            item = self._data.get(key)
            if not item:
                return None
            value, expires_at = item
            if expires_at < now:
                self._data.pop(key, None)
                return None
            self._data.move_to_end(key)
            return value

    def set(self, key: str, value: Any) -> None:
        now = time.time()
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = (value, now + self._ttl_seconds)
            if len(self._data) > self._max_size:
                self._data.popitem(last=False)
