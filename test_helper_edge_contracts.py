"""Leftover helper contracts for truncation, slugs, cache clamps, and rate limits.

Base tests cover happy-path helpers. These pin surprising edges that affect
LLM context size, PII-adjacent truncation, cache memory, and rate limiting.
"""
from __future__ import annotations

import asyncio
import hashlib
import time
from typing import Any

from cache import SimpleLRUCache
from compliance_utils import (
    clamp,
    extract_json_block,
    hash_text,
    jurisdiction_filter_values,
    normalize_jurisdiction,
    safe_truncate,
    slugify,
    validate_collection_name,
)
from rate_limiter import RateLimiter


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_safe_truncate_non_positive_max_chars_returns_full_text() -> None:
    text = "unbounded statute chunk that would otherwise be sent to the LLM"
    assert safe_truncate(text, 0) == text
    assert safe_truncate(text, -5) == text


def test_slugify_empty_or_punctuation_uses_fallback() -> None:
    assert slugify("", "section") == "section"
    assert slugify("   ", "section_1") == "section_1"
    assert slugify("!!!", "section") == "section"
    assert slugify("Data Retention") == "data_retention"


def test_extract_json_block_empty_or_missing_object_is_none() -> None:
    assert extract_json_block("") is None
    assert extract_json_block("no braces here") is None
    assert extract_json_block('prefix {"ok": true} suffix') == '{"ok": true}'


def test_normalize_jurisdiction_unknown_uppercases_original() -> None:
    assert normalize_jurisdiction("unknown-place") == "UNKNOWN-PLACE"
    assert normalize_jurisdiction("") == ""
    assert normalize_jurisdiction("  eu  ") == "EU"


def test_jurisdiction_filter_values_empty_is_empty_list() -> None:
    assert jurisdiction_filter_values("") == []
    assert jurisdiction_filter_values(None) == []  # type: ignore[arg-type]


def test_clamp_hash_and_collection_name_edges() -> None:
    assert clamp(-0.5) == 0.0
    assert clamp(0.25) == 0.25
    assert hash_text("abc") == hashlib.sha256(b"abc").hexdigest()
    assert validate_collection_name("") is False
    assert validate_collection_name("ok_name") is True


def test_lru_cache_clamps_max_size_and_ttl_and_updates_in_place(monkeypatch: Any) -> None:
    now = 1000.0

    def fake_time() -> float:
        return now

    monkeypatch.setattr(time, "time", fake_time)
    cache = SimpleLRUCache(max_size=0, ttl_seconds=0)
    cache.set("only", "value")
    assert cache.get("only") == "value"
    cache.set("other", "evicts")
    assert cache.get("only") is None
    assert cache.get("other") == "evicts"

    sized = SimpleLRUCache(max_size=2, ttl_seconds=10)
    sized.set("a", 1)
    sized.set("b", 2)
    sized.set("a", 11)
    sized.set("c", 3)
    assert sized.get("a") == 11
    assert sized.get("b") is None
    assert sized.get("c") == 3


def test_rate_limiter_zero_rate_clamps_to_one_token() -> None:
    limiter = RateLimiter(rate_per_minute=0)
    assert _run(limiter.allow()) is True
    assert _run(limiter.allow()) is False
