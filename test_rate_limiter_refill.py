"""Token-bucket edges for the compliance rate limiter.

The base suite only checks a one-token limiter after a full minute. These
tests pin partial refill, the burst cap, the non-positive rate floor, and
the fact that a denied call still moves the window forward.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

from rate_limiter import RateLimiter


def _clock(monkeypatch: Any, start: float) -> dict[str, float]:
    now = {"t": start}

    def fake_monotonic() -> float:
        return now["t"]

    monkeypatch.setattr(time, "monotonic", fake_monotonic)
    return now


def _allow(limiter: RateLimiter) -> bool:
    return asyncio.run(limiter.allow())


def test_denied_call_advances_window_so_elapsed_time_is_not_double_counted(
    monkeypatch: Any,
) -> None:
    now = _clock(monkeypatch, 1000.0)
    limiter = RateLimiter(rate_per_minute=1)

    assert _allow(limiter) is True
    assert _allow(limiter) is False

    now["t"] = 1030.0
    assert _allow(limiter) is False

    # 15s after the denial is only 0.75 of a token. Counting from the
    # original success would already be 45s and would grant a token early.
    now["t"] = 1045.0
    assert _allow(limiter) is False

    now["t"] = 1060.0
    assert _allow(limiter) is True
    assert _allow(limiter) is False


def test_long_idle_does_not_burst_past_the_configured_rate(monkeypatch: Any) -> None:
    now = _clock(monkeypatch, 1000.0)
    limiter = RateLimiter(rate_per_minute=2)

    assert _allow(limiter) is True
    assert _allow(limiter) is True
    assert _allow(limiter) is False

    now["t"] = 1600.0
    assert _allow(limiter) is True
    assert _allow(limiter) is True
    assert _allow(limiter) is False


def test_non_positive_rate_floors_to_one_token_per_minute(monkeypatch: Any) -> None:
    now = _clock(monkeypatch, 1000.0)
    for configured in (0, -4):
        limiter = RateLimiter(rate_per_minute=configured)
        assert _allow(limiter) is True
        assert _allow(limiter) is False
        now["t"] += 30.0
        assert _allow(limiter) is False
        now["t"] += 30.0
        assert _allow(limiter) is True


def test_backward_clock_does_not_grant_a_token(monkeypatch: Any) -> None:
    now = _clock(monkeypatch, 1000.0)
    limiter = RateLimiter(rate_per_minute=1)

    assert _allow(limiter) is True
    now["t"] = 940.0
    assert _allow(limiter) is False

    now["t"] = 1000.0
    assert _allow(limiter) is False

    now["t"] = 1060.0
    assert _allow(limiter) is True
