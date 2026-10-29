"""Guards around LLM calls: client timeout, in-flight concurrency, and statute truncation."""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from compliance_config import load_config
from compliance_suite_service import ComplianceSuiteService
from compliance_utils import safe_truncate
from llm_client import LLM_TIMEOUT_SECONDS, AnthropicLLMClient


def _service(concurrency: int) -> ComplianceSuiteService:
    config = load_config()
    config.llm_concurrency = concurrency
    return ComplianceSuiteService(
        MagicMock(),
        config,
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )


def test_anthropic_client_sets_call_timeout(monkeypatch) -> None:
    """Hung model calls must time out; the client is built with the shared limit."""
    created: dict[str, object] = {}

    class _FakeAnthropic:
        def __init__(self, *, api_key: str, timeout: float) -> None:
            created["api_key"] = api_key
            created["timeout"] = timeout

    monkeypatch.setattr("llm_client.anthropic.Anthropic", _FakeAnthropic)

    AnthropicLLMClient("test-key", load_config())

    assert created == {"api_key": "test-key", "timeout": LLM_TIMEOUT_SECONDS}
    assert LLM_TIMEOUT_SECONDS == 60


def test_llm_semaphore_is_created_on_first_use() -> None:
    """Construction must not bind a semaphore to an event loop that is not running."""
    service = _service(2)
    assert service._ComplianceSuiteService__semaphore is None

    async def _touch() -> asyncio.Semaphore:
        first = service._sem
        second = service._sem
        assert first is second
        assert first._value == 2
        return first

    semaphore = asyncio.run(_touch())
    assert semaphore._value == 2


def test_llm_semaphore_serializes_when_concurrency_is_one() -> None:
    """A second LLM call waits until the in-flight call releases the semaphore."""
    service = _service(1)
    progress: list[str] = []

    async def _run() -> None:
        semaphore = service._sem
        assert semaphore._value == 1

        async def _second() -> None:
            progress.append("started")
            async with semaphore:
                progress.append("acquired")

        async with semaphore:
            waiter = asyncio.create_task(_second())
            for _ in range(50):
                await asyncio.sleep(0)
                if progress:
                    break
            assert progress == ["started"]
            assert not waiter.done()
        await waiter

    asyncio.run(_run())
    assert progress == ["started", "acquired"]


def test_llm_semaphore_of_zero_starts_locked() -> None:
    """Non-positive concurrency is not floored, unlike the per-minute rate limiter."""
    service = _service(0)

    async def _touch() -> asyncio.Semaphore:
        return service._sem

    semaphore = asyncio.run(_touch())
    assert semaphore._value == 0
    assert semaphore.locked() is True


def test_safe_truncate_non_positive_limit_keeps_statute_text() -> None:
    """A zero or negative cap passes the full statute excerpt through to the LLM."""
    text = "x" * 40
    assert safe_truncate(text, 0) is text
    assert safe_truncate(text, -3) == text
    assert safe_truncate("", 0) == ""


def test_safe_truncate_below_ellipsis_budget_does_not_cap() -> None:
    """Limits under 3 use a negative slice, so the configured cap is not applied."""
    text = "abcdefghij"
    assert safe_truncate(text, 1) == "abcdefgh..."
    assert safe_truncate(text, 2) == "abcdefghi..."
    assert safe_truncate(text, 3) == "..."


def test_safe_truncate_rstrips_the_cut_before_ellipsis() -> None:
    """Whitespace at the cut is removed so the prompt does not end on a dangling space."""
    assert safe_truncate("abcd  xyz", 8) == "abcd..."
    assert safe_truncate("hello", 5) == "hello"
    assert safe_truncate("hello  ", 7) == "hello  "
