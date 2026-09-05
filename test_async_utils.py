"""Regression tests for the shared run_in_thread helper.

audit_logger, embedder, compliance_storage, adaptive feedback, and the LLM
client all dispatch Mongo/model work through this helper. A broken fallback
(Python 3.8 / missing asyncio.to_thread) would fail those paths together.
"""
from __future__ import annotations

import asyncio

import pytest

from async_utils import run_in_thread


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_run_in_thread_passes_args_and_kwargs() -> None:
    def add(left: int, right: int, extra: int = 0) -> int:
        return left + right + extra

    assert _run(run_in_thread(add, 2, 3, extra=4)) == 9


def test_run_in_thread_falls_back_to_executor_when_to_thread_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if hasattr(asyncio, "to_thread"):
        monkeypatch.delattr(asyncio, "to_thread")

    seen: list[str] = []

    def fake_run_in_executor(_executor, func):
        seen.append("executor")
        result = func()
        fut: asyncio.Future = asyncio.Future()
        fut.set_result(result)
        return fut

    class _Loop:
        def run_in_executor(self, executor, func):
            return fake_run_in_executor(executor, func)

    monkeypatch.setattr(asyncio, "get_event_loop", lambda: _Loop())

    def work(value: int) -> int:
        return value * 2

    assert _run(run_in_thread(work, 21)) == 42
    assert seen == ["executor"]


def test_run_in_thread_propagates_sync_exceptions() -> None:
    def boom() -> None:
        raise ValueError("thread work failed")

    with pytest.raises(ValueError, match="thread work failed"):
        _run(run_in_thread(boom))
