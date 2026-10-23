"""Argument forwarding for the shared thread helper.

Audit logging, storage, embeddings, retrieval, and the LLM client all call
this helper. The executor fallback must keep positional and keyword
arguments when asyncio.to_thread is unavailable.
"""
from __future__ import annotations

import asyncio

from async_utils import run_in_thread


def _record(a: int, b: int = 0) -> int:
    return a + b


def test_run_in_thread_forwards_args_and_kwargs() -> None:
    assert asyncio.run(run_in_thread(_record, 2, b=3)) == 5


def test_run_in_thread_executor_fallback_forwards_kwargs(monkeypatch) -> None:
    monkeypatch.delattr(asyncio, "to_thread", raising=False)
    assert asyncio.run(run_in_thread(_record, 4, b=6)) == 10
