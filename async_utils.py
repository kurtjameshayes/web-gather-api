"""Shared async utilities."""
from __future__ import annotations

import asyncio
from typing import Any, Callable, TypeVar

T = TypeVar("T")


async def run_in_thread(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run a synchronous function in a thread pool executor.

    Uses asyncio.to_thread when available (Python 3.9+), otherwise falls back
    to loop.run_in_executor.
    """
    if hasattr(asyncio, "to_thread"):
        return await asyncio.to_thread(func, *args, **kwargs)
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
