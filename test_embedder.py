"""Regression tests for Embedder cache-miss and lazy model loading."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.modules.setdefault("sentence_transformers", MagicMock())

from cache import SimpleLRUCache
from embedder import Embedder


async def _run_inline(func, *args, **kwargs):
    return func(*args, **kwargs)


@pytest.mark.anyio
async def test_embedder_cache_miss_encodes_and_stores_vector() -> None:
    """On cache miss, Embedder encodes text, caches the vector, and returns it."""
    cache = SimpleLRUCache(10, 60)
    embedder = Embedder("test-model", cache)
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.25, 0.75]])
    embedder._model = mock_model

    with patch("embedder.hash_text", return_value="key-1"), patch(
        "embedder.run_in_thread",
        side_effect=_run_inline,
    ):
        result = await embedder.embed("uncached text")

    assert result == [0.25, 0.75]
    mock_model.encode.assert_called_once_with(
        ["uncached text"],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    assert cache.get("key-1") == [0.25, 0.75]


@pytest.mark.anyio
async def test_embedder_lazy_loads_model_once() -> None:
    """_get_model constructs SentenceTransformer once and reuses it."""
    cache = SimpleLRUCache(10, 60)
    embedder = Embedder("lazy-model", cache)
    constructed = MagicMock()
    constructed.encode.return_value = np.array([[1.0, 0.0]])

    with patch("embedder.SentenceTransformer", return_value=constructed) as ctor, patch(
        "embedder.hash_text",
        side_effect=["a", "b"],
    ), patch(
        "embedder.run_in_thread",
        side_effect=_run_inline,
    ):
        first = await embedder.embed("one")
        second = await embedder.embed("two")

    assert first == [1.0, 0.0]
    assert second == [1.0, 0.0]
    ctor.assert_called_once_with("lazy-model")
    assert constructed.encode.call_count == 2
