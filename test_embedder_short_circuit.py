"""Embedder empty-input short-circuit vs whitespace encode path."""
from __future__ import annotations

import asyncio
import sys
from unittest.mock import MagicMock

import numpy as np

sys.modules["sentence_transformers"] = MagicMock()

from cache import SimpleLRUCache
from embedder import Embedder


def test_embedder_empty_and_none_skip_model_and_cache() -> None:
    """Empty or None text returns [] without touching the cache or SentenceTransformer."""
    cache = MagicMock(spec=SimpleLRUCache)
    embedder = Embedder("test-model", cache)
    embedder._model = MagicMock()

    assert asyncio.run(embedder.embed("")) == []
    assert asyncio.run(embedder.embed(None)) == []  # type: ignore[arg-type]
    cache.get.assert_not_called()
    cache.set.assert_not_called()
    embedder._model.encode.assert_not_called()


def test_embedder_whitespace_still_encodes() -> None:
    """Whitespace-only text is truthy and is embedded (not treated as empty)."""
    cache = MagicMock(spec=SimpleLRUCache)
    cache.get.return_value = None
    embedder = Embedder("test-model", cache)
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.25, 0.75]])
    embedder._model = mock_model

    result = asyncio.run(embedder.embed("   "))

    assert result == [0.25, 0.75]
    mock_model.encode.assert_called_once()
    cache.set.assert_called_once()
