"""Embedder cache-miss behavior for the vector used by statute retrieval.

Empty text returns before the model and is not cached. Whitespace is embedded
as-is. The stored value is the Python list from the first row, and a repeat
call returns that same list without encoding again.
"""
from __future__ import annotations

import asyncio
import sys
from unittest.mock import MagicMock

sys.modules.setdefault("sentence_transformers", MagicMock())

import numpy as np

import embedder as embedder_module
from cache import SimpleLRUCache
from compliance_utils import hash_text
from embedder import Embedder


def test_empty_text_skips_model_and_whitespace_is_cached_as_a_list() -> None:
    constructed: list[str] = []
    encoded: list[tuple] = []

    class _Model:
        def encode(self, texts, convert_to_numpy=False, normalize_embeddings=False):
            encoded.append((list(texts), convert_to_numpy, normalize_embeddings))
            return np.array([[0.2, 0.4]])

    def _constructor(name: str) -> _Model:
        constructed.append(name)
        return _Model()

    cache = SimpleLRUCache(4, 60)
    embedder = Embedder("mini-test", cache)
    embedder_module_model = embedder_module.SentenceTransformer

    embedder_module.SentenceTransformer = _constructor
    try:
        assert asyncio.run(embedder.embed("")) == []
        assert asyncio.run(embedder.embed(None)) == []
        assert constructed == []
        assert cache.get(hash_text("")) is None
        assert cache.get(hash_text("  ")) is None

        first = asyncio.run(embedder.embed("  "))
        second = asyncio.run(embedder.embed("  "))
        other = asyncio.run(embedder.embed("x"))
    finally:
        embedder_module.SentenceTransformer = embedder_module_model

    assert first == [0.2, 0.4]
    assert second is first
    assert other == [0.2, 0.4]
    assert other is not first
    assert constructed == ["mini-test"]
    assert encoded == [
        (["  "], True, True),
        (["x"], True, True),
    ]
    assert cache.get(hash_text("  ")) is first
    assert cache.get(hash_text("x")) is other
    assert cache.get(hash_text(" x")) is None
