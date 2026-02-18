"""Embedding wrapper with caching."""
from __future__ import annotations

import asyncio
from typing import List

from sentence_transformers import SentenceTransformer

from cache import SimpleLRUCache
from compliance_utils import hash_text


async def _run_in_thread(func, *args, **kwargs):
    """Run sync function in a thread (Python 3.8 compat: asyncio.to_thread added in 3.9)."""
    if hasattr(asyncio, "to_thread"):
        return await asyncio.to_thread(func, *args, **kwargs)
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


class Embedder:
    def __init__(self, model_name: str, cache: SimpleLRUCache) -> None:
        self._model_name = model_name
        self._cache = cache
        self._model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    async def embed(self, text: str) -> List[float]:
        if not text:
            return []
        cache_key = hash_text(text)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        model = self._get_model()
        embedding = await _run_in_thread(
            lambda: model.encode(
                [text],
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        )
        vector = embedding[0].tolist()
        self._cache.set(cache_key, vector)
        return vector
