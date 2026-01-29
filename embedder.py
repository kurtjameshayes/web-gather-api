"""Embedding wrapper with caching."""
from __future__ import annotations

import asyncio
from typing import List

from sentence_transformers import SentenceTransformer

from cache import SimpleLRUCache
from compliance_utils import hash_text


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
        embedding = await asyncio.to_thread(
            model.encode,
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        vector = embedding[0].tolist()
        self._cache.set(cache_key, vector)
        return vector
