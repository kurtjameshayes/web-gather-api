"""Regression tests for VectorRetriever.retrieve result caching.

A transient embeddings-search failure is stored as an empty list and reused.
An empty embedding vector is not stored, so the next call searches again.
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from cache import SimpleLRUCache
from compliance_config import load_config
from vector_retriever import VectorRetriever


class _CountingEmbedder:
    def __init__(self, vector):
        self.vector = vector
        self.calls = 0

    async def embed(self, text):
        self.calls += 1
        return list(self.vector)


def _retriever(embedder, collection, *, use_embeddings: bool):
    config = load_config()
    config.use_embeddings_collection = use_embeddings
    config.statutes_collection = "statutes"
    config.embeddings_collection = "statute_embeddings"
    config.embedding_vector_field = "embedding"
    config.vector_field = "vector"
    client = MagicMock()
    client.__getitem__.return_value.__getitem__.return_value = collection
    retriever = VectorRetriever(client, embedder, config, SimpleLRUCache(8, 3600))
    return retriever, client


def _statute_row() -> dict:
    return {
        "_id": "stat-1",
        "jurisdiction": "US",
        "title": "Retention",
        "section": "§ 1",
        "section_text": "Keep data for one year.",
        "section_id": "S-1",
        "score": 0.4,
    }


def test_successful_statute_search_is_reused_without_a_second_query() -> None:
    collection = MagicMock()
    collection.aggregate.return_value = [_statute_row()]
    embedder = _CountingEmbedder([0.2, 0.3])
    retriever, client = _retriever(embedder, collection, use_embeddings=False)

    async def run():
        first = await retriever.retrieve("privacy_db", "Keep data", "US", None, 3)
        first.append(first[0])
        second = await retriever.retrieve("privacy_db", "Keep data", "US", None, 3)
        await retriever.retrieve("privacy_db", "Delete data", "US", None, 3)
        await retriever.retrieve("privacy_db", "Keep data", "US", None, 4)
        await retriever.retrieve("other_db", "Keep data", "US", None, 3)
        return first, second

    first, second = asyncio.run(run())

    assert first[0].statute_id == "stat-1"
    assert second is first
    assert len(second) == 2
    assert embedder.calls == 4
    assert collection.aggregate.call_count == 4
    limits = [
        call.args[0][0]["$vectorSearch"]["limit"]
        for call in collection.aggregate.call_args_list
    ]
    assert limits == [3, 3, 4, 3]
    assert [call.args[0] for call in client.__getitem__.call_args_list] == [
        "privacy_db",
        "privacy_db",
        "privacy_db",
        "other_db",
    ]


def test_zero_vector_still_searches_and_empty_vector_does_not_cache() -> None:
    collection = MagicMock()
    collection.aggregate.return_value = [_statute_row()]
    zeros = _CountingEmbedder([0.0])
    retriever, _client = _retriever(zeros, collection, use_embeddings=False)

    async def run_zero():
        return await retriever.retrieve("privacy_db", "Keep data", "US", None, 1)

    found = asyncio.run(run_zero())
    assert found[0].statute_id == "stat-1"
    assert zeros.calls == 1
    assert collection.aggregate.call_count == 1

    collection.aggregate.reset_mock()
    empty = _CountingEmbedder([])
    retriever, _client = _retriever(empty, collection, use_embeddings=False)

    async def run_empty():
        first = await retriever.retrieve("privacy_db", "Keep data", "US", None, 1)
        second = await retriever.retrieve("privacy_db", "Keep data", "US", None, 1)
        return first, second

    first, second = asyncio.run(run_empty())
    assert first == []
    assert second == []
    assert empty.calls == 2
    collection.aggregate.assert_not_called()


def test_embeddings_search_failure_is_cached_as_an_empty_result() -> None:
    collection = MagicMock()
    collection.aggregate.side_effect = RuntimeError("vector index unavailable")
    embedder = _CountingEmbedder([0.5, 0.5])
    retriever, _client = _retriever(embedder, collection, use_embeddings=True)

    async def run():
        first = await retriever.retrieve("privacy_db", "Keep data", "California", None, 2)
        first.append("stale")
        second = await retriever.retrieve("privacy_db", "Keep data", "California", None, 2)
        return first, second

    first, second = asyncio.run(run())

    assert first == ["stale"]
    assert second is first
    assert embedder.calls == 1
    assert collection.aggregate.call_count == 1
    collection.find.assert_not_called()


def test_successful_empty_embeddings_search_is_cached() -> None:
    collection = MagicMock()
    collection.aggregate.return_value = []
    embedder = _CountingEmbedder([0.1])
    retriever, _client = _retriever(embedder, collection, use_embeddings=True)

    async def run():
        first = await retriever.retrieve("privacy_db", "Keep data", "California", None, 2)
        second = await retriever.retrieve("privacy_db", "Keep data", "California", None, 2)
        return first, second

    first, second = asyncio.run(run())

    assert first == []
    assert second is first
    assert embedder.calls == 1
    assert collection.aggregate.call_count == 1
    collection.find.assert_not_called()
