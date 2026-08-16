"""Regression tests for VectorRetriever.retrieve cache and dispatch.

Lower-level statute/embeddings retrieve helpers are covered elsewhere.
This pins the shared wrapper: cache hit, empty-embedding short-circuit
(without caching), and embeddings-vs-statutes dispatch.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from cache import SimpleLRUCache
from compliance_config import load_config
from compliance_utils import hash_text
from vector_retriever import StatuteCandidate, VectorRetriever


def _candidate() -> StatuteCandidate:
    return StatuteCandidate(
        statute_id="stat-1",
        jurisdiction="CA",
        title="Retention",
        section_id="R-1",
        chunk_text="Retain data no longer than necessary.",
        score=0.91,
        chunk_id="c-1",
    )


def _retriever(embedder, config=None) -> VectorRetriever:
    return VectorRetriever(
        MagicMock(),
        embedder,
        config or load_config(),
        SimpleLRUCache(10, 60),
    )


def test_retrieve_returns_empty_when_embedding_missing() -> None:
    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[])
    retriever = _retriever(embedder)
    retriever._retrieve_from_statutes = AsyncMock()
    retriever._retrieve_from_embeddings = AsyncMock()

    results = asyncio.run(
        retriever.retrieve(
            database="privacy_db",
            section_text="We retain data.",
            jurisdiction="CA",
            statute_corpus_id=None,
            top_k=3,
        )
    )

    assert results == []
    retriever._retrieve_from_statutes.assert_not_awaited()
    retriever._retrieve_from_embeddings.assert_not_awaited()
    # Empty embeddings must not be cached so a later successful embed can run.
    assert retriever._cache.get(hash_text("privacy_db:CA::3:We retain data.")) is None


def test_retrieve_cache_hit_skips_embed_and_search() -> None:
    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[0.1, 0.2])
    config = load_config()
    config.use_embeddings_collection = False
    retriever = _retriever(embedder, config)
    cached = [_candidate()]
    retriever._retrieve_from_statutes = AsyncMock(return_value=cached)
    retriever._retrieve_from_embeddings = AsyncMock()

    first = asyncio.run(
        retriever.retrieve(
            database="privacy_db",
            section_text="We retain data.",
            jurisdiction="CA",
            statute_corpus_id=None,
            top_k=3,
        )
    )
    second = asyncio.run(
        retriever.retrieve(
            database="privacy_db",
            section_text="We retain data.",
            jurisdiction="CA",
            statute_corpus_id=None,
            top_k=3,
        )
    )

    assert first == cached
    assert second == cached
    assert embedder.embed.await_count == 1
    assert retriever._retrieve_from_statutes.await_count == 1
    retriever._retrieve_from_embeddings.assert_not_awaited()


def test_retrieve_dispatches_to_embeddings_collection() -> None:
    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[0.4, 0.5])
    config = load_config()
    config.use_embeddings_collection = True
    retriever = _retriever(embedder, config)
    expected = [_candidate()]
    retriever._retrieve_from_embeddings = AsyncMock(return_value=expected)
    retriever._retrieve_from_statutes = AsyncMock()

    results = asyncio.run(
        retriever.retrieve(
            database="privacy_db",
            section_text="Disclosure of sale.",
            jurisdiction="CA",
            statute_corpus_id="corpus-1",
            top_k=2,
        )
    )

    assert results == expected
    retriever._retrieve_from_embeddings.assert_awaited_once_with(
        "privacy_db",
        [0.4, 0.5],
        "CA",
        "corpus-1",
        2,
    )
    retriever._retrieve_from_statutes.assert_not_awaited()
