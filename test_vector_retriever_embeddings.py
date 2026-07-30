"""Regression tests for embeddings-collection retrieval and multi-query dedup."""
from __future__ import annotations

import asyncio
import sys
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

# Avoid loading sentence_transformers in unit tests.
sys.modules["sentence_transformers"] = MagicMock()

from cache import SimpleLRUCache
from compliance_config import load_config
from vector_retriever import StatuteCandidate, VectorRetriever


class FakeCollection:
    def __init__(
        self,
        *,
        aggregate_result: Any = None,
        find_results: List[Dict[str, Any]] | None = None,
    ) -> None:
        self.aggregate_result = [] if aggregate_result is None else aggregate_result
        self.find_results = list(find_results or [])
        self.aggregate_calls: List[List[Dict[str, Any]]] = []
        self.find_calls: List[tuple[Dict[str, Any], Dict[str, Any] | None]] = []

    def aggregate(self, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self.aggregate_calls.append(pipeline)
        if isinstance(self.aggregate_result, Exception):
            raise self.aggregate_result
        return list(self.aggregate_result)

    def find(
        self,
        query: Dict[str, Any],
        projection: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        self.find_calls.append((query, projection))
        return list(self.find_results)


class FakeDatabase:
    def __init__(self, collections: Dict[str, FakeCollection]) -> None:
        self.collections = collections

    def __getitem__(self, name: str) -> FakeCollection:
        return self.collections[name]


class FakeMongo:
    def __init__(self, database: FakeDatabase) -> None:
        self.database = database

    def __getitem__(self, name: str) -> FakeDatabase:
        return self.database


def _make_retriever(
    mongo: FakeMongo,
    *,
    use_embeddings_collection: bool = True,
) -> tuple[VectorRetriever, Any]:
    config = load_config()
    config.use_embeddings_collection = use_embeddings_collection
    config.statute_collection_tag = "statute_chunks"
    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])
    retriever = VectorRetriever(
        mongo,
        embedder=embedder,
        config=config,
        cache=SimpleLRUCache(10, 60),
    )
    return retriever, config


def test_retrieve_from_embeddings_hydrates_parent_and_filters_jurisdiction() -> None:
    """Embeddings path should hydrate statute metadata and drop non-CA results."""
    config = load_config()
    emb_coll = FakeCollection(
        aggregate_result=[
            {
                config.embedding_doc_id_field: "doc-ca",
                config.embedding_text_field: "Retain data no longer than necessary.",
                config.embedding_chunk_id_field: "0",
                config.statute_jurisdiction_field: "California",
                config.statute_section_field: "1798.100",
                config.statute_chunk_header_field: "Right to Know",
                "score": 0.91,
            },
            {
                config.embedding_doc_id_field: "doc-us",
                config.embedding_text_field: "Federal retention rule.",
                config.embedding_chunk_id_field: "1",
                config.statute_jurisdiction_field: "US",
                "score": 0.95,
            },
        ]
    )
    statutes_coll = FakeCollection(
        find_results=[
            {
                config.statute_id_field: "doc-ca",
                "document_id": "doc-ca",
                config.statute_title_field: "CCPA Right to Know",
                config.statute_jurisdiction_field: "California",
            }
        ]
    )
    db = FakeDatabase(
        {
            config.embeddings_collection: emb_coll,
            config.statutes_collection: statutes_coll,
        }
    )
    retriever, config = _make_retriever(FakeMongo(db), use_embeddings_collection=True)

    results = asyncio.run(
        retriever.retrieve(
            database="privacy_db",
            section_text="How long do we keep data?",
            jurisdiction="CA",
            statute_corpus_id=None,
            top_k=5,
        )
    )

    assert len(results) == 1
    assert results[0].statute_id == "doc-ca"
    assert results[0].title == "CCPA Right to Know"
    assert results[0].section_id == "1798.100"
    assert results[0].chunk_header_text == "Right to Know"
    assert results[0].score == 0.91
    assert "necessary" in results[0].chunk_text

    assert len(emb_coll.aggregate_calls) == 1
    vs = emb_coll.aggregate_calls[0][0]["$vectorSearch"]
    assert vs["path"] == config.embedding_vector_field
    assert vs["filter"][config.embedding_collection_tag_field] == "statute_chunks"
    assert config.statute_jurisdiction_field in vs["filter"]

    assert len(statutes_coll.find_calls) == 1
    find_query = statutes_coll.find_calls[0][0]
    assert "$or" in find_query


def test_retrieve_from_embeddings_returns_empty_when_vector_search_fails() -> None:
    """Aggregate failures on the embeddings collection must not raise."""
    config = load_config()
    emb_coll = FakeCollection(aggregate_result=RuntimeError("vector index missing"))
    statutes_coll = FakeCollection()
    db = FakeDatabase(
        {
            config.embeddings_collection: emb_coll,
            config.statutes_collection: statutes_coll,
        }
    )
    retriever, _ = _make_retriever(FakeMongo(db), use_embeddings_collection=True)

    results = asyncio.run(
        retriever.retrieve(
            database="privacy_db",
            section_text="retention",
            jurisdiction="California",
            statute_corpus_id=None,
            top_k=3,
        )
    )

    assert results == []
    assert statutes_coll.find_calls == []


def test_retrieve_by_queries_deduplicates_and_keeps_higher_score() -> None:
    """Multi-query retrieval should dedupe by statute/chunk and keep the best score."""
    config = load_config()
    config.use_embeddings_collection = False
    retriever = VectorRetriever(
        MagicMock(),
        embedder=MagicMock(),
        config=config,
        cache=SimpleLRUCache(10, 60),
    )

    async def fake_retrieve(**kwargs: Any) -> List[StatuteCandidate]:
        query = kwargs["section_text"]
        if query == "right to know":
            return [
                StatuteCandidate(
                    statute_id="s1",
                    jurisdiction="CA",
                    title="A",
                    section_id="100",
                    chunk_text="know",
                    score=0.70,
                    chunk_id="0",
                ),
                StatuteCandidate(
                    statute_id="s2",
                    jurisdiction="CA",
                    title="B",
                    section_id="110",
                    chunk_text="access",
                    score=0.60,
                    chunk_id="1",
                ),
            ]
        return [
            StatuteCandidate(
                statute_id="s1",
                jurisdiction="CA",
                title="A",
                section_id="100",
                chunk_text="know better",
                score=0.85,
                chunk_id="0",
            ),
            StatuteCandidate(
                statute_id="s3",
                jurisdiction="CA",
                title="C",
                section_id="120",
                chunk_text="delete",
                score=0.50,
                chunk_id="2",
            ),
        ]

    retriever.retrieve = fake_retrieve  # type: ignore[method-assign]

    results = asyncio.run(
        retriever.retrieve_by_queries(
            database="privacy_db",
            jurisdiction="CA",
            queries=["right to know", "right to delete"],
            top_k_per_query=2,
        )
    )

    assert [c.statute_id for c in results] == ["s1", "s2", "s3"]
    assert results[0].score == 0.85
    assert results[0].chunk_text == "know better"
    assert len(results) == 3
