"""Regression tests for the statutes-collection vector retrieval path."""
from __future__ import annotations

import asyncio
import sys
from typing import Any
from unittest.mock import MagicMock

# Avoid loading sentence_transformers in unit tests.
sys.modules["sentence_transformers"] = MagicMock()

from cache import SimpleLRUCache
from compliance_config import load_config
from vector_retriever import VectorRetriever


class FakeCollection:
    def __init__(self, aggregate_results: list[dict[str, Any]] | None = None) -> None:
        self.aggregate_results = aggregate_results or []
        self.aggregate_calls: list[list[dict[str, Any]]] = []

    def aggregate(self, pipeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.aggregate_calls.append(pipeline)
        return list(self.aggregate_results)


class FakeDatabase:
    def __init__(self, collections: dict[str, FakeCollection]) -> None:
        self.collections = collections

    def __getitem__(self, name: str) -> FakeCollection:
        return self.collections[name]


class FakeMongo:
    def __init__(self, databases: dict[str, FakeDatabase]) -> None:
        self.databases = databases

    def __getitem__(self, name: str) -> FakeDatabase:
        return self.databases[name]


class DummyEmbedder:
    async def embed(self, text: str) -> list[float]:
        return [0.25, 0.75]


def test_retrieve_from_statutes_builds_jurisdiction_and_corpus_filters() -> None:
    """Direct statutes retrieval must apply jurisdiction aliases and corpus scoping."""
    config = load_config()
    config.use_embeddings_collection = False
    config.max_statute_chunk_chars = 40

    statutes = FakeCollection(
        aggregate_results=[
            {
                config.statute_id_field: "stat-ca-1",
                config.statute_jurisdiction_field: "California",
                config.statute_title_field: "Right to Know",
                config.statute_text_field: "A business shall disclose the categories of personal information collected from consumers over a long period.",
                config.statute_section_field: "1798.100",
                config.statute_section_id_field: "sec-1",
                config.statute_chunk_header_field: "Disclosure Duties",
                "score": 0.88,
            }
        ]
    )
    mongo = FakeMongo({"privacy_db": FakeDatabase({config.statutes_collection: statutes})})
    retriever = VectorRetriever(mongo, DummyEmbedder(), config, SimpleLRUCache(8, 60))

    results = asyncio.run(
        retriever.retrieve(
            database="privacy_db",
            section_text="What do we disclose?",
            jurisdiction="CA",
            statute_corpus_id="ccpa-corpus",
            top_k=3,
        )
    )

    assert len(results) == 1
    assert results[0].statute_id == "stat-ca-1"
    assert results[0].section_id == "1798.100"
    assert results[0].chunk_id == "sec-1"
    assert results[0].chunk_header_text == "Disclosure Duties"
    assert results[0].score == 0.88
    assert results[0].chunk_text.endswith("...")
    assert len(results[0].chunk_text) <= config.max_statute_chunk_chars

    assert len(statutes.aggregate_calls) == 1
    vector_search = statutes.aggregate_calls[0][0]["$vectorSearch"]
    assert vector_search["index"] == config.vector_index_name
    assert vector_search["path"] == config.vector_field
    assert vector_search["limit"] == 3
    assert vector_search["numCandidates"] == 30
    assert vector_search["queryVector"] == [0.25, 0.75]

    filter_doc = vector_search["filter"]
    jurisdiction_filter = filter_doc[config.statute_jurisdiction_field]
    assert isinstance(jurisdiction_filter, dict)
    assert "$in" in jurisdiction_filter
    assert "CA" in jurisdiction_filter["$in"]
    assert any(value.lower() == "california" for value in jurisdiction_filter["$in"])
    assert filter_doc["$or"] == [
        {config.statute_corpus_field: "ccpa-corpus"},
        {f"metadata.{config.statute_corpus_field}": "ccpa-corpus"},
    ]


def test_retrieve_from_statutes_returns_empty_when_embedding_missing() -> None:
    """Empty embeddings should short-circuit before hitting Mongo aggregate."""
    config = load_config()
    config.use_embeddings_collection = False
    statutes = FakeCollection(aggregate_results=[{"score": 1.0}])
    mongo = FakeMongo({"privacy_db": FakeDatabase({config.statutes_collection: statutes})})

    class EmptyEmbedder:
        async def embed(self, text: str) -> list[float]:
            return []

    retriever = VectorRetriever(mongo, EmptyEmbedder(), config, SimpleLRUCache(8, 60))
    results = asyncio.run(
        retriever.retrieve(
            database="privacy_db",
            section_text="unused",
            jurisdiction="US",
            statute_corpus_id=None,
            top_k=2,
        )
    )

    assert results == []
    assert statutes.aggregate_calls == []
