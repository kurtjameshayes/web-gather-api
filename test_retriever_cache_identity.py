"""Retrieval cache identity for falsy corpus ids and the embeddings flag.

The cache key stringifies a falsy ``statute_corpus_id`` to an empty string and
does not include ``use_embeddings_collection``. Those identities decide whether
a later call skips Mongo.
"""
from __future__ import annotations

import asyncio
import sys
from unittest.mock import MagicMock

sys.modules["sentence_transformers"] = MagicMock()

from cache import SimpleLRUCache
from compliance_config import load_config
from vector_retriever import VectorRetriever


class _CountingEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    async def embed(self, text: str) -> list[float]:
        self.calls += 1
        return [0.1, 0.2]


def _retriever():
    collection = MagicMock()
    collection.aggregate.return_value = [
        {
            "_id": "stat-1",
            "jurisdiction": "TX",
            "title": "Duty",
            "section_text": "Controllers must disclose the sale.",
            "section_id": "s-1",
            "score": 0.4,
        }
    ]
    client = MagicMock()
    client.__getitem__.return_value.__getitem__.return_value = collection
    config = load_config()
    config.use_embeddings_collection = False
    embedder = _CountingEmbedder()
    retriever = VectorRetriever(client, embedder, config, SimpleLRUCache(8, 60))
    return retriever, embedder, collection, config


def _search(retriever: VectorRetriever, corpus_id):
    return retriever.retrieve(
        database="privacy_db",
        section_text="We sell personal information.",
        jurisdiction="TX",
        statute_corpus_id=corpus_id,
        top_k=3,
    )


def test_falsy_corpus_ids_share_one_uncached_search() -> None:
    retriever, embedder, collection, config = _retriever()

    first = asyncio.run(_search(retriever, None))
    second = asyncio.run(_search(retriever, ""))
    third = asyncio.run(_search(retriever, 0))
    fourth = asyncio.run(_search(retriever, False))

    assert embedder.calls == 1
    assert collection.aggregate.call_count == 1
    assert second is first
    assert third is first
    assert fourth is first
    assert first[0].statute_id == "stat-1"

    vector_filter = collection.aggregate.call_args[0][0][0]["$vectorSearch"]["filter"]
    assert "$or" not in vector_filter
    assert vector_filter[config.statute_jurisdiction_field] == "TX"
    assert collection.aggregate.call_args[0][0][0]["$vectorSearch"]["limit"] == 3


def test_whitespace_corpus_id_is_a_separate_search() -> None:
    retriever, embedder, collection, config = _retriever()

    asyncio.run(_search(retriever, None))
    asyncio.run(_search(retriever, "  "))

    assert embedder.calls == 2
    assert collection.aggregate.call_count == 2
    corpus_field = config.statute_corpus_field
    vector_filter = collection.aggregate.call_args[0][0][0]["$vectorSearch"]["filter"]
    assert vector_filter["$or"] == [
        {corpus_field: "  "},
        {f"metadata.{corpus_field}": "  "},
    ]


def test_embeddings_flag_is_not_part_of_the_cache_key() -> None:
    retriever, embedder, collection, config = _retriever()

    cached = asyncio.run(_search(retriever, "corpus-9"))
    config.use_embeddings_collection = True
    again = asyncio.run(_search(retriever, "corpus-9"))

    assert again is cached
    assert embedder.calls == 1
    assert collection.aggregate.call_count == 1
    vector_stage = collection.aggregate.call_args[0][0][0]["$vectorSearch"]
    assert vector_stage["path"] == config.vector_field
    assert "embedding" not in vector_stage["path"]
