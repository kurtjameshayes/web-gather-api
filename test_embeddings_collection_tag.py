"""Embeddings retrieval collection-tag precedence."""
from __future__ import annotations

import asyncio
import sys
from typing import Any
from unittest.mock import MagicMock

sys.modules["sentence_transformers"] = MagicMock()

from cache import SimpleLRUCache
from compliance_config import load_config
from vector_retriever import VectorRetriever


class _Embedder:
    async def embed(self, text: str) -> list[float]:
        return [0.2, 0.1]


def _retriever(tag: str) -> tuple[VectorRetriever, MagicMock]:
    config = load_config()
    config.use_embeddings_collection = True
    config.statute_collection_tag = tag
    config.embedding_collection_tag_field = "collection"
    config.embedding_vector_field = "embedding"
    config.embeddings_collection = "statute_embeddings"
    mongo = MagicMock()
    coll = MagicMock()
    coll.aggregate.return_value = []
    mongo.__getitem__.return_value.__getitem__.return_value = coll
    cache = SimpleLRUCache(4, 60)
    return VectorRetriever(mongo, _Embedder(), config, cache), coll


def _filter(coll: MagicMock) -> dict[str, Any]:
    pipeline = coll.aggregate.call_args[0][0]
    stage = pipeline[0]["$vectorSearch"]
    assert stage["path"] == "embedding"
    return stage["filter"]


def _retrieve(retriever: VectorRetriever, corpus_id: Any) -> None:
    asyncio.run(
        retriever.retrieve(
            database="privacy-compliance",
            section_text="notice at collection",
            jurisdiction="CA",
            statute_corpus_id=corpus_id,
            top_k=3,
        )
    )


def test_corpus_id_overrides_configured_collection_tag() -> None:
    retriever, coll = _retriever("statutes")
    _retrieve(retriever, "corp-1")
    filt = _filter(coll)
    assert filt["collection"] == "corp-1"
    assert "statute_corpus_id" not in filt
    assert "$or" not in filt


def test_blank_corpus_id_uses_configured_collection_tag() -> None:
    retriever, coll = _retriever("statutes")
    _retrieve(retriever, "")
    assert _filter(coll)["collection"] == "statutes"


def test_missing_corpus_id_uses_configured_collection_tag() -> None:
    retriever, coll = _retriever("statutes")
    _retrieve(retriever, None)
    assert _filter(coll)["collection"] == "statutes"


def test_numeric_zero_corpus_id_falls_through_to_configured_tag() -> None:
    retriever, coll = _retriever("statutes")
    _retrieve(retriever, 0)
    assert _filter(coll)["collection"] == "statutes"


def test_whitespace_corpus_id_is_not_stripped() -> None:
    retriever, coll = _retriever("statutes")
    _retrieve(retriever, "  ")
    assert _filter(coll)["collection"] == "  "


def test_whitespace_configured_tag_is_kept_when_corpus_is_omitted() -> None:
    retriever, coll = _retriever("  ")
    _retrieve(retriever, None)
    assert _filter(coll)["collection"] == "  "
