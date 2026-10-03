"""Regression tests for VectorRetriever embeddings retrieval.

Production config sets use_embeddings_collection, so statute lookup goes
through _retrieve_from_embeddings. The Mongo filter can request any
jurisdiction, but a later pass keeps only the configured jurisdiction
(California by default) and prefers parent-statute fields over the
embedding row.
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

from compliance_config import load_config
from cache import SimpleLRUCache
import vector_retriever
from vector_retriever import VectorRetriever


class _Embedder:
    async def embed(self, text: str) -> list[float]:
        return [0.1, 0.2]


def _config() -> Any:
    cfg = load_config()
    cfg.use_embeddings_collection = True
    cfg.embeddings_collection = "embeddings"
    cfg.statutes_collection = "statutes"
    cfg.vector_index_name = "vector_index"
    cfg.vector_field = "vector"
    cfg.embedding_vector_field = "embedding"
    cfg.embedding_text_field = "chunk_text"
    cfg.embedding_doc_id_field = "document_id"
    cfg.embedding_chunk_id_field = "chunk_index"
    cfg.embedding_collection_tag_field = "collection"
    cfg.statute_collection_tag = ""
    cfg.statute_jurisdiction_field = "jurisdiction"
    cfg.statute_section_field = "section"
    cfg.statute_section_id_field = "section_id"
    cfg.statute_chunk_header_field = "chunk_header_text"
    cfg.statute_title_field = "title"
    cfg.statute_id_field = "_id"
    cfg.max_statute_chunk_chars = 1200
    return cfg


def _retriever(cfg: Any, embeddings: MagicMock, statutes: MagicMock) -> tuple[VectorRetriever, dict]:
    captured: dict = {}

    def aggregate(pipeline):
        captured["pipeline"] = pipeline
        return captured["rows"]

    def find(query):
        captured["find"] = query
        return captured["parents"]

    embeddings.aggregate.side_effect = aggregate
    statutes.find.side_effect = find

    client = MagicMock()
    db = MagicMock()

    def get_coll(name: str) -> MagicMock:
        if name == "embeddings":
            return embeddings
        if name == "statutes":
            return statutes
        raise AssertionError(name)

    client.__getitem__.return_value = db
    db.__getitem__.side_effect = get_coll
    retriever = VectorRetriever(client, _Embedder(), cfg, SimpleLRUCache(8, 60))
    return retriever, captured


def _run(coro):
    return asyncio.run(coro)


def test_post_filter_keeps_california_when_search_filter_is_eu(monkeypatch) -> None:
    monkeypatch.setattr(vector_retriever, "VECTOR_SEARCH_JURISDICTION", "California")
    cfg = _config()
    embeddings, statutes = MagicMock(), MagicMock()
    retriever, captured = _retriever(cfg, embeddings, statutes)
    captured["parents"] = []
    captured["rows"] = [
        {"document_id": "ca-1", "jurisdiction": "California", "chunk_text": "keep", "score": 0.9, "chunk_index": 4},
        {"document_id": "alias", "jurisdiction": "CA", "chunk_text": "alias", "score": 0.4, "chunk_index": 1},
        {"document_id": "eu-1", "jurisdiction": "European Union", "chunk_text": "drop", "score": 0.99, "chunk_index": 2},
        {"document_id": "blank", "jurisdiction": "  ", "chunk_text": "blank", "score": 0.2, "chunk_index": 3},
        {"document_id": "padded", "jurisdiction": " California ", "chunk_text": "padded", "score": 0.3, "chunk_index": 5},
        {"document_id": "missing-jur", "chunk_text": "none", "score": 0.1, "chunk_index": 6},
    ]

    results = _run(
        retriever.retrieve(
            database="privacy_db",
            section_text="privacy rights",
            jurisdiction="EU",
            statute_corpus_id=None,
            top_k=3,
        )
    )

    stage = captured["pipeline"][0]["$vectorSearch"]
    assert stage["index"] == "vector_index"
    assert stage["path"] == "embedding"
    assert stage["limit"] == 3
    assert stage["numCandidates"] == 30
    assert "EU" in stage["filter"]["jurisdiction"]["$in"]
    assert [item.statute_id for item in results] == ["ca-1", "alias", "padded"]
    assert results[0].jurisdiction == "California"
    assert results[2].jurisdiction == " California "
    assert results[0].chunk_id == "4"
    assert all(item.statute_id != "eu-1" for item in results)


def test_parent_statute_fields_win_and_falsy_embedding_fields_fall_back(monkeypatch) -> None:
    monkeypatch.setattr(vector_retriever, "VECTOR_SEARCH_JURISDICTION", "California")
    cfg = _config()
    cfg.max_statute_chunk_chars = 10
    embeddings, statutes = MagicMock(), MagicMock()
    retriever, captured = _retriever(cfg, embeddings, statutes)
    captured["parents"] = [
        {
            "_id": "other",
            "document_id": "p1",
            "jurisdiction": " California ",
            "section": "§ parent",
            "title": "Parent Title",
            "chunk_header_text": "Parent Header",
        },
        {
            "_id": "raw-id",
            "jurisdiction": "California",
            "title": "By Id",
            "section": "§ by-id",
        },
        {
            "_id": "p2-oid",
            "document_id": "p2",
            "jurisdiction": "California",
            "section": "§ ignored",
            "chunk_header_text": "Parent Header 2",
            "title": "Should Not Matter",
        },
        {
            "_id": "p3-oid",
            "document_id": "p3",
            "jurisdiction": "Virginia",
            "section": "§ va",
            "title": "Virginia Parent",
        },
    ]
    captured["rows"] = [
        {
            "document_id": "p1",
            "jurisdiction": "European Union",
            "section": "",
            "section_id": 0,
            "chunk_text": "abcdefghijklmnop",
        },
        {
            "document_id": "raw-id",
            "jurisdiction": "CA",
            "chunk_text": "by id",
            "score": 0.5,
            "chunk_index": 8,
        },
        {
            "document_id": "p2",
            "jurisdiction": "Virginia",
            "section": "§ emb",
            "chunk_header_text": "  Emb  ",
            "chunk_text": "short",
            "score": True,
            "chunk_index": 2,
        },
        {
            "document_id": "p3",
            "jurisdiction": "California",
            "section": "§ emb3",
            "chunk_text": "dropped",
            "score": 0.8,
            "chunk_index": 9,
        },
    ]

    results = _run(
        retriever.retrieve(
            database="privacy_db",
            section_text="parent fields",
            jurisdiction="VA",
            statute_corpus_id=None,
            top_k=2,
        )
    )
    by_id = {item.statute_id: item for item in results}

    assert set(by_id) == {"p1", "raw-id", "p2"}
    assert by_id["p1"].jurisdiction == " California "
    assert by_id["p1"].section_id == "§ parent"
    assert by_id["p1"].chunk_header_text == "Parent Header"
    assert by_id["p1"].title == "Parent Title"
    assert by_id["p1"].chunk_text == "abcdefg..."
    assert by_id["p1"].score == 0.0
    assert by_id["p1"].chunk_id == "0"
    assert by_id["raw-id"].title == "By Id"
    assert by_id["raw-id"].section_id == "§ by-id"
    assert by_id["p2"].section_id == "§ emb"
    assert by_id["p2"].chunk_header_text == "  Emb  "
    assert by_id["p2"].jurisdiction == "California"
    assert by_id["p2"].score == 1.0
    assert by_id["p2"].chunk_id == "2"
    assert captured["find"] == {
        "$or": [
            {"_id": {"$in": ["p1", "raw-id", "p2", "p3"]}},
            {"document_id": {"$in": ["p1", "raw-id", "p2", "p3"]}},
        ]
    }


def test_falsy_document_ids_skip_the_statute_fetch(monkeypatch) -> None:
    monkeypatch.setattr(vector_retriever, "VECTOR_SEARCH_JURISDICTION", "California")
    cfg = _config()
    embeddings, statutes = MagicMock(), MagicMock()
    retriever, captured = _retriever(cfg, embeddings, statutes)
    captured["rows"] = [
        {"jurisdiction": "California", "chunk_text": "orphan", "score": 0.2, "chunk_index": 1},
        {"document_id": 0, "jurisdiction": "California", "chunk_text": "zero", "score": 0.3, "chunk_index": 2},
        {"document_id": None, "jurisdiction": "CA", "chunk_text": None, "score": 0.4, "chunk_index": 3},
    ]
    captured["parents"] = [{"_id": "should-not-load", "jurisdiction": "California", "title": "Nope"}]

    results = _run(
        retriever.retrieve(
            database="privacy_db",
            section_text="no ids",
            jurisdiction="CA",
            statute_corpus_id=None,
            top_k=1,
        )
    )

    statutes.find.assert_not_called()
    assert [(item.statute_id, item.chunk_text, item.title) for item in results] == [
        ("", "orphan", ""),
        ("0", "zero", ""),
        ("None", "None", ""),
    ]


def test_non_numeric_embedding_score_propagates(monkeypatch) -> None:
    monkeypatch.setattr(vector_retriever, "VECTOR_SEARCH_JURISDICTION", "California")
    cfg = _config()
    embeddings, statutes = MagicMock(), MagicMock()
    retriever, captured = _retriever(cfg, embeddings, statutes)
    captured["rows"] = [
        {"document_id": "bad", "jurisdiction": "California", "chunk_text": "x", "score": "bad", "chunk_index": 1}
    ]
    captured["parents"] = []

    try:
        _run(
            retriever.retrieve(
                database="privacy_db",
                section_text="bad score",
                jurisdiction="CA",
                statute_corpus_id=None,
                top_k=1,
            )
        )
    except ValueError:
        return
    raise AssertionError("non-numeric score should propagate")
