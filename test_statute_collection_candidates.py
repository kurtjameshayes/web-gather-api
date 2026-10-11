"""Statute-collection retrieval: candidate field mapping and multi-query dedup.

The embeddings-collection path is covered elsewhere. These tests pin the
default statutes-collection path, which still feeds section identity, quotes,
and scores into compliance analysis.
"""
from __future__ import annotations

import asyncio
import sys
from typing import Any, List
from unittest.mock import MagicMock

import pytest

sys.modules["sentence_transformers"] = MagicMock()

from cache import SimpleLRUCache
from compliance_config import load_config
from vector_retriever import VectorRetriever


class _Embedder:
    def __init__(self, vector: List[float] | None = None) -> None:
        self.calls: List[str] = []
        self._vector = [0.2, 0.4] if vector is None else vector

    async def embed(self, text: str) -> List[float]:
        self.calls.append(text)
        return list(self._vector)


def _config(**overrides: Any):
    config = load_config()
    config.use_embeddings_collection = False
    config.statutes_collection = "statutes"
    config.vector_index_name = "vector_index"
    config.vector_field = "vector"
    config.statute_text_field = "section_text"
    config.statute_title_field = "title"
    config.statute_id_field = "_id"
    config.statute_section_field = "section"
    config.statute_section_id_field = "section_id"
    config.statute_jurisdiction_field = "jurisdiction"
    config.statute_chunk_header_field = "chunk_header_text"
    config.max_statute_chunk_chars = 1200
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _retriever(rows: Any, **config_overrides: Any):
    collection = MagicMock()
    if isinstance(rows, list) and rows and isinstance(rows[0], list):
        collection.aggregate.side_effect = rows
    else:
        collection.aggregate.return_value = rows
    database = MagicMock()
    database.__getitem__.return_value = collection
    client = MagicMock()
    client.__getitem__.return_value = database
    embedder = _Embedder()
    retriever = VectorRetriever(
        client,
        embedder,
        _config(**config_overrides),
        SimpleLRUCache(8, 60),
    )
    return retriever, collection, embedder, client, database


def test_statute_rows_map_section_score_header_and_truncation() -> None:
    rows = [
        {
            "_id": "id-1",
            "section": "§ 1798.100",
            "section_id": "sid-1",
            "section_text": "short",
            "title": "Know",
            "jurisdiction": "CA",
            "score": 0.5,
            "chunk_header_text": "  Header  ",
        },
        {
            "_id": "id-2",
            "section": 0,
            "section_id": "fallback-id",
            "section_text": "body",
            "title": "Delete",
            "jurisdiction": "",
            "chunk_header_text": "",
        },
        {
            "section": "",
            "section_id": None,
            "section_text": None,
            "title": None,
            "score": True,
        },
        {
            "_id": "id-4",
            "section": "Sec",
            "section_id": 0,
            "section_text": "ABC    xyz",
            "title": "Trim",
            "jurisdiction": "TX",
            "score": 0.25,
            "chunk_header_text": "H",
        },
    ]
    retriever, collection, embedder, client, database = _retriever(
        rows, max_statute_chunk_chars=8
    )

    results = asyncio.run(
        retriever.retrieve(
            database="privacy_db",
            section_text="right to know",
            jurisdiction="TX",
            statute_corpus_id=None,
            top_k=4,
        )
    )

    client.__getitem__.assert_called_with("privacy_db")
    database.__getitem__.assert_called_with("statutes")
    pipeline = collection.aggregate.call_args.args[0]
    vector_search = pipeline[0]["$vectorSearch"]
    assert vector_search["index"] == "vector_index"
    assert vector_search["path"] == "vector"
    assert vector_search["queryVector"] == [0.2, 0.4]
    assert vector_search["limit"] == 4
    assert vector_search["numCandidates"] == 40
    assert pipeline[1]["$project"]["chunk_header_text"] == 1
    assert pipeline[1]["$project"]["section_text"] == 1
    assert embedder.calls == ["right to know"]

    formal, falsy_section, empty_ids, truncated = results

    assert formal.statute_id == "id-1"
    assert formal.section_id == "§ 1798.100"
    assert formal.chunk_id == "sid-1"
    assert formal.chunk_text == "short"
    assert formal.chunk_header_text == "  Header  "
    assert formal.score == 0.5
    assert formal.title == "Know"
    assert formal.jurisdiction == "CA"

    assert falsy_section.section_id == "fallback-id"
    assert falsy_section.chunk_id == "fallback-id"
    assert falsy_section.score == 0.0
    assert falsy_section.jurisdiction == ""
    assert falsy_section.chunk_header_text == ""

    assert empty_ids.statute_id == ""
    assert empty_ids.section_id == ""
    assert empty_ids.chunk_id == "None"
    assert empty_ids.chunk_text == "None"
    assert empty_ids.title == "None"
    assert empty_ids.jurisdiction == ""
    assert empty_ids.score == 1.0

    assert truncated.section_id == "Sec"
    assert truncated.chunk_id == "0"
    assert truncated.chunk_text == "ABC..."
    assert truncated.score == 0.25


def test_exact_length_text_is_not_truncated_and_nonpositive_cap_keeps_text() -> None:
    text = "12345678"
    retriever, _, _, _, _ = _retriever(
        [
            {
                "_id": "exact",
                "section": "S",
                "section_id": "1",
                "section_text": text,
                "score": 0.1,
            }
        ],
        max_statute_chunk_chars=8,
    )
    exact = asyncio.run(
        retriever.retrieve("db", "q", "TX", None, 1)
    )
    assert exact[0].chunk_text == text

    retriever, _, _, _, _ = _retriever(
        [
            {
                "_id": "full",
                "section": "S",
                "section_id": "1",
                "section_text": "ABCDEFGHIJKLMNO",
                "score": 0.1,
            }
        ],
        max_statute_chunk_chars=0,
    )
    kept = asyncio.run(retriever.retrieve("db", "q", "TX", None, 1))
    assert kept[0].chunk_text == "ABCDEFGHIJKLMNO"


def test_blank_header_field_is_omitted_from_the_projection() -> None:
    retriever, collection, _, _, _ = _retriever(
        [{"_id": "id", "section_id": "s", "section_text": "t", "score": 1}],
        statute_chunk_header_field="",
    )
    results = asyncio.run(retriever.retrieve("db", "q", "TX", None, 1))
    project = collection.aggregate.call_args.args[0][1]["$project"]
    assert "chunk_header_text" not in project
    assert "" not in project
    assert results[0].chunk_header_text == ""


def test_non_numeric_score_is_not_cached() -> None:
    retriever, collection, embedder, _, _ = _retriever(
        [{"_id": "id", "section_id": "s", "section_text": "t", "score": "nope"}]
    )
    with pytest.raises(ValueError):
        asyncio.run(retriever.retrieve("db", "same query", "TX", None, 1))
    with pytest.raises(TypeError):
        collection.aggregate.return_value = [
            {"_id": "id", "section_id": "s", "section_text": "t", "score": None}
        ]
        asyncio.run(retriever.retrieve("db", "same query", "TX", None, 1))

    assert embedder.calls == ["same query", "same query"]
    assert collection.aggregate.call_count == 2


def test_retrieve_by_queries_keeps_the_higher_score_for_a_chunk_id() -> None:
    first = [
        {
            "_id": "A",
            "section": "§ 1",
            "section_id": "same",
            "section_text": "low",
            "title": "Low",
            "jurisdiction": "TX",
            "score": 0.2,
        },
        {
            "_id": "B",
            "section_id": "b1",
            "section_text": "top",
            "title": "Top",
            "jurisdiction": "TX",
            "score": 0.95,
        },
    ]
    second = [
        {
            "_id": "A",
            "section": "§ 2",
            "section_id": "same",
            "section_text": "high",
            "title": "High",
            "jurisdiction": "TX",
            "score": 0.8,
        },
        {
            "_id": "B",
            "section_id": "b1",
            "section_text": "drop",
            "title": "Drop",
            "jurisdiction": "TX",
            "score": 0.4,
        },
        {
            "_id": "C",
            "section_id": "c1",
            "section_text": "tail",
            "title": "Tail",
            "jurisdiction": "TX",
            "score": 0.1,
        },
    ]
    retriever, collection, embedder, _, _ = _retriever([first, second])

    results = asyncio.run(
        retriever.retrieve_by_queries(
            database="privacy_db",
            jurisdiction="TX",
            queries=["know", "delete"],
            top_k_per_query=2,
        )
    )

    assert embedder.calls == ["know", "delete"]
    assert collection.aggregate.call_count == 2
    assert [(item.statute_id, item.chunk_id, item.score, item.title, item.section_id) for item in results] == [
        ("B", "b1", 0.95, "Top", "b1"),
        ("A", "same", 0.8, "High", "§ 2"),
        ("C", "c1", 0.1, "Tail", "c1"),
    ]
    limits = [
        call.args[0][0]["$vectorSearch"]["limit"]
        for call in collection.aggregate.call_args_list
    ]
    assert limits == [2, 2]


def test_retrieve_by_queries_equal_score_keeps_the_first_hit() -> None:
    retriever, _, _, _, _ = _retriever(
        [
            [
                {
                    "_id": "A",
                    "section_id": "s",
                    "section_text": "first",
                    "title": "First",
                    "score": 0.5,
                }
            ],
            [
                {
                    "_id": "A",
                    "section_id": "s",
                    "section_text": "second",
                    "title": "Second",
                    "score": 0.5,
                }
            ],
        ]
    )
    results = asyncio.run(
        retriever.retrieve_by_queries("db", "TX", ["one", "two"], 1)
    )
    assert len(results) == 1
    assert results[0].title == "First"
    assert results[0].chunk_text == "first"
    assert results[0].score == 0.5


def test_retrieve_by_queries_with_no_queries_does_not_search() -> None:
    retriever, collection, embedder, _, _ = _retriever([])
    results = asyncio.run(
        retriever.retrieve_by_queries("db", "TX", [], 3)
    )
    assert results == []
    assert embedder.calls == []
    collection.aggregate.assert_not_called()
