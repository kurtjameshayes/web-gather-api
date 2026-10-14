"""Regression tests for v3 statute/policy pair retrieval bounds.

These pin which statute rows are searched and which vector is sent. A wrong
row or a dropped embedding fallback changes gap-analysis matches.
"""
from __future__ import annotations

import asyncio
import sys
from typing import Any, Dict, List
from unittest.mock import MagicMock

sys.modules.setdefault("sentence_transformers", MagicMock())

from cache import SimpleLRUCache
from compliance_config import load_config
from compliance_utils import jurisdiction_filter_values
from vector_retriever import VECTOR_SEARCH_JURISDICTION, VectorRetriever
import vector_retriever


class FakeCollection:
    def __init__(self, find_results: List[Dict[str, Any]] | None = None) -> None:
        self.find_results = list(find_results or [])
        self.find_calls: List[tuple[Any, Any]] = []
        self.aggregate_calls: List[List[Dict[str, Any]]] = []
        self.find_one_calls: List[tuple[Any, Any]] = []

    def find(self, query: Dict[str, Any], projection: Dict[str, Any] | None = None):
        self.find_calls.append((query, projection))
        return list(self.find_results)

    def aggregate(self, pipeline: List[Dict[str, Any]]):
        self.aggregate_calls.append(pipeline)
        return []

    def find_one(self, query: Dict[str, Any], projection: Dict[str, Any] | None = None):
        self.find_one_calls.append((query, projection))
        return None


class FakeDatabase:
    def __init__(self, collections: Dict[str, FakeCollection]) -> None:
        self.collections = collections

    def __getitem__(self, name: str) -> FakeCollection:
        return self.collections[name]


class FakeMongoClient:
    def __init__(self, database: FakeDatabase) -> None:
        self.database = database

    def __getitem__(self, name: str) -> FakeDatabase:
        return self.database


def _retriever(statute_docs: List[Dict[str, Any]]) -> tuple[VectorRetriever, FakeCollection]:
    config = load_config()
    statute_coll = FakeCollection(statute_docs)
    policy_coll = FakeCollection()
    database = FakeDatabase(
        {
            config.statute_sub_embeddings_collection: statute_coll,
            config.policy_sub_embeddings_collection: policy_coll,
            config.policy_embeddings_collection: FakeCollection(),
            config.statute_embeddings_collection: FakeCollection(),
        }
    )
    retriever = VectorRetriever(
        FakeMongoClient(database),
        MagicMock(),
        config,
        SimpleLRUCache(4, 60),
    )
    return retriever, policy_coll


def _statute(config, doc_id: str, vector: Any) -> Dict[str, Any]:
    return {
        "_id": doc_id,
        config.embedding_vector_field: vector,
        config.statute_jurisdiction_field: "CA",
    }


def _query_vectors(policy_coll: FakeCollection) -> List[List[float]]:
    return [
        call[0]["$vectorSearch"]["queryVector"]
        for call in policy_coll.aggregate_calls
    ]


def test_num_rows_slices_statutes_before_vector_search() -> None:
    """num_rows is an in-memory cap. Later statute vectors must not be searched."""
    config = load_config()
    docs = [
        _statute(config, "first", [0.1, 0.2]),
        _statute(config, "second", [0.3, 0.4]),
        _statute(config, "third", [0.5, 0.6]),
    ]
    retriever, policy_coll = _retriever(docs)

    pairs = asyncio.run(
        retriever.retrieve_statute_policy_pairs_v3(
            database="privacy-compliance",
            policy_document_id="policy-1",
            applicable_jurisdictions=["CA"],
            num_rows=1,
        )
    )

    assert [pair.statute_doc["_id"] for pair in pairs] == ["first"]
    assert _query_vectors(policy_coll) == [[0.1, 0.2]]

    empty, empty_policy = _retriever(docs)
    none = asyncio.run(
        empty.retrieve_statute_policy_pairs_v3(
            database="privacy-compliance",
            policy_document_id="policy-1",
            applicable_jurisdictions=["CA"],
            num_rows=0,
        )
    )
    assert none == []
    assert empty_policy.aggregate_calls == []


def test_missing_vector_field_falls_back_to_embedding_and_skips_bad_rows() -> None:
    """Configured vector wins; an empty vector uses `embedding`; bad rows are skipped.

    The shipped config stores vectors in `embedding`, so this test points the
    configured field at `vector` to make the literal `embedding` fallback visible.
    A present but non-numeric vector does not fall through to `embedding`.
    """
    docs = [
        {"_id": "from-embedding", "embedding": [1, 2]},
        {"_id": "empty-vector", "vector": [], "embedding": ["3", 4]},
        {"_id": "bad", "vector": ["nope"], "embedding": [9, 9]},
        {"_id": "zero-vector", "vector": [0, 1], "embedding": [8, 8]},
    ]
    retriever, policy_coll = _retriever(docs)
    retriever._config.embedding_vector_field = "vector"

    pairs = asyncio.run(
        retriever.retrieve_statute_policy_pairs_v3(
            database="privacy-compliance",
            policy_document_id="policy-1",
            applicable_jurisdictions=["CA"],
        )
    )

    assert [pair.statute_doc["_id"] for pair in pairs] == [
        "from-embedding",
        "empty-vector",
        "zero-vector",
    ]
    assert _query_vectors(policy_coll) == [[1.0, 2.0], [3.0, 4.0], [0.0, 1.0]]


def test_search_window_and_single_jurisdiction_are_bound_per_statute() -> None:
    """numCandidates stays >= limit, and each search lambda captures its own pipeline."""
    config = load_config()
    docs = [_statute(config, "tx-1", [0.5, 0.25])]
    retriever, policy_coll = _retriever(docs)
    captured: List[Any] = []
    real_run = vector_retriever.run_in_thread

    async def _spy(func, *args, **kwargs):
        captured.append(func)
        return await real_run(func, *args, **kwargs)

    vector_retriever.run_in_thread = _spy
    try:
        asyncio.run(
            retriever.retrieve_statute_policy_pairs_v3(
                database="privacy-compliance",
                policy_document_id="policy-1",
                applicable_jurisdictions=["Texas"],
                top_k=1,
                num_candidates=10,
            )
        )
    finally:
        vector_retriever.run_in_thread = real_run

    statute_coll = retriever._mongo_client["privacy-compliance"][
        config.statute_sub_embeddings_collection
    ]
    jurisdiction_filter = statute_coll.find_calls[0][0][config.statute_jurisdiction_field]
    assert jurisdiction_filter == "TEXAS"

    stage = policy_coll.aggregate_calls[0][0]["$vectorSearch"]
    assert stage["limit"] == 500
    assert stage["numCandidates"] == 500
    assert stage["limit"] <= stage["numCandidates"]
    assert stage["queryVector"] == [0.5, 0.25]

    bound = [fn for fn in captured if fn.__defaults__]
    assert len(bound) == 1
    assert bound[0].__defaults__[0][0]["$vectorSearch"]["queryVector"] == [0.5, 0.25]
    assert bound[0].__defaults__[0][0]["$vectorSearch"]["filter"] == {
        config.policy_document_id_field: "policy-1"
    }


def test_empty_jurisdictions_use_raw_vector_search_label() -> None:
    """Omitted jurisdictions are not normalized before alias expansion.

    `VECTOR_SEARCH_JURISDICTION` is passed straight to `jurisdiction_filter_values`.
    The default label `California` is not a map key, so the Mongo filter is the
    string `California` rather than the CA alias set.
    """
    config = load_config()
    retriever, _policy = _retriever([_statute(config, "ca-1", [0.2, 0.2])])
    asyncio.run(
        retriever.retrieve_statute_policy_pairs_v3(
            database="privacy-compliance",
            policy_document_id="policy-1",
            applicable_jurisdictions=[],
        )
    )
    statute_coll = retriever._mongo_client["privacy-compliance"][
        config.statute_sub_embeddings_collection
    ]
    jurisdiction_filter = statute_coll.find_calls[0][0][config.statute_jurisdiction_field]
    expected_values = jurisdiction_filter_values(VECTOR_SEARCH_JURISDICTION)
    expected = expected_values[0] if len(expected_values) == 1 else {"$in": expected_values}
    assert jurisdiction_filter == expected


def test_score_boundary_text_fallback_and_missing_parent() -> None:
    """Threshold is inclusive. Whitespace subchunk text blocks later fallbacks."""
    config = load_config()
    statute_coll = FakeCollection(
        [
            {
                "_id": "stat-1",
                config.embedding_vector_field: [0.4, 0.4],
                "parent_chunk_id": "missing-parent",
            }
        ]
    )
    policy_coll = FakeCollection()
    policy_coll.aggregate = lambda pipeline: (  # type: ignore[method-assign]
        policy_coll.aggregate_calls.append(pipeline) or [
            {
                config.policy_subchunk_text_field: "   ",
                "text": "should not be used",
                "score": 0.70,
                "section_id": "",
                "chunk_index": 0,
            },
            {
                "text": "  notice  ",
                "chunk_index": 3,
            },
            {
                config.policy_chunk_text_field: "  retention  ",
            },
        ]
    )
    parents = FakeCollection()
    database = FakeDatabase(
        {
            config.statute_sub_embeddings_collection: statute_coll,
            config.policy_sub_embeddings_collection: policy_coll,
            config.policy_embeddings_collection: parents,
            config.statute_embeddings_collection: FakeCollection(),
        }
    )
    retriever = VectorRetriever(
        FakeMongoClient(database),
        MagicMock(),
        config,
        SimpleLRUCache(4, 60),
    )

    pairs = asyncio.run(
        retriever.retrieve_statute_policy_pairs_v3(
            database="privacy-compliance",
            policy_document_id="policy-1",
            applicable_jurisdictions=["CA"],
            score_threshold=0.70,
        )
    )

    assert len(pairs) == 1
    assert pairs[0].above_threshold_count == 1
    assert pairs[0].statute_parent_context is None
    matches = pairs[0].policy_matches
    assert [match.text for match in matches] == ["", "notice", "retention"]
    assert [match.score for match in matches] == [0.70, 0.0, 0.0]
    assert matches[0].section_id is None
    assert matches[1].section_id == "3"
    assert matches[0].parent_context is None
    assert parents.find_calls == []
    statute_parents = database.collections[config.statute_embeddings_collection]
    assert statute_parents.find_one_calls[0][0] == {"_id": "missing-parent"}
