"""Regression tests for v1/v2 statute-to-policy pair retrieval.

These methods drive gap analysis v1 (subchunks) and v2 (chunks). They previously
had no direct coverage: route/service tests mock the retriever, and open PRs
cover only statutes/embeddings retrieve plus v3 pair hydration.
"""
from __future__ import annotations

import asyncio
import sys
from typing import Any, Dict, Iterable, List
from unittest.mock import MagicMock

sys.modules["sentence_transformers"] = MagicMock()

from cache import SimpleLRUCache
from compliance_config import load_config
from compliance_utils import jurisdiction_filter_values, normalize_jurisdiction
from vector_retriever import VECTOR_SEARCH_JURISDICTION, VectorRetriever


class FakeCollection:
    def __init__(
        self,
        *,
        find_results: Iterable[Dict[str, Any]] | None = None,
        aggregate_results: List[Any] | None = None,
    ) -> None:
        self.find_results = list(find_results or [])
        self.aggregate_results = list(aggregate_results or [])
        self.find_calls: List[tuple[Dict[str, Any], Dict[str, Any] | None]] = []
        self.aggregate_calls: List[List[Dict[str, Any]]] = []

    def find(
        self,
        query: Dict[str, Any],
        projection: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        self.find_calls.append((query, projection))
        return list(self.find_results)

    def aggregate(self, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self.aggregate_calls.append(pipeline)
        if not self.aggregate_results:
            return []
        result = self.aggregate_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return list(result)


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


def _retriever(database: FakeDatabase) -> VectorRetriever:
    return VectorRetriever(
        FakeMongoClient(database),
        MagicMock(),
        load_config(),
        SimpleLRUCache(10, 60),
    )


def test_retrieve_policy_subchunks_empty_statutes_short_circuits() -> None:
    config = load_config()
    statute_coll = FakeCollection(find_results=[])
    policy_coll = FakeCollection()
    retriever = _retriever(
        FakeDatabase(
            {
                config.statute_sub_embeddings_collection: statute_coll,
                config.policy_sub_embeddings_collection: policy_coll,
            }
        )
    )

    result = asyncio.run(
        retriever.retrieve_policy_subchunks_for_statute_subchunks(
            database="privacy-compliance",
            policy_document_id="policy-1",
            applicable_jurisdictions=["CA"],
        )
    )

    assert result.pairs == []
    assert result.statute_subchunks_considered == 0
    assert policy_coll.aggregate_calls == []
    statute_query, _ = statute_coll.find_calls[0]
    assert "CA" in statute_query[config.statute_jurisdiction_field]["$in"]
    assert "California" in statute_query[config.statute_jurisdiction_field]["$in"]


def test_retrieve_policy_subchunks_default_jurisdiction_and_skips_bad_vectors() -> None:
    config = load_config()
    vec_field = config.embedding_vector_field
    statute_coll = FakeCollection(
        find_results=[
            {"_id": "missing-vec", config.statute_jurisdiction_field: "California"},
            {
                "_id": "not-a-list",
                vec_field: "0.1,0.2",
                config.statute_jurisdiction_field: "California",
            },
            {
                "_id": "non-numeric",
                vec_field: ["x", 0.2],
                config.statute_jurisdiction_field: "California",
            },
        ]
    )
    policy_coll = FakeCollection()
    retriever = _retriever(
        FakeDatabase(
            {
                config.statute_sub_embeddings_collection: statute_coll,
                config.policy_sub_embeddings_collection: policy_coll,
            }
        )
    )

    result = asyncio.run(
        retriever.retrieve_policy_subchunks_for_statute_subchunks(
            database="privacy-compliance",
            policy_document_id="policy-1",
            applicable_jurisdictions=[],
        )
    )

    assert result.pairs == []
    assert result.statute_subchunks_considered == 3
    assert policy_coll.aggregate_calls == []
    statute_query, _ = statute_coll.find_calls[0]
    default_values = jurisdiction_filter_values(
        normalize_jurisdiction(VECTOR_SEARCH_JURISDICTION)
    )
    assert statute_query[config.statute_jurisdiction_field]["$in"] == default_values
    assert "document_id" not in statute_query


def test_retrieve_policy_subchunks_filter_fallback_and_payload_shape() -> None:
    config = load_config()
    vec_field = config.embedding_vector_field
    statute_coll = FakeCollection(
        find_results=[
            {
                "_id": "stat-1",
                vec_field: ["0.1", 0.2],
                config.statute_jurisdiction_field: "Texas",
                "document_id": "statute-doc-1",
            },
            {
                "_id": "stat-no-match",
                vec_field: [0.3, 0.4],
                config.statute_jurisdiction_field: "Texas",
            },
        ]
    )
    policy_coll = FakeCollection(
        aggregate_results=[
            RuntimeError("filtered $vectorSearch unsupported"),
            [
                {
                    "_id": 99,
                    vec_field: [9.0, 8.0],
                    "score": 0.88,
                    "subchunk_text": "Users may opt out of sale.",
                    "document_id": "policy-1",
                }
            ],
            [],
        ]
    )
    retriever = _retriever(
        FakeDatabase(
            {
                config.statute_sub_embeddings_collection: statute_coll,
                config.policy_sub_embeddings_collection: policy_coll,
            }
        )
    )

    result = asyncio.run(
        retriever.retrieve_policy_subchunks_for_statute_subchunks(
            database="privacy-compliance",
            policy_document_id="policy-1",
            applicable_jurisdictions=["Texas"],
            statute_document_id="statute-doc-1",
            top_k_per_statute=3,
        )
    )

    assert result.statute_subchunks_considered == 2
    assert len(result.pairs) == 1
    pair = result.pairs[0]
    assert pair.statute_doc["_id"] == "stat-1"
    assert pair.score == 0.88
    assert pair.policy_doc["_id"] == "99"
    assert vec_field not in pair.policy_doc
    assert pair.policy_doc["subchunk_text"] == "Users may opt out of sale."

    statute_query, _ = statute_coll.find_calls[0]
    assert statute_query["document_id"] == "statute-doc-1"
    assert statute_query[config.statute_jurisdiction_field] == "TEXAS"

    assert len(policy_coll.aggregate_calls) == 3
    filtered, fallback, second_filtered = policy_coll.aggregate_calls
    policy_filter = {config.policy_document_id_field: "policy-1"}
    assert filtered[0]["$vectorSearch"]["filter"] == policy_filter
    assert "filter" not in fallback[0]["$vectorSearch"]
    assert fallback[2] == {"$match": policy_filter}
    assert fallback[3] == {"$limit": 3}
    assert filtered[0]["$vectorSearch"]["limit"] == 1500
    assert filtered[0]["$vectorSearch"]["numCandidates"] == 1500
    assert filtered[0]["$vectorSearch"]["queryVector"] == [0.1, 0.2]
    assert filtered[0]["$vectorSearch"]["index"] == config.vector_index_name
    assert filtered[0]["$vectorSearch"]["path"] == vec_field
    assert second_filtered[0]["$vectorSearch"]["queryVector"] == [0.3, 0.4]


def test_retrieve_policy_chunks_uses_chunk_collections_and_default_limits() -> None:
    config = load_config()
    vec_field = config.embedding_vector_field
    statute_coll = FakeCollection(
        find_results=[
            {
                "_id": "chunk-1",
                vec_field: [0.5, 0.6],
                config.statute_jurisdiction_field: "CA",
            }
        ]
    )
    policy_coll = FakeCollection(
        aggregate_results=[
            [
                {
                    "_id": "policy-chunk-1",
                    vec_field: [1.0],
                    "score": "0.71",
                    "chunk_text": "We honor deletion requests.",
                }
            ]
        ]
    )
    retriever = _retriever(
        FakeDatabase(
            {
                config.statute_embeddings_collection: statute_coll,
                config.policy_embeddings_collection: policy_coll,
                config.statute_sub_embeddings_collection: FakeCollection(),
                config.policy_sub_embeddings_collection: FakeCollection(),
            }
        )
    )

    result = asyncio.run(
        retriever.retrieve_policy_chunks_for_statute_chunks(
            database="privacy-compliance",
            policy_document_id="policy-9",
            applicable_jurisdictions=["CA", "VA"],
        )
    )

    assert result.statute_chunks_considered == 1
    assert len(result.pairs) == 1
    assert result.pairs[0].score == 0.71
    assert result.pairs[0].policy_doc["_id"] == "policy-chunk-1"
    assert vec_field not in result.pairs[0].policy_doc

    statute_query, _ = statute_coll.find_calls[0]
    jur_filter = statute_query[config.statute_jurisdiction_field]["$in"]
    assert "CA" in jur_filter
    assert "VA" in jur_filter or "Virginia" in jur_filter or "va" in jur_filter

    vs = policy_coll.aggregate_calls[0][0]["$vectorSearch"]
    assert vs["filter"] == {config.policy_document_id_field: "policy-9"}
    assert vs["limit"] == 500
    assert vs["numCandidates"] == 1000
    assert vs["queryVector"] == [0.5, 0.6]
    assert statute_coll.find_calls, "v2 must read statute_embeddings, not subchunk collections"
    assert policy_coll.aggregate_calls, "v2 must search policy_embeddings, not subchunk collections"
