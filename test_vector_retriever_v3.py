"""Regression tests for v3 statute-to-policy vector retrieval."""
from __future__ import annotations

import asyncio
import sys
from typing import Any, Dict, Iterable, List
from unittest.mock import MagicMock

# Avoid loading sentence_transformers in unit tests.
sys.modules["sentence_transformers"] = MagicMock()

from cache import SimpleLRUCache
from compliance_config import load_config
from vector_retriever import VectorRetriever


class FakeCollection:
    def __init__(
        self,
        *,
        find_results: Iterable[Dict[str, Any]] | None = None,
        aggregate_results: List[Any] | None = None,
        find_one_results: Dict[Any, Dict[str, Any]] | None = None,
    ) -> None:
        self.find_results = list(find_results or [])
        self.aggregate_results = list(aggregate_results or [])
        self.find_one_results = find_one_results or {}
        self.find_calls: List[tuple[Dict[str, Any], Dict[str, Any] | None]] = []
        self.aggregate_calls: List[List[Dict[str, Any]]] = []
        self.find_one_calls: List[tuple[Dict[str, Any], Dict[str, Any] | None]] = []

    def find(
        self,
        query: Dict[str, Any],
        projection: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        self.find_calls.append((query, projection))
        return list(self.find_results)

    def aggregate(self, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self.aggregate_calls.append(pipeline)
        result = self.aggregate_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return list(result)

    def find_one(
        self,
        query: Dict[str, Any],
        projection: Dict[str, Any] | None = None,
    ) -> Dict[str, Any] | None:
        self.find_one_calls.append((query, projection))
        return self.find_one_results.get(query.get("_id"))


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


def test_retrieve_statute_policy_pairs_v3_fallback_and_parent_contexts() -> None:
    config = load_config()
    vec_field = config.embedding_vector_field
    statute_sub_text = config.statute_subchunk_text_field
    policy_sub_text = config.policy_subchunk_text_field
    policy_chunk_text = config.policy_chunk_text_field
    statute_chunk_text = config.statute_chunk_text_field

    statute_coll = FakeCollection(
        find_results=[
            {
                "_id": "stat-sub-1",
                vec_field: ["0.1", 0.2],
                statute_sub_text: "Businesses must honor opt-out requests.",
                "statute_reference": "CCPA 1798.120",
                config.statute_jurisdiction_field: "California",
                "parent_chunk_id": "stat-parent-1",
            },
            {
                "_id": "stat-sub-ignored",
                vec_field: "not-a-vector",
                statute_sub_text: "Invalid embedding row should not trigger search.",
                config.statute_jurisdiction_field: "California",
            },
        ]
    )
    policy_results = [
        {
            "_id": "policy-sub-1",
            policy_sub_text: "Users can opt out of sale or sharing.",
            "score": 0.91,
            "section_id": "privacy_choices",
            "parent_chunk_id": "policy-parent-1",
        },
        {
            "_id": "policy-sub-2",
            policy_sub_text: "General analytics disclosure.",
            "score": 0.42,
            "chunk_index": 7,
            "parent_chunk_id": "policy-parent-2",
        },
    ]
    policy_coll = FakeCollection(
        aggregate_results=[
            RuntimeError("filtered vector search unsupported"),
            policy_results,
        ]
    )
    policy_parent_coll = FakeCollection(
        find_results=[
            {"_id": "policy-parent-1", policy_chunk_text: "Privacy choices parent text."},
            {"_id": "policy-parent-2", "text": "Analytics parent fallback text."},
        ]
    )
    statute_parent_coll = FakeCollection(
        find_one_results={
            "stat-parent-1": {statute_chunk_text: "Full statute parent context."}
        }
    )
    database = FakeDatabase(
        {
            config.statute_sub_embeddings_collection: statute_coll,
            config.policy_sub_embeddings_collection: policy_coll,
            config.policy_embeddings_collection: policy_parent_coll,
            config.statute_embeddings_collection: statute_parent_coll,
        }
    )
    retriever = VectorRetriever(
        FakeMongoClient(database),
        MagicMock(),
        config,
        SimpleLRUCache(10, 60),
    )

    pairs = asyncio.run(
        retriever.retrieve_statute_policy_pairs_v3(
            database="privacy-compliance",
            policy_document_id="policy-123",
            applicable_jurisdictions=["CA"],
            statute_document_id="statute-doc-1",
            top_k=2,
            num_candidates=10,
            score_threshold=0.70,
        )
    )

    assert len(pairs) == 1
    assert pairs[0].statute_doc["_id"] == "stat-sub-1"
    assert pairs[0].above_threshold_count == 1
    assert pairs[0].statute_parent_context == "Full statute parent context."
    assert [match.text for match in pairs[0].policy_matches] == [
        "Users can opt out of sale or sharing.",
        "General analytics disclosure.",
    ]
    assert [match.parent_context for match in pairs[0].policy_matches] == [
        "Privacy choices parent text.",
        "Analytics parent fallback text.",
    ]
    assert pairs[0].policy_matches[1].section_id == "7"

    assert len(policy_coll.aggregate_calls) == 2
    filtered_pipeline, fallback_pipeline = policy_coll.aggregate_calls
    assert filtered_pipeline[0]["$vectorSearch"]["filter"] == {
        config.policy_document_id_field: "policy-123"
    }
    assert "filter" not in fallback_pipeline[0]["$vectorSearch"]
    assert fallback_pipeline[2] == {
        "$match": {config.policy_document_id_field: "policy-123"}
    }

    statute_query, _ = statute_coll.find_calls[0]
    assert statute_query["document_id"] == "statute-doc-1"
    assert "CA" in statute_query[config.statute_jurisdiction_field]["$in"]
    assert policy_parent_coll.find_calls[0][0] == {
        "_id": {"$in": ["policy-parent-1", "policy-parent-2"]}
    }
    assert len(statute_parent_coll.find_one_calls) == 1
