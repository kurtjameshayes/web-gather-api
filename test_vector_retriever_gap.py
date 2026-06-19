"""Regression tests for statute-policy vector retrieval fallback paths."""
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
    def __init__(
        self,
        *,
        find_results: list[dict[str, Any]] | None = None,
        aggregate_effects: list[Any] | None = None,
        find_one_result: dict[str, Any] | None = None,
    ) -> None:
        self.find_results = find_results or []
        self.aggregate_effects = aggregate_effects or []
        self.find_one_result = find_one_result
        self.find_calls: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
        self.aggregate_calls: list[list[dict[str, Any]]] = []
        self.find_one_calls: list[tuple[dict[str, Any], dict[str, Any] | None]] = []

    def find(
        self,
        query: dict[str, Any],
        projection: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        self.find_calls.append((query, projection))
        return list(self.find_results)

    def aggregate(self, pipeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.aggregate_calls.append(pipeline)
        effect = self.aggregate_effects.pop(0)
        if isinstance(effect, Exception):
            raise effect
        return list(effect)

    def find_one(
        self,
        query: dict[str, Any],
        projection: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        self.find_one_calls.append((query, projection))
        return self.find_one_result


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


def _retriever(mongo: FakeMongo) -> tuple[VectorRetriever, Any]:
    config = load_config()
    return VectorRetriever(
        mongo,
        embedder=MagicMock(),
        config=config,
        cache=SimpleLRUCache(10, 60),
    ), config


def test_policy_chunk_retrieval_falls_back_to_post_match_when_vector_filter_fails() -> None:
    retriever, config = _retriever(
        FakeMongo(
            {
                "privacy_db": FakeDatabase(
                    {
                        "statute_embeddings": FakeCollection(
                            find_results=[
                                {
                                    "_id": "stat-1",
                                    config.embedding_vector_field: [0.1, "0.2"],
                                    config.statute_jurisdiction_field: "California",
                                }
                            ],
                        ),
                        "policy_embeddings": FakeCollection(
                            aggregate_effects=[
                                RuntimeError("filter not indexed"),
                                [
                                    {
                                        "_id": "policy-chunk-1",
                                        config.embedding_vector_field: [0.9, 0.8],
                                        config.policy_document_id_field: "policy-1",
                                        "chunk_text": "Deletion request policy text.",
                                        "score": 0.87,
                                    }
                                ],
                            ],
                        ),
                    }
                )
            }
        )
    )
    policy_coll = retriever._mongo_client["privacy_db"]["policy_embeddings"]

    result = asyncio.run(
        retriever.retrieve_policy_chunks_for_statute_chunks(
            database="privacy_db",
            policy_document_id="policy-1",
            applicable_jurisdictions=["CA"],
        )
    )

    assert result.statute_chunks_considered == 1
    assert len(result.pairs) == 1
    assert result.pairs[0].score == 0.87
    assert config.embedding_vector_field not in result.pairs[0].policy_doc

    filtered_pipeline, fallback_pipeline = policy_coll.aggregate_calls
    policy_filter = {config.policy_document_id_field: "policy-1"}
    assert filtered_pipeline[0]["$vectorSearch"]["filter"] == policy_filter
    assert "filter" not in fallback_pipeline[0]["$vectorSearch"]
    assert {"$match": policy_filter} in fallback_pipeline


def test_v3_pair_retrieval_falls_back_and_attaches_parent_context() -> None:
    config = load_config()
    statute_sub_coll = FakeCollection(
        find_results=[
            {
                "_id": "stat-sub-1",
                config.embedding_vector_field: [0.3, 0.4],
                config.statute_jurisdiction_field: "California",
                config.statute_subchunk_text_field: "Consumers may request deletion.",
                "statute_reference": "CCPA 1798.105",
                "parent_chunk_id": "stat-parent-1",
            }
        ]
    )
    policy_sub_coll = FakeCollection(
        aggregate_effects=[
            RuntimeError("filter not indexed"),
            [
                {
                    "_id": "policy-sub-1",
                    config.policy_subchunk_text_field: "You can ask us to delete your data.",
                    "section_id": "deletion",
                    "parent_chunk_id": "policy-parent-1",
                    "score": 0.82,
                }
            ],
        ]
    )
    policy_parent_coll = FakeCollection(
        find_results=[
            {
                "_id": "policy-parent-1",
                config.policy_chunk_text_field: "Full deletion rights section.",
            }
        ]
    )
    statute_parent_coll = FakeCollection(
        find_one_result={
            "_id": "stat-parent-1",
            config.statute_chunk_text_field: "Full statutory deletion requirement.",
        }
    )
    mongo = FakeMongo(
        {
            "privacy_db": FakeDatabase(
                {
                    config.statute_sub_embeddings_collection: statute_sub_coll,
                    config.policy_sub_embeddings_collection: policy_sub_coll,
                    config.policy_embeddings_collection: policy_parent_coll,
                    config.statute_embeddings_collection: statute_parent_coll,
                }
            )
        }
    )
    retriever, _ = _retriever(mongo)

    pairs = asyncio.run(
        retriever.retrieve_statute_policy_pairs_v3(
            database="privacy_db",
            policy_document_id="policy-1",
            applicable_jurisdictions=["CA"],
            top_k=1,
            score_threshold=0.7,
        )
    )

    assert len(pairs) == 1
    assert pairs[0].above_threshold_count == 1
    assert pairs[0].statute_parent_context == "Full statutory deletion requirement."
    assert pairs[0].policy_matches[0].text == "You can ask us to delete your data."
    assert pairs[0].policy_matches[0].parent_context == "Full deletion rights section."

    filtered_pipeline, fallback_pipeline = policy_sub_coll.aggregate_calls
    policy_filter = {config.policy_document_id_field: "policy-1"}
    assert filtered_pipeline[0]["$vectorSearch"]["filter"] == policy_filter
    assert "filter" not in fallback_pipeline[0]["$vectorSearch"]
    assert {"$match": policy_filter} in fallback_pipeline
