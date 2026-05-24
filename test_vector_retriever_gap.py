"""Regression tests for statute-to-policy vector retrieval used by gap analysis."""
from __future__ import annotations

import asyncio
import sys
from unittest.mock import MagicMock

sys.modules["sentence_transformers"] = MagicMock()

from cache import SimpleLRUCache
from compliance_config import load_config
from vector_retriever import VectorRetriever


async def _sync_run(func, *args, **kwargs):
    return func(*args, **kwargs)


def _build_retriever(statute_docs: list[dict], policy_results) -> tuple[VectorRetriever, MagicMock, MagicMock]:
    config = load_config()
    statute_coll = MagicMock(name="statute_sub_embeddings")
    statute_coll.find.return_value = statute_docs
    policy_coll = MagicMock(name="policy_sub_embeddings")
    policy_coll.aggregate.side_effect = policy_results

    collections = {
        config.statute_sub_embeddings_collection: statute_coll,
        config.policy_sub_embeddings_collection: policy_coll,
    }
    database = MagicMock(name="privacy-compliance-db")
    database.__getitem__.side_effect = lambda name: collections[name]
    mongo = MagicMock(name="mongo")
    mongo.__getitem__.return_value = database

    retriever = VectorRetriever(
        mongo_client=mongo,
        embedder=MagicMock(),
        config=config,
        cache=SimpleLRUCache(10, 60),
    )
    return retriever, statute_coll, policy_coll


def test_policy_subchunk_retrieval_falls_back_when_filtered_vector_search_fails(monkeypatch) -> None:
    """Unsupported vector filter indexes should retry with post-search matching."""
    monkeypatch.setattr("vector_retriever.run_in_thread", _sync_run)
    retriever, statute_coll, policy_coll = _build_retriever(
        statute_docs=[
            {
                "_id": "stat-sub-1",
                "document_id": "ccpa",
                "jurisdiction": "CA",
                "vector": [0.1, "0.2"],
                "subchunk_text": "Businesses must disclose deletion rights.",
            }
        ],
        policy_results=[
            RuntimeError("filter is not indexed"),
            [
                {
                    "_id": "policy-sub-1",
                    "document_id": "policy-1",
                    "chunk_text": "We provide deletion rights.",
                    "vector": [9.9, 8.8],
                    "score": 0.93,
                }
            ],
        ],
    )

    result = asyncio.run(
        retriever.retrieve_policy_subchunks_for_statute_subchunks(
            database="privacy-compliance",
            policy_document_id="policy-1",
            applicable_jurisdictions=["CA"],
            statute_document_id="ccpa",
            top_k_per_statute=1,
        )
    )

    statute_coll.find.assert_called_once()
    assert statute_coll.find.call_args[0][0]["document_id"] == "ccpa"
    assert policy_coll.aggregate.call_count == 2

    filtered_pipeline = policy_coll.aggregate.call_args_list[0][0][0]
    fallback_pipeline = policy_coll.aggregate.call_args_list[1][0][0]
    assert filtered_pipeline[0]["$vectorSearch"]["filter"] == {"document_id": "policy-1"}
    assert "filter" not in fallback_pipeline[0]["$vectorSearch"]
    assert fallback_pipeline[2] == {"$match": {"document_id": "policy-1"}}

    assert result.statute_subchunks_considered == 1
    assert len(result.pairs) == 1
    pair = result.pairs[0]
    assert pair.score == 0.93
    assert pair.policy_doc["_id"] == "policy-sub-1"
    assert "vector" not in pair.policy_doc


def test_policy_subchunk_retrieval_skips_invalid_statute_vectors(monkeypatch) -> None:
    """Bad stored embeddings should not trigger policy searches or crash analysis."""
    monkeypatch.setattr("vector_retriever.run_in_thread", _sync_run)
    retriever, _, policy_coll = _build_retriever(
        statute_docs=[
            {"_id": "missing-vector", "jurisdiction": "CA"},
            {"_id": "string-vector", "jurisdiction": "CA", "vector": "0.1,0.2"},
            {"_id": "bad-value", "jurisdiction": "CA", "vector": [0.1, object()]},
        ],
        policy_results=[],
    )

    result = asyncio.run(
        retriever.retrieve_policy_subchunks_for_statute_subchunks(
            database="privacy-compliance",
            policy_document_id="policy-1",
            applicable_jurisdictions=["CA"],
        )
    )

    assert result.statute_subchunks_considered == 3
    assert result.pairs == []
    policy_coll.aggregate.assert_not_called()
