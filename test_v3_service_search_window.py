"""Gap analysis v3 service search window.

The live service does not call VectorRetriever.retrieve_statute_policy_pairs_v3.
It builds its own Atlas query and does not apply gap_analysis_v3_score_threshold
or gap_analysis_v3_num_candidates.
"""
from __future__ import annotations

import asyncio
from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock

from compliance_config import load_config
from compliance_suite_schemas import GapAnalysisRequest
from gap_analysis_service_v3 import GapAnalysisServiceV3


def _run_with_top_k(top_k: int):
    cfg = replace(
        load_config(),
        statute_database="",
        compliance_database="privacy-compliance",
        gap_analysis_v3_top_k=top_k,
        gap_analysis_v3_num_candidates=7,
        gap_analysis_v3_score_threshold=0.99,
        vector_index_name="shared-index",
        policy_sub_embeddings_vector_index="unused-v3-index",
    )
    collections: dict[str, MagicMock] = {}

    def _coll(name: str) -> MagicMock:
        if name not in collections:
            collections[name] = MagicMock(name=name)
        return collections[name]

    db = MagicMock()
    db.__getitem__.side_effect = _coll
    mongo = MagicMock()
    mongo.__getitem__.return_value = db

    policies = _coll(cfg.policies_collection)
    policies.find_one.return_value = {
        "document_id": "pol-1",
        "text": "FULL POLICY TEXT",
        "company_name": "Acme",
    }
    policy_sub = _coll(cfg.policy_sub_embeddings_collection)
    policy_sub.count_documents.return_value = 1
    captured: list = []

    def _aggregate(pipeline):
        captured.append(pipeline)
        return [
            {
                "score": 0.05,
                cfg.policy_subchunk_text_field: "LOW SCORE POLICY PHRASE",
            }
        ]

    policy_sub.aggregate.side_effect = _aggregate
    statute_sub = _coll(cfg.statute_sub_embeddings_collection)
    statute_sub.find.return_value = [
        {
            "_id": "stat-1",
            cfg.embedding_vector_field: [0.2, 0.4],
            cfg.statute_subchunk_text_field: "Consumers may request deletion.",
            "jurisdiction": "CA",
            "statute_reference": "CCPA 1798.105",
        }
    ]

    llm = MagicMock()
    llm.gap_check_v3 = AsyncMock(
        return_value={
            "status": "missing",
            "policy_quote": None,
            "requirement_summary": "Must delete",
            "conflict_description": "kept description",
            "confidence": "low",
            "_analysis_failed": False,
        }
    )

    class _Allow:
        async def allow(self) -> bool:
            return True

    service = GapAnalysisServiceV3(
        mongo_client=mongo,
        config=cfg,
        retriever=MagicMock(),
        llm_client=llm,
        storage=MagicMock(),
        rate_limiter=_Allow(),
    )
    response = asyncio.run(
        service.run(
            GapAnalysisRequest(
                policy_document_id="pol-1",
                applicable_jurisdictions=["CA"],
                save_results=False,
            )
        )
    )
    assert response.summary.missing == 1
    assert llm.gap_check_v3.await_count == 1
    kwargs = llm.gap_check_v3.await_args.kwargs
    assert kwargs["policy_chunk_text"] == "LOW SCORE POLICY PHRASE"
    assert "Consumers may request deletion." in kwargs["statute_chunk_text"]
    assert captured, "expected a vector search"
    return captured, kwargs, cfg


def test_v3_service_sends_below_threshold_hits_in_a_wide_window() -> None:
    """A score under the configured threshold still reaches the model.

    The candidate window is max(500, top_k * 500), not gap_analysis_v3_num_candidates.
    The index is vector_index_name, not policy_sub_embeddings_vector_index.
    """
    captured, _kwargs, cfg = _run_with_top_k(2)
    stage = captured[0][0]["$vectorSearch"]
    assert stage["index"] == "shared-index"
    assert stage["path"] == cfg.embedding_vector_field
    assert stage["queryVector"] == [0.2, 0.4]
    assert stage["numCandidates"] == 1000
    assert stage["limit"] == 1000
    assert stage["filter"] == {"document_id": "pol-1"}
    assert captured[0][2] == {"$limit": 2}
    assert "unused-v3-index" not in str(captured[0])


def test_v3_service_floors_the_search_window_at_500() -> None:
    """top_k 1 still searches 500 candidates, not the configured 7."""
    captured, _kwargs, _cfg = _run_with_top_k(1)
    stage = captured[0][0]["$vectorSearch"]
    assert stage["numCandidates"] == 500
    assert stage["limit"] == 500
    assert captured[0][2] == {"$limit": 1}
