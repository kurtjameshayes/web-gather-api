"""Service-level regression tests for V3 gap analysis orchestration."""
from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Lightweight stubs for optional heavy imports used transitively.
sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("anthropic", MagicMock())

from compliance_config import load_config
from compliance_suite_schemas import GapAnalysisRequest
from gap_analysis_service_v3 import GapAnalysisServiceV3, GapAnalysisServiceV3Error


class FakeDb:
    def __init__(self, collections: dict[str, MagicMock]) -> None:
        self._collections = collections

    def __getitem__(self, name: str) -> MagicMock:
        return self._collections[name]


class FakeMongo:
    def __init__(self, dbs: dict[str, FakeDb]) -> None:
        self._dbs = dbs

    def __getitem__(self, name: str) -> FakeDb:
        return self._dbs[name]


def _statute_doc(
    *,
    doc_id: str = "stat-1",
    statute_reference: str = "CCPA §1798.100",
    sub_chunk_text: str = "Businesses must disclose consumer access rights.",
    embedding: list[float] | None = None,
    jurisdiction: str = "CA",
    parent_chunk_id: str | None = "parent-stat-1",
) -> dict:
    return {
        "_id": doc_id,
        "document_id": "ccpa",
        "statute_reference": statute_reference,
        "sub_chunk_text": sub_chunk_text,
        "chunk_text": sub_chunk_text,
        "embedding": embedding if embedding is not None else [0.1, 0.2, 0.3],
        "jurisdiction": jurisdiction,
        "parent_chunk_id": parent_chunk_id,
    }


def _build_service(
    *,
    policy_doc: dict | None = None,
    statute_docs: list[dict] | None = None,
    policy_results: list[dict] | None = None,
    llm_result: dict | None = None,
    indexed_count: int = 1,
    aggregate_side_effect=None,
) -> tuple[GapAnalysisServiceV3, MagicMock, AsyncMock, MagicMock, MagicMock, MagicMock]:
    config = load_config()
    config.default_jurisdictions = ["CA"]

    policies = MagicMock()
    policies.find_one.return_value = policy_doc or {
        "document_id": "pol-1",
        "company_name": "Example Co",
        "text": "Consumers may request access to personal information through account settings.",
    }

    policy_sub = MagicMock()
    policy_sub.count_documents.return_value = indexed_count
    if aggregate_side_effect is not None:
        policy_sub.aggregate.side_effect = aggregate_side_effect
    else:
        policy_sub.aggregate.return_value = policy_results or [
            {
                "score": 0.91,
                "sub_chunk_text": "Consumers may request access to personal information through account settings.",
                "parent_chunk_id": "parent-pol-1",
            }
        ]

    statute_sub = MagicMock()
    statute_sub.find.return_value = statute_docs if statute_docs is not None else [_statute_doc()]

    statute_parent = MagicMock()
    statute_parent.find_one.return_value = {
        "_id": "parent-stat-1",
        "chunk_text": "Parent statute context about access rights.",
    }

    policy_parent = MagicMock()
    policy_parent.find.return_value = [
        {"_id": "parent-pol-1", "chunk_text": "Parent policy section on consumer access."}
    ]

    results = MagicMock()
    run_log = MagicMock()

    db_name = (config.statute_database or "").strip() or config.compliance_database
    collections = {
        config.policies_collection: policies,
        config.policy_sub_embeddings_collection: policy_sub,
        config.statute_sub_embeddings_collection: statute_sub,
        config.statute_embeddings_collection: statute_parent,
        config.policy_embeddings_collection: policy_parent,
        config.compliance_results_collection: results,
        config.compliance_run_log_collection: run_log,
    }
    db = FakeDb(collections)
    mongo = FakeMongo({db_name: db, config.compliance_database: db})

    llm = MagicMock()
    llm.gap_check_v3 = AsyncMock(
        return_value=llm_result
        or {
            "status": "addressed",
            "policy_quote": "Consumers may request access to personal information through account settings.",
            "confidence": "high",
            "requirement_summary": "Disclose access rights",
            "statute_quote": "Businesses must disclose consumer access rights.",
        }
    )

    rate_limiter = MagicMock()
    rate_limiter.allow = AsyncMock(return_value=True)

    service = GapAnalysisServiceV3(
        mongo_client=mongo,
        config=config,
        retriever=MagicMock(),
        llm_client=llm,
        storage=MagicMock(),
        rate_limiter=rate_limiter,
    )
    return service, rate_limiter, llm.gap_check_v3, results, run_log, policy_sub


def test_run_requires_policy_id() -> None:
    service, _, _, _, _, _ = _build_service()
    with pytest.raises(GapAnalysisServiceV3Error) as exc:
        asyncio.run(service.run(GapAnalysisRequest.model_construct()))
    assert exc.value.status_code == 400
    assert "policy_document_id" in str(exc.value)


def test_run_rate_limit_short_circuits() -> None:
    service, rate_limiter, gap_check, results, _, policy_sub = _build_service()
    rate_limiter.allow = AsyncMock(return_value=False)

    with pytest.raises(GapAnalysisServiceV3Error) as exc:
        asyncio.run(
            service.run(
                GapAnalysisRequest(policy_document_id="pol-1", applicable_jurisdictions=["CA"], run_async=False)
            )
        )

    assert exc.value.status_code == 429
    policy_sub.count_documents.assert_not_called()
    gap_check.assert_not_awaited()
    results.insert_one.assert_not_called()


def test_run_policy_not_found() -> None:
    service, _, gap_check, _, _, _ = _build_service(policy_doc=None)
    # find_one returns None for both document_id and _id lookups
    service.mongo[service.cfg.compliance_database][service.cfg.policies_collection].find_one.return_value = None

    with pytest.raises(GapAnalysisServiceV3Error) as exc:
        asyncio.run(
            service.run(
                GapAnalysisRequest(policy_document_id="missing", applicable_jurisdictions=["CA"], run_async=False)
            )
        )

    assert exc.value.status_code == 404
    assert "Policy not found" in str(exc.value)
    gap_check.assert_not_awaited()


def test_run_unindexed_policy_rejected() -> None:
    service, _, gap_check, _, _, policy_sub = _build_service(indexed_count=0)

    with pytest.raises(GapAnalysisServiceV3Error) as exc:
        asyncio.run(
            service.run(
                GapAnalysisRequest(policy_document_id="pol-1", applicable_jurisdictions=["CA"], run_async=False)
            )
        )

    assert exc.value.status_code == 400
    assert "not indexed" in str(exc.value).lower()
    assert service.cfg.policy_sub_embeddings_collection in str(exc.value)
    policy_sub.aggregate.assert_not_called()
    gap_check.assert_not_awaited()


def test_run_citation_binding_downgrades_addressed_and_persists() -> None:
    service, _, gap_check, results, run_log, _ = _build_service(
        llm_result={
            "status": "addressed",
            "policy_quote": "a quote that is not present in the policy",
            "confidence": "high",
            "requirement_summary": "Disclose access rights",
            "statute_quote": "Businesses must disclose consumer access rights.",
        }
    )
    req = GapAnalysisRequest(
        policy_document_id="pol-1",
        applicable_jurisdictions=["CA"],
        run_async=False,
    )

    with patch("uuid.uuid4", return_value="run-v3-1"):
        response = asyncio.run(service.run(req))

    assert response.version == "v3"
    assert response.run_types == ["gap_v3"]
    assert response.company_name == "Example Co"
    assert response.summary.total_requirements == 1
    assert response.summary.missing == 1
    assert response.summary.addressed == 0
    assert response.gaps[0].status == "missing"
    assert response.gaps[0].policy_quote is None
    assert response.gaps[0].citation_binding_failed is True
    assert response.gaps[0].conflict_description == (
        "The policy does not contain provisions that address this statutory requirement."
    )
    assert "Consumers may request access" in gap_check.await_args.kwargs["policy_chunk_text"]
    assert "Parent policy section" in gap_check.await_args.kwargs["policy_chunk_text"]
    assert gap_check.await_args.kwargs["statute_chunk_text"] == (
        "Businesses must disclose consumer access rights."
    )

    results.insert_one.assert_called_once()
    persisted = results.insert_one.call_args.args[0]
    assert persisted["_id"] == "run-v3-1"
    assert persisted["run_type"] == "gap_analysis_v3"
    assert persisted["gaps"][0]["citation_binding_failed"] is True
    run_log.insert_one.assert_called_once()
    assert run_log.insert_one.call_args.args[0]["statute_item_ids"] == ["stat-1"]


def test_run_skips_invalid_embeddings_and_duplicates_honors_num_rows() -> None:
    statutes = [
        _statute_doc(doc_id="bad", embedding=["not-a-float"]),
        _statute_doc(doc_id="stat-1", statute_reference="REF-A"),
        _statute_doc(doc_id="stat-1", statute_reference="REF-A"),  # duplicate key
        _statute_doc(doc_id="stat-2", statute_reference="REF-B", sub_chunk_text="Second requirement."),
    ]
    service, _, gap_check, results, _, _ = _build_service(
        statute_docs=statutes,
        llm_result={
            "status": "missing",
            "policy_quote": None,
            "confidence": "low",
            "requirement_summary": "Need disclosure",
            "conflict_description": "",
        },
    )
    req = GapAnalysisRequest(
        policy_document_id="pol-1",
        applicable_jurisdictions=["CA"],
        num_rows=3,
        save_results=False,
        run_async=False,
    )

    response = asyncio.run(service.run(req))

    # num_rows=3 keeps bad + first two; invalid embedding skipped; duplicate skipped => 1 gap
    assert response.summary.total_requirements == 1
    assert response.summary.missing == 1
    assert response.gaps[0].conflict_description == (
        "The policy does not contain provisions that address this statutory requirement."
    )
    assert gap_check.await_count == 1
    results.insert_one.assert_not_called()


def test_run_vector_search_falls_back_when_filtered_aggregate_fails() -> None:
    fallback_results = [
        {
            "score": 0.88,
            "sub_chunk_text": "Consumers may request access to personal information through account settings.",
            "parent_chunk_id": "parent-pol-1",
        }
    ]
    service, _, gap_check, _, _, policy_sub = _build_service(
        aggregate_side_effect=[RuntimeError("filter unsupported"), fallback_results],
        llm_result={
            "status": "addressed",
            "policy_quote": "Consumers may request access to personal information through account settings.",
            "confidence": "medium",
            "requirement_summary": "Access rights",
        },
    )
    req = GapAnalysisRequest(
        policy_document_id="pol-1",
        applicable_jurisdictions=["CA"],
        save_results=False,
        run_async=False,
    )

    response = asyncio.run(service.run(req))

    assert response.summary.addressed == 1
    assert policy_sub.aggregate.call_count == 2
    second_pipeline = policy_sub.aggregate.call_args_list[1].args[0]
    assert any("$match" in stage for stage in second_pipeline)
    gap_check.assert_awaited_once()


def test_run_loads_policy_chunks_when_text_missing() -> None:
    service, _, gap_check, _, _, _ = _build_service(
        policy_doc={
            "document_id": "pol-1",
            "company_name": "Chunk Co",
            "policy_chunks": [
                {"chunk_text": "First chunk about notices."},
                {"chunk_text": "Second chunk about deletion."},
            ],
        },
        llm_result={
            "status": "conflict",
            "policy_quote": "First chunk about notices.",
            "confidence": "high",
            "requirement_summary": "Notice timing",
            "conflict_description": "Conflict on timing.",
        },
    )
    req = GapAnalysisRequest(
        policy_document_id="pol-1",
        applicable_jurisdictions=["CA"],
        save_results=False,
        run_async=False,
    )

    response = asyncio.run(service.run(req))

    assert response.company_name == "Chunk Co"
    assert response.summary.conflicts == 1
    assert response.gaps[0].status == "conflict"
    assert response.gaps[0].citation_binding_failed is None
    gap_check.assert_awaited_once()


def test_run_persistence_failure_does_not_fail_completed_analysis() -> None:
    service, _, _, results, _, _ = _build_service()
    results.insert_one.side_effect = RuntimeError("mongo write failed")
    req = GapAnalysisRequest(
        policy_document_id="pol-1",
        applicable_jurisdictions=["CA"],
        run_async=False,
    )

    response = asyncio.run(service.run(req))

    assert response.summary.total_requirements == 1
    assert response.summary.addressed == 1
    results.insert_one.assert_called_once()
