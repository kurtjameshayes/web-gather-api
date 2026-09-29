"""Regression tests for GapAnalysisServiceV3 retrieval and citation contracts.

V3 compares statute subchunks to policy text and then decides whether an LLM
quote is allowed to count. These tests pin the decisions that change gap
status: which text is sent, when vector search may widen, and when a quote is
demoted.
"""
from __future__ import annotations

import asyncio
import sys
from unittest.mock import MagicMock

import pytest

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("anthropic", MagicMock())

from compliance_config import load_config
from compliance_suite_schemas import GapAnalysisRequest
from gap_analysis_service_v3 import GapAnalysisServiceV3, GapAnalysisServiceV3Error

MISSING_DEFAULT = (
    "The policy does not contain provisions that address this statutory requirement."
)


class _Limiter:
    def __init__(self, allowed: bool = True) -> None:
        self.allowed = allowed
        self.calls = 0

    async def allow(self) -> bool:
        self.calls += 1
        return self.allowed


def _service(allowed: bool = True):
    collections: dict[str, MagicMock] = {}
    db = MagicMock()

    def get_coll(name: str) -> MagicMock:
        collections.setdefault(name, MagicMock())
        return collections[name]

    db.__getitem__.side_effect = get_coll
    mongo = MagicMock()
    mongo.__getitem__.return_value = db
    config = load_config()
    llm = MagicMock()
    limiter = _Limiter(allowed)
    service = GapAnalysisServiceV3(
        mongo,
        config,
        MagicMock(),
        llm,
        MagicMock(),
        limiter,
    )

    def collection(name: str) -> MagicMock:
        return get_coll(name)

    return service, collection, config, llm, limiter, mongo


def _policy_found(policies, document):
    def find_one(query):
        if query.get("document_id") == document.get("document_id"):
            return document
        return None

    policies.find_one.side_effect = find_one


def _vec(config, vector, **doc):
    doc[config.embedding_vector_field] = vector
    return doc


def _request(**overrides) -> GapAnalysisRequest:
    payload = {
        "policy_document_id": "pol-1",
        "applicable_jurisdictions": ["ZZ"],
        "save_results": False,
    }
    payload.update(overrides)
    return GapAnalysisRequest(**payload)


def test_v3_citation_binding_prefers_full_policy_text() -> None:
    service, collection, config, llm, _limiter, _mongo = _service()
    policies = collection(config.policies_collection)
    _policy_found(
        policies,
        {
            "document_id": "pol-1",
            "text": "We  disclose\ncategories of personal information.",
            "company_name": "Acme",
        },
    )
    collection(config.policy_sub_embeddings_collection).count_documents.return_value = 1
    collection(config.statute_sub_embeddings_collection).find.return_value = [
        _vec(
            config,
            [1, 0],
            _id="bound",
            statute_reference="Civ. 1",
            jurisdiction="ZZ",
            sub_chunk_text="Duty to disclose categories.",
        ),
        _vec(
            config,
            [0, 1],
            _id="demote",
            statute_reference="Civ. 2",
            jurisdiction="ZZ",
            sub_chunk_text="Consumers may opt out of sale.",
        ),
        _vec(
            config,
            [0, 2],
            _id="keep-conflict",
            statute_reference="Civ. 3",
            jurisdiction="ZZ",
            sub_chunk_text="Do not sell personal information.",
        ),
    ]

    def aggregate(pipe):
        vector = pipe[0]["$vectorSearch"]["queryVector"]
        quotes = {
            (1.0, 0.0): "unrelated match",
            (0.0, 1.0): "Consumers may opt out of sale.",
            (0.0, 2.0): "not in the policy",
        }
        return [{"score": 0.9, "sub_chunk_text": quotes[tuple(vector)]}]

    collection(config.policy_sub_embeddings_collection).aggregate.side_effect = aggregate
    responses = [
        {
            "status": "addressed",
            "policy_quote": "we disclose categories of personal information.",
            "requirement_summary": "Disclose categories",
            "confidence": "high",
        },
        {
            "status": "addressed",
            "policy_quote": "Consumers may opt out of sale.",
            "requirement_summary": "Opt out",
            "confidence": "high",
        },
        {
            "status": "conflict",
            "policy_quote": "not present anywhere",
            "requirement_summary": "Sale limit",
            "conflict_description": "Policy allows sale.",
            "confidence": "medium",
        },
    ]

    async def gap_check_v3(**_kwargs):
        return responses.pop(0)

    llm.gap_check_v3 = gap_check_v3

    response = asyncio.run(service.run(_request()))

    bound, demoted, conflict = response.gaps
    assert bound.status == "addressed"
    assert bound.policy_quote == "we disclose categories of personal information."
    assert bound.citation_binding_failed is None
    assert demoted.status == "missing"
    assert demoted.policy_quote is None
    assert demoted.citation_binding_failed is True
    assert demoted.conflict_description == MISSING_DEFAULT
    assert conflict.status == "conflict"
    assert conflict.policy_quote == "not present anywhere"
    assert conflict.citation_binding_failed is True
    assert conflict.conflict_description == "Policy allows sale."
    assert response.summary.addressed == 1
    assert response.summary.missing == 1
    assert response.summary.conflicts == 1
    assert response.company_name == "Acme"
    assert response.policy_document_ids is None


def test_v3_statute_parent_is_not_sent_to_the_llm() -> None:
    service, collection, config, llm, _limiter, _mongo = _service()
    policies = collection(config.policies_collection)
    _policy_found(
        policies,
        {"document_id": "pol-1", "text": "Policy body.", "company_name": "Acme"},
    )
    collection(config.policy_sub_embeddings_collection).count_documents.return_value = 1
    collection(config.statute_sub_embeddings_collection).find.return_value = [
        _vec(
            config,
            [1.0],
            _id="s1",
            parent_chunk_id="statute-parent",
            statute_reference="Civ. 9",
            jurisdiction="ZZ",
            sub_chunk_text="Statute subchunk only.",
        )
    ]
    collection(config.statute_embeddings_collection).find_one.return_value = {
        "_id": "statute-parent",
        "chunk_text": "Statute parent context.",
    }
    collection(config.policy_embeddings_collection).find.return_value = [
        {"_id": "policy-parent", "chunk_text": "Policy parent section."}
    ]
    collection(config.policy_sub_embeddings_collection).aggregate.return_value = [
        {
            "score": 0.8,
            "sub_chunk_text": "Policy subchunk sentence.",
            "parent_chunk_id": "policy-parent",
        }
    ]
    captured = {}

    async def gap_check_v3(**kwargs):
        captured.update(kwargs)
        return {
            "status": "missing",
            "requirement_summary": "Parent handling",
            "confidence": "low",
        }

    llm.gap_check_v3 = gap_check_v3

    response = asyncio.run(service.run(_request()))

    assert captured["statute_chunk_text"] == "Statute subchunk only."
    assert "Statute parent context." not in captured["statute_chunk_text"]
    assert captured["policy_chunk_text"] == "Policy parent section.\n\nPolicy subchunk sentence."
    gap = response.gaps[0]
    assert gap.statute_chunk_text == "Statute subchunk only."
    assert gap.statute_subchunk_text == "Statute subchunk only."
    assert gap.policy_subchunk_text == "Policy subchunk sentence."
    assert gap.policy_combined_sections == "Policy parent section.\n\nPolicy subchunk sentence."
    collection(config.statute_embeddings_collection).find_one.assert_called_once()
    assert response.summary.missing == 1


def test_v3_vector_search_fallback_and_low_score_kept() -> None:
    service, collection, config, llm, _limiter, _mongo = _service()
    policies = collection(config.policies_collection)
    _policy_found(policies, {"document_id": "pol-1", "text": "Policy body."})
    sub_coll = collection(config.policy_sub_embeddings_collection)
    sub_coll.count_documents.return_value = 1
    collection(config.statute_sub_embeddings_collection).find.return_value = [
        _vec(config, "nope", _id="bad", statute_reference="A", sub_chunk_text="skip"),
        _vec(config, [1, 0], _id="empty", statute_reference="B", sub_chunk_text="empty search"),
        _vec(config, [2, 0], _id="fallback", statute_reference="C", sub_chunk_text="fallback search"),
    ]
    calls: list[list] = []

    def aggregate(pipe):
        calls.append(pipe)
        vector = tuple(pipe[0]["$vectorSearch"]["queryVector"])
        if vector == (1.0, 0.0):
            assert "filter" in pipe[0]["$vectorSearch"]
            return []
        if vector == (2.0, 0.0) and "filter" in pipe[0]["$vectorSearch"]:
            raise RuntimeError("index has no filter")
        assert {"$match": {config.policy_document_id_field: "pol-1"}} in pipe
        return [{"score": 0.01, "sub_chunk_text": "low score still included"}]

    sub_coll.aggregate.side_effect = aggregate
    prompts: list[str] = []

    async def gap_check_v3(**kwargs):
        prompts.append(kwargs["policy_chunk_text"])
        return {"status": "missing", "requirement_summary": "Searched", "confidence": "low"}

    llm.gap_check_v3 = gap_check_v3

    response = asyncio.run(service.run(_request()))

    assert [call[0]["$vectorSearch"]["queryVector"] for call in calls] == [
        [1.0, 0.0],
        [2.0, 0.0],
        [2.0, 0.0],
    ]
    assert calls[0][0]["$vectorSearch"]["filter"] == {config.policy_document_id_field: "pol-1"}
    assert "filter" not in calls[2][0]["$vectorSearch"]
    assert prompts == ["", "low score still included"]
    assert response.retrieval_metadata.statute_items_considered == 3
    assert response.retrieval_metadata.statute_pairs_matched == 2
    assert [gap.statute_chunk_id for gap in response.gaps] == ["empty", "fallback"]


def test_v3_duplicate_key_skips_second_llm_after_search() -> None:
    service, collection, config, llm, _limiter, _mongo = _service()
    policies = collection(config.policies_collection)
    _policy_found(policies, {"document_id": "pol-1", "text": "Policy body."})
    collection(config.policy_sub_embeddings_collection).count_documents.return_value = 1
    collection(config.statute_sub_embeddings_collection).find.return_value = [
        _vec(config, [1, 0], _id="same", statute_reference="Civ. 1", sub_chunk_text="First"),
        _vec(config, [0, 1], _id="same", statute_reference="Civ. 1", sub_chunk_text="Second"),
    ]
    collection(config.policy_sub_embeddings_collection).aggregate.return_value = [
        {"score": 0.5, "sub_chunk_text": "match"}
    ]
    calls = {"n": 0}

    async def gap_check_v3(**_kwargs):
        calls["n"] += 1
        return {"status": "missing", "requirement_summary": "Once", "confidence": "low"}

    llm.gap_check_v3 = gap_check_v3

    response = asyncio.run(service.run(_request()))

    assert collection(config.policy_sub_embeddings_collection).aggregate.call_count == 2
    assert calls["n"] == 1
    assert response.retrieval_metadata.statute_items_considered == 2
    assert response.retrieval_metadata.statute_pairs_matched == 1
    assert response.statute_chunk_ids_used == ["same"]


def test_v3_failure_confidence_and_persist() -> None:
    service, collection, config, llm, _limiter, _mongo = _service()
    policies = collection(config.policies_collection)
    _policy_found(policies, {"document_id": "pol-1", "text": "Policy body."})
    collection(config.policy_sub_embeddings_collection).count_documents.return_value = 1
    collection(config.statute_sub_embeddings_collection).find.return_value = [
        _vec(config, [1.0], _id="s1", statute_reference="Civ. 1", sub_chunk_text="Duty")
    ]
    collection(config.policy_sub_embeddings_collection).aggregate.return_value = []

    async def gap_check_v3(**_kwargs):
        return {
            "status": "missing",
            "policy_quote": "invented",
            "requirement_summary": "Failed parse",
            "confidence": "High",
            "_analysis_failed": True,
        }

    llm.gap_check_v3 = gap_check_v3
    results = collection(config.compliance_results_collection)
    run_log = collection(config.compliance_run_log_collection)
    results.insert_one.side_effect = RuntimeError("write failed")

    failed = asyncio.run(service.run(_request(save_results=True)))

    assert failed.gaps[0].analysis_failed is True
    assert failed.gaps[0].policy_quote == "invented"
    assert failed.gaps[0].citation_binding_failed is None
    assert failed.gaps[0].confidence is None
    assert failed.summary.total_requirements == 1
    assert failed.summary.missing == 0
    assert failed.summary.analysis_failures == 1
    results.insert_one.assert_called_once()
    stored = results.insert_one.call_args[0][0]
    assert stored["run_type"] == "gap_analysis_v3"
    assert stored["version"] == "v3"
    assert "policy_document_ids" not in stored
    run_log.insert_one.assert_not_called()

    results.insert_one.reset_mock()
    skipped = asyncio.run(service.run(_request(save_results=False)))
    assert skipped.summary.analysis_failures == 1
    results.insert_one.assert_not_called()
    run_log.insert_one.assert_not_called()


def test_v3_rejects_before_retrieval() -> None:
    service, _collection, config, _llm, limiter, mongo = _service(allowed=False)

    with pytest.raises(GapAnalysisServiceV3Error) as empty_ids:
        asyncio.run(
            service.run(
                GapAnalysisRequest.model_construct(
                    policy_document_id=None,
                    policy_document_ids=None,
                )
            )
        )
    assert empty_ids.value.status_code == 400
    assert limiter.calls == 0
    mongo.__getitem__.assert_not_called()

    with pytest.raises(GapAnalysisServiceV3Error) as limited:
        asyncio.run(service.run(_request()))
    assert limited.value.status_code == 429
    mongo.__getitem__.assert_not_called()

    service, collection, config, _llm, _limiter, _mongo = _service()
    collection(config.policies_collection).find_one.return_value = None
    with pytest.raises(GapAnalysisServiceV3Error) as missing:
        asyncio.run(service.run(_request(policy_document_id="missing-policy")))
    assert missing.value.status_code == 404
    assert "missing-policy" in str(missing.value)
    collection(config.policy_sub_embeddings_collection).count_documents.assert_not_called()

    service, collection, config, _llm, _limiter, _mongo = _service()
    _policy_found(
        collection(config.policies_collection),
        {"document_id": "pol-1", "text": "Policy body."},
    )
    collection(config.policy_sub_embeddings_collection).count_documents.return_value = 0
    with pytest.raises(GapAnalysisServiceV3Error) as unindexed:
        asyncio.run(service.run(_request()))
    assert unindexed.value.status_code == 400
    assert "Policy not indexed" in str(unindexed.value)
    count_query = collection(config.policy_sub_embeddings_collection).count_documents.call_args[0][0]
    assert count_query == {config.policy_document_id_field: {"$in": ["pol-1"]}}
    collection(config.statute_sub_embeddings_collection).find.assert_not_called()


def test_v3_multi_policy_assembles_chunks_and_ignores_single_id() -> None:
    service, collection, config, llm, _limiter, _mongo = _service()
    documents = {
        "pol-a": {
            "document_id": "pol-a",
            "text": "   ",
            "company_name": 12,
            "policy_chunks": [
                "not-a-chunk",
                {"chunk_text": "  "},
                {"chunk_text": "First chunk."},
                {"chunk_header_text": "ignored header"},
                {"chunk_text": "Second chunk."},
            ],
        },
        "pol-b": {
            "document_id": "pol-b",
            "text": "Beta policy text.",
            "company_name": "Beta",
        },
    }

    def find_one(query):
        return documents.get(query.get("document_id"))

    collection(config.policies_collection).find_one.side_effect = find_one
    collection(config.policy_sub_embeddings_collection).count_documents.return_value = 2
    collection(config.statute_sub_embeddings_collection).find.return_value = [
        _vec(config, [1.0], _id="s1", statute_reference="Civ. 1", sub_chunk_text="Duty")
    ]
    collection(config.policy_sub_embeddings_collection).aggregate.return_value = []
    assembled = "First chunk.\n\n\n\nSecond chunk.\n\n---\n\nBeta policy text."

    async def gap_check_v3(**_kwargs):
        return {
            "status": "addressed",
            "policy_quote": assembled,
            "requirement_summary": "Chunks",
            "confidence": "high",
        }

    llm.gap_check_v3 = gap_check_v3

    response = asyncio.run(
        service.run(
            _request(
                policy_document_id="ignored-single",
                policy_document_ids=["pol-a", "pol-b"],
            )
        )
    )

    count_query = collection(config.policy_sub_embeddings_collection).count_documents.call_args[0][0]
    assert count_query == {config.policy_document_id_field: {"$in": ["pol-a", "pol-b"]}}
    vector_filter = collection(config.policy_sub_embeddings_collection).aggregate.call_args[0][0][0][
        "$vectorSearch"
    ]["filter"]
    assert vector_filter == {config.policy_document_id_field: {"$in": ["pol-a", "pol-b"]}}
    assert response.policy_document_id == "pol-a"
    assert response.policy_document_ids == ["pol-a", "pol-b"]
    assert response.company_name == "Beta"
    assert response.gaps[0].status == "addressed"
    assert response.gaps[0].policy_quote == assembled
    assert response.gaps[0].citation_binding_failed is None


def test_v3_num_rows_slices_before_search() -> None:
    service, collection, config, llm, _limiter, _mongo = _service()
    _policy_found(
        collection(config.policies_collection),
        {"document_id": "pol-1", "text": "Policy body."},
    )
    collection(config.policy_sub_embeddings_collection).count_documents.return_value = 1
    collection(config.statute_sub_embeddings_collection).find.return_value = [
        _vec(config, [1, 0], _id="first", statute_reference="A", sub_chunk_text="First duty"),
        _vec(config, [0, 1], _id="second", statute_reference="B", sub_chunk_text="Second duty"),
    ]
    collection(config.policy_sub_embeddings_collection).aggregate.return_value = []

    async def gap_check_v3(**kwargs):
        return {
            "status": "missing",
            "requirement_summary": kwargs["statute_chunk_text"],
            "confidence": "low",
        }

    llm.gap_check_v3 = gap_check_v3

    response = asyncio.run(service.run(_request(num_rows=1)))

    assert collection(config.policy_sub_embeddings_collection).aggregate.call_count == 1
    vector = collection(config.policy_sub_embeddings_collection).aggregate.call_args[0][0][0][
        "$vectorSearch"
    ]["queryVector"]
    assert vector == [1.0, 0.0]
    assert response.retrieval_metadata.statute_items_considered == 1
    assert len(response.gaps) == 1
    assert response.gaps[0].statute_chunk_id == "first"
