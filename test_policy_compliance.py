"""Tests for statute-policy compliance service."""
from __future__ import annotations

import asyncio
import json
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest
from flask import Flask

# Avoid loading sentence_transformers in unit tests.
sys.modules["sentence_transformers"] = MagicMock()

from cache import SimpleLRUCache
from compliance_config import load_config
from compliance_evaluator import ComplianceEvaluator
import compliance_routes
from compliance_routes import compliance_bp
from compliance_suite_schemas import ConsumerRightsRouterRequest
from compliance_suite_service import ComplianceSuiteService
from llm_client import build_prompt
from redactor import Redactor
from segmenter import PolicySegmenter
from vector_retriever import StatuteCandidate, VectorRetriever
from rate_limiter import RateLimiter
from schemas import StatutePolicyComplianceResponse


def test_segmenter_splits_on_headings():
    segmenter = PolicySegmenter(max_section_chars=2000)
    text = "DATA RETENTION\nWe keep data for analytics.\n\nSECURITY\nWe protect data."
    sections = segmenter.segment(text)
    assert len(sections) == 2
    assert sections[0].section_id.startswith("data_retention")
    assert "We keep data" in sections[0].section_text


def test_build_prompt_uses_template():
    candidate = StatuteCandidate(
        statute_id="stat-1",
        jurisdiction="US",
        title="Retention Limits",
        section_id="R-1",
        chunk_text="Data retention must not exceed 3 years.",
        score=0.87,
        chunk_id="chunk-1",
    )
    prompt = build_prompt("We retain user data for 10 years.", [candidate])
    assert "You are a privacy law analyst." in prompt
    assert "Candidate statutes (top K):" in prompt
    assert "[stat-1] US Retention Limits" in prompt
    assert "Output only valid JSON" in prompt
    assert "Use exactly these keys:" in prompt


def test_evaluator_respects_confidence_threshold():
    evaluator = ComplianceEvaluator(evidence_score_threshold=0.2)
    candidates = [
        StatuteCandidate(
            statute_id="stat-1",
            jurisdiction="US",
            title="Retention Limits",
            section_id="R-1",
            chunk_text="Data retention must not exceed 3 years.",
            score=0.9,
            chunk_id="chunk-1",
        )
    ]
    llm_payload = json.dumps(
        {
            "section_id": "section_1",
            "applied_statutes": [
                {
                    "statute_id": "stat-1",
                    "jurisdiction": "US",
                    "title": "Retention Limits",
                    "matched_span": "10 years",
                    "evidence_score": 0.9,
                }
            ],
            "compliance": "compliant",
            "confidence": 0.5,
            "rationale": "Insufficient alignment.",
            "remediation_suggestions": ["Shorten retention."],
        }
    )
    result = evaluator.evaluate(
        section_id="section_1",
        section_text="We retain user data for 10 years.",
        candidates=candidates,
        llm_response_text=llm_payload,
        thresholds=load_config().confidence_thresholds,
        explainability=True,
    )
    assert result.compliance == "neither"


def test_retriever_returns_statute_candidates():
    class DummyEmbedder:
        async def embed(self, text):
            return [0.1, 0.2]

    mock_collection = MagicMock()
    mock_collection.aggregate.return_value = [
        {
            "_id": "stat-1",
            "jurisdiction": "US",
            "title": "Retention Limits",
            "section_text": "Retention must not exceed 3 years.",
            "section_id": "R-1",
            "score": 0.77,
        }
    ]
    mock_client = MagicMock()
    mock_client.__getitem__.return_value.__getitem__.return_value = mock_collection

    config = load_config()
    config.use_embeddings_collection = False

    retriever = VectorRetriever(
        mock_client,
        DummyEmbedder(),
        config,
        SimpleLRUCache(10, 60),
    )

    results = asyncio.run(
        retriever.retrieve(
            database="privacy_db",
            section_text="We retain data for 10 years.",
            jurisdiction="US",
            statute_corpus_id=None,
            top_k=1,
        )
    )
    assert results[0].statute_id == "stat-1"
    assert results[0].score == 0.77


def test_statute_policy_compliance_endpoint_delegates_to_v4():
    """Verify the statute-policy endpoint delegates to GapAnalysisServiceV4 and returns gap items."""
    from compliance_suite_schemas import GapAnalysisResponse, GapItem, GapSummary, RetrievalMetadata

    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    config = load_config()
    config.auth_required = False

    fake_response = GapAnalysisResponse(
        policy_document_id="test-policy-id",
        applicable_jurisdictions=["CA"],
        analyzed_at="2026-03-10T00:00:00Z",
        gaps=[
            GapItem(
                jurisdiction="CA",
                statute_reference="CCPA § 1798.100(a)",
                requirement_summary="Right to know what personal information is collected",
                status="missing",
                conflict_description="The policy does not contain provisions that address this statutory requirement.",
            )
        ],
        summary=GapSummary(total_requirements=1, missing=1),
        retrieval_metadata=RetrievalMetadata(statute_items_considered=1, statute_pairs_matched=1),
    )

    mock_v4 = MagicMock()
    mock_v4.run = AsyncMock(return_value=fake_response)

    old_v4 = compliance_routes._gap_analysis_v4_service
    old_config = compliance_routes._config
    compliance_routes._gap_analysis_v4_service = mock_v4
    compliance_routes._config = config

    try:
        app = Flask(__name__)
        app.register_blueprint(compliance_bp)
        client = app.test_client()
        response = client.post(
            "/statute-policy-compliance",
            json={
                "policy_id": "test-policy-id",
                "jurisdiction": "CA",
            },
        )

        assert response.status_code == 200
        body = response.get_json()
        StatutePolicyComplianceResponse.model_validate(body)
        assert len(body["gaps"]) == 1
        assert body["gaps"][0]["status"] == "missing"
        assert body["gaps"][0]["statute_reference"] == "CCPA § 1798.100(a)"
        assert body["summary"]["total_requirements"] == 1
        assert body["summary"]["missing"] == 1
        assert body["applicable_jurisdictions"] == ["CA"]

        call_args = mock_v4.run.call_args[0][0]
        assert call_args.policy_document_id == "test-policy-id"
        assert call_args.applicable_jurisdictions == ["CA"]
        assert call_args.save_results is False
    finally:
        compliance_routes._gap_analysis_v4_service = old_v4
        compliance_routes._config = old_config


def test_adversarial_ambiguous_language_neither():
    evaluator = ComplianceEvaluator(evidence_score_threshold=0.2)
    candidates = [
        StatuteCandidate(
            statute_id="stat-2",
            jurisdiction="US",
            title="Consent Requirements",
            section_id="C-1",
            chunk_text="Explicit consent is required for data sharing.",
            score=0.88,
            chunk_id="C-1",
        )
    ]
    llm_payload = json.dumps(
        {
            "section_id": "section_1",
            "applied_statutes": [],
            "compliance": "neither",
            "confidence": 0.4,
            "rationale": "The policy language is ambiguous.",
            "remediation_suggestions": ["Clarify consent requirements."],
        }
    )
    result = evaluator.evaluate(
        section_id="section_1",
        section_text="We may share data as appropriate.",
        candidates=candidates,
        llm_response_text=llm_payload,
        thresholds=load_config().confidence_thresholds,
        explainability=False,
    )
    assert result.compliance == "neither"


def test_precision_recall_f1_on_labeled_dataset():
    labeled = [
        ("retain 10 years", "non_compliant"),
        ("we obtain consent", "compliant"),
        ("ambiguous statement", "neither"),
    ]
    predicted = ["non_compliant", "compliant", "neither"]

    def metrics(labels, preds, target):
        tp = sum(1 for l, p in zip(labels, preds) if l == target and p == target)
        fp = sum(1 for l, p in zip(labels, preds) if l != target and p == target)
        fn = sum(1 for l, p in zip(labels, preds) if l == target and p != target)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return precision, recall, f1

    labels = [item[1] for item in labeled]
    precision, recall, f1 = metrics(labels, predicted, "non_compliant")
    assert precision == 1.0
    assert recall == 1.0
    assert f1 == 1.0


def test_consumer_rights_router_endpoint_validation_error_details_are_json_safe():
    app = Flask(__name__)
    app.register_blueprint(compliance_bp)
    client = app.test_client()

    response = client.post(
        "/consumer-rights-router",
        json={
            "text": "Privacy policy text",
            "request_types": ["invalid-type"],
        },
    )

    assert response.status_code == 422
    body = response.get_json()
    assert body["error"] == "Validation error"
    assert isinstance(body["details"], list) and body["details"]
    assert "Invalid request_types" in body["details"][0]["msg"]
    if "ctx" in body["details"][0]:
        assert isinstance(body["details"][0]["ctx"], str)


def _build_service_for_consumer_router_tests():
    config = load_config()
    config.default_jurisdictions = ["CA"]
    config.llm_concurrency = 4
    config.consumer_rights_router_prompt_path = "prompts/consumer_rights_router.yaml"

    rate_limiter = MagicMock()
    rate_limiter.allow = AsyncMock(return_value=True)

    llm_client = MagicMock()
    llm_client.consumer_rights_router = AsyncMock()

    storage = MagicMock()
    storage.write_compliance_result = AsyncMock()

    service = ComplianceSuiteService(
        mongo_client=MagicMock(),
        config=config,
        retriever=MagicMock(),
        llm_client=llm_client,
        storage=storage,
        rate_limiter=rate_limiter,
    )
    return service, llm_client, storage


def test_consumer_rights_router_service_handles_none_results_and_fallback_jurisdictions():
    service, llm_client, storage = _build_service_for_consumer_router_tests()
    service._fetch_v4_statute_docs = AsyncMock(
        return_value={
            "CA": [{"header_text": "CCPA", "subtopic_text": "Delete requests"}],
            "NY": [],
        }
    )

    llm_client.consumer_rights_router.side_effect = [
        {
            "trees": {"california": {"id": "ca-del-1", "action": "DELETE"}},
            "policy_gaps": {"california": {"covered": True, "gap": None}},
        },
        None,
    ]

    req = ConsumerRightsRouterRequest(
        text="Sample policy text",
        applicable_jurisdictions=["CA", "NY"],
        request_types=["deletion", "access"],
        save_results=True,
    )
    response = asyncio.run(service.consumer_rights_router(req))

    assert response.policy_document_id is None
    assert response.company_name is None
    assert response.states["california"].abbr == "CA"
    assert response.states["ny"].abbr == "NY"
    assert response.request_types["deletion"] == "Right to Delete"
    assert response.request_types["access"] == "Right to Access / Know"
    assert response.trees["deletion"]["california"]["id"] == "ca-del-1"
    assert response.trees["access"] == {}
    assert response.policy_gaps["access"] == {}
    assert storage.write_compliance_result.await_count == 0
    assert llm_client.consumer_rights_router.await_count == 2
    first_call = llm_client.consumer_rights_router.await_args_list[0]
    assert first_call.kwargs["request_type_label"] == "Right to Delete"
    assert first_call.kwargs["prompt_path"] == "prompts/consumer_rights_router.yaml"


def test_consumer_rights_router_service_persists_for_document_id_and_skips_failed_type():
    service, llm_client, storage = _build_service_for_consumer_router_tests()
    service._load_policy_from_chunks = AsyncMock(return_value=("Policy text from chunks", "Acme"))
    service._fetch_v4_statute_docs = AsyncMock(return_value={"CA": []})

    llm_client.consumer_rights_router.side_effect = [
        RuntimeError("llm failed"),
        {
            "trees": {"california": {"id": "ca-opt-1", "action": "HANDLE_OPT_OUT"}},
            "policy_gaps": {
                "california": {"covered": 1, "gap": "Missing explicit deadline"},
                "ignore_me": "not-a-dict",
            },
        },
    ]

    req = ConsumerRightsRouterRequest(
        policy_document_id="doc-9",
        applicable_jurisdictions=["CA"],
        request_types=["deletion", "optout"],
        save_results=True,
    )
    response = asyncio.run(service.consumer_rights_router(req))

    assert response.policy_document_id == "doc-9"
    assert response.company_name == "Acme"
    assert "deletion" not in response.trees
    assert response.trees["optout"]["california"]["id"] == "ca-opt-1"
    assert response.policy_gaps["optout"]["california"].covered is True
    assert response.policy_gaps["optout"]["california"].gap == "Missing explicit deadline"

    assert storage.write_compliance_result.await_count == 1
    saved_doc = storage.write_compliance_result.await_args.args[0]
    assert saved_doc["result_type"] == "consumer_rights_router"
    assert saved_doc["policy_document_id"] == "doc-9"
