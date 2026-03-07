"""Coverage for v4 gap-analysis service and route behavior."""
from __future__ import annotations

import asyncio
import sys
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

from flask import Flask

# Avoid loading sentence_transformers in unit tests.
sys.modules["sentence_transformers"] = MagicMock()
sys.modules["anthropic"] = MagicMock()

from compliance_config import load_config
from compliance_routes_v4 import gap_analysis_v4
from compliance_suite_schemas import GapAnalysisRequest
from gap_analysis_service_v4 import GapAnalysisServiceV4, GapAnalysisServiceV4Error


def _build_v4_service(llm_result: Dict[str, Any]) -> GapAnalysisServiceV4:
    """Create a v4 service with deterministic in-memory mocks."""
    config = load_config()
    mongo = MagicMock()

    db = MagicMock()
    mongo.__getitem__.return_value = db

    policies_coll = MagicMock()
    statute_coll = MagicMock()
    policy_legal_coll = MagicMock()
    cat_map_coll = MagicMock()

    def _get_collection(name: str) -> MagicMock:
        if name == config.policies_collection:
            return policies_coll
        if name == config.statute_sub_topic_embeddings_collection:
            return statute_coll
        if name == config.policy_legal_embeddings_collection:
            return policy_legal_coll
        if name == config.category_mapping_collection:
            return cat_map_coll
        # Defensive fallback for any non-essential collection access.
        return MagicMock()

    db.__getitem__.side_effect = _get_collection

    policies_coll.find_one.return_value = {
        config.policy_document_id_field: "policy-1",
        "text": "Policy body with privacy commitments.",
        "company_name": "Acme",
    }
    policy_legal_coll.count_documents.return_value = 1
    cat_map_coll.find.return_value = [
        {
            "statute_category": "consumer_rights",
            "policy_categories": ["notice"],
            "sub_topic": "retention",
        }
    ]

    statute_doc = {
        "_id": "stat-1",
        "document_id": "ca-privacy-act",
        "header_text": "Notice requirement",
        "subtopic_text": "Must disclose retention period.",
        "requirement_summary": "Disclose retention period.",
        "jurisdiction": "CA",
    }
    context_doc = {
        "category": "definitions",
        "header_text": "Definition",
        "subtopic_text": "Personal data includes identifiers.",
    }

    def _statute_find(query: Dict[str, Any], _projection: Dict[str, int] | None = None):
        if query.get("category") == "consumer_rights":
            return [statute_doc]
        category_filter = query.get("category")
        if isinstance(category_filter, dict) and "$in" in category_filter:
            return [context_doc]
        return []

    statute_coll.find.side_effect = _statute_find
    policy_legal_coll.find.return_value = [
        {"chunk_text": "This policy discusses data handling and retention timelines."}
    ]

    llm = MagicMock()
    llm.gap_check_v4 = AsyncMock(return_value=llm_result)

    limiter = MagicMock()
    limiter.allow = AsyncMock(return_value=True)

    return GapAnalysisServiceV4(
        mongo_client=mongo,
        config=config,
        llm_client=llm,
        rate_limiter=limiter,
    )


def test_gap_analysis_v4_downgrades_addressed_when_quote_not_bound() -> None:
    service = _build_v4_service(
        {
            "status": "addressed",
            "policy_quote": "quote not present in policy text",
            "confidence": "high",
            "requirement_summary": "Disclose retention period.",
            "statute_quote": "Must disclose retention period.",
        }
    )
    request_model = GapAnalysisRequest(
        policy_document_id="policy-1",
        run_async=False,
        save_results=False,
    )

    response = asyncio.run(service.run(request_model))

    assert len(response.gaps) == 1
    gap = response.gaps[0]
    assert gap.status == "missing"
    assert gap.policy_quote is None
    assert gap.citation_binding_failed is True
    assert response.summary.missing == 1
    assert response.summary.addressed == 0


def test_gap_analysis_v4_keeps_partial_when_quote_not_bound() -> None:
    service = _build_v4_service(
        {
            "status": "partial",
            "policy_quote": "quote not present in policy text",
            "confidence": "medium",
            "requirement_summary": "Disclose retention period.",
            "statute_quote": "Must disclose retention period.",
        }
    )
    request_model = GapAnalysisRequest(
        policy_document_id="policy-1",
        run_async=False,
        save_results=False,
    )

    response = asyncio.run(service.run(request_model))

    assert len(response.gaps) == 1
    gap = response.gaps[0]
    assert gap.status == "partial"
    assert gap.policy_quote == "quote not present in policy text"
    assert gap.citation_binding_failed is True
    assert response.summary.partial == 1
    assert response.summary.missing == 0


def test_gap_analysis_v4_route_async_returns_job_id() -> None:
    app = Flask(__name__)
    with app.test_request_context(
        "/api/v4/compliance/gap-analysis",
        method="POST",
        json={"policy_document_id": "policy-1", "run_async": True},
    ):
        with patch("compliance_routes_v4._get_config", return_value=load_config()), patch(
            "compliance_routes_v4.authorize_request"
        ), patch(
            "compliance_routes_v4._start_v4_gap_analysis_job", return_value="job-123"
        ):
            response, status = asyncio.run(gap_analysis_v4())

    assert status == 202
    body = response.get_json()
    assert body["job_id"] == "job-123"
    assert body["status"] == "pending"


def test_gap_analysis_v4_route_propagates_service_error() -> None:
    app = Flask(__name__)
    service = MagicMock()
    service.run = AsyncMock(
        side_effect=GapAnalysisServiceV4Error("Rate limit exceeded", status_code=429)
    )

    with app.test_request_context(
        "/api/v4/compliance/gap-analysis",
        method="POST",
        json={"policy_document_id": "policy-1", "run_async": False},
    ):
        with patch("compliance_routes_v4._get_config", return_value=load_config()), patch(
            "compliance_routes_v4.authorize_request"
        ), patch(
            "compliance_routes_v4._get_gap_analysis_v4_service", return_value=service
        ):
            response, status = asyncio.run(gap_analysis_v4())

    assert status == 429
    body = response.get_json()
    assert "Rate limit exceeded" in body["error"]
