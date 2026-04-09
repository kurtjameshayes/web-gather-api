"""Regression tests for gap analysis v4 service and routes."""
from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from flask import Flask

# Avoid loading sentence_transformers in unit tests.
sys.modules["sentence_transformers"] = MagicMock()

from compliance_config import load_config
from compliance_routes_v4 import compliance_v4_bp
from compliance_suite_schemas import GapAnalysisRequest
from gap_analysis_service_v4 import GapAnalysisServiceV4
from rate_limiter import RateLimiter


class FakeCollection:
    """Small in-memory Mongo-like collection for deterministic tests."""

    def __init__(self, docs=None) -> None:
        self.docs = list(docs or [])
        self.inserted_docs = []

    @staticmethod
    def _matches(doc, query) -> bool:
        for key, expected in (query or {}).items():
            value = doc.get(key)
            if isinstance(expected, dict):
                if "$in" in expected:
                    if value not in expected["$in"]:
                        return False
                else:
                    return False
            elif value != expected:
                return False
        return True

    def find_one(self, query):
        for doc in self.docs:
            if self._matches(doc, query):
                return doc
        return None

    def find(self, query=None, projection=None):
        del projection  # Projection is not needed in these tests.
        return [doc for doc in self.docs if self._matches(doc, query or {})]

    def count_documents(self, query):
        return len(self.find(query))

    def insert_one(self, doc):
        self.inserted_docs.append(doc)
        return MagicMock(inserted_id=doc.get("_id", "inserted"))


class StubLLM:
    def __init__(self, result: dict) -> None:
        self.result = result
        self.calls = []

    async def gap_check_v4(self, reference_context: str, statutory_requirement: str, policy_text: str):
        self.calls.append(
            {
                "reference_context": reference_context,
                "statutory_requirement": statutory_requirement,
                "policy_text": policy_text,
            }
        )
        return dict(self.result)


def _build_service(mongo):
    cfg = load_config()
    llm = StubLLM(
        {
            "status": "addressed",
            "policy_quote": "Quote not present in policy text",
            "confidence": "high",
            "requirement_summary": "Consumers can request access details.",
        }
    )
    service = GapAnalysisServiceV4(
        mongo_client=mongo,
        config=cfg,
        llm_client=llm,
        rate_limiter=RateLimiter(1000),
    )
    return service, llm


def test_gap_analysis_v4_downgrades_unbound_addressed_quote_to_missing() -> None:
    cfg = load_config()
    mongo = {
        cfg.compliance_database: {
            cfg.policies_collection: FakeCollection(
                [
                    {
                        cfg.policy_document_id_field: "policy-1",
                        "text": "We notify users within 45 days.",
                        "company_name": "Acme Corp",
                    }
                ]
            ),
            cfg.statute_sub_topic_embeddings_collection: FakeCollection(
                [
                    {
                        "_id": "stat-1",
                        "document_id": "cpra-doc",
                        "category": "consumer_rights",
                        "sub_topic": "access",
                        "header_text": "Right to access",
                        "subtopic_text": "Consumers may request access to personal information.",
                        "jurisdiction": "CA",
                    },
                    {
                        "_id": "ctx-1",
                        "document_id": "cpra-doc",
                        "category": "definitions",
                        "header_text": "Definitions",
                        "subtopic_text": "Personal information has a broad definition.",
                    },
                ]
            ),
            cfg.policy_legal_embeddings_collection: FakeCollection(
                [
                    {
                        cfg.policy_document_id_field: "policy-1",
                        "category": "notice",
                        "chunk_text": "We notify users within 45 days.",
                    }
                ]
            ),
            cfg.category_mapping_collection: FakeCollection(
                [
                    {
                        "statute_category": "consumer_rights",
                        "policy_categories": ["notice"],
                        "sub_topic": "access",
                    }
                ]
            ),
            "category_mappings": FakeCollection([]),
            cfg.compliance_results_collection: FakeCollection([]),
            cfg.compliance_run_log_collection: FakeCollection([]),
        }
    }
    service, llm = _build_service(mongo)
    req = GapAnalysisRequest(policy_document_id="policy-1", run_async=False, save_results=False)

    response = asyncio.run(service.run(req))

    assert len(response.gaps) == 1
    gap = response.gaps[0]
    assert gap.status == "missing"
    assert gap.policy_quote is None
    assert gap.citation_binding_failed is True
    assert "does not contain provisions" in (gap.conflict_description or "")
    assert response.summary.missing == 1
    assert response.summary.addressed == 0
    assert len(llm.calls) == 1


def test_gap_analysis_v4_uses_category_mappings_fallback_collection() -> None:
    cfg = load_config()
    policy_text = "We notify users within 45 days."
    mongo = {
        cfg.compliance_database: {
            cfg.policies_collection: FakeCollection(
                [
                    {
                        cfg.policy_document_id_field: "policy-2",
                        "text": policy_text,
                    }
                ]
            ),
            cfg.statute_sub_topic_embeddings_collection: FakeCollection(
                [
                    {
                        "_id": "stat-2",
                        "document_id": "cpra-doc",
                        "category": "consumer_rights",
                        "sub_topic": "access",
                        "header_text": "Right to access",
                        "subtopic_text": "Consumers may request access to personal information.",
                        "jurisdiction": "CA",
                    }
                ]
            ),
            cfg.policy_legal_embeddings_collection: FakeCollection(
                [
                    {
                        cfg.policy_document_id_field: "policy-2",
                        "category": "notice",
                        "chunk_text": policy_text,
                    }
                ]
            ),
            cfg.category_mapping_collection: FakeCollection([]),
            "category_mappings": FakeCollection(
                [
                    {
                        "statute_category": "consumer_rights",
                        "policy_categories": ["notice"],
                        "sub_topic": "access",
                    }
                ]
            ),
            cfg.compliance_results_collection: FakeCollection([]),
            cfg.compliance_run_log_collection: FakeCollection([]),
        }
    }
    service, llm = _build_service(mongo)
    llm.result = {
        "status": "partial",
        "policy_quote": "We notify users within 45 days.",
        "confidence": "medium",
        "requirement_summary": "Consumers can request access details.",
    }
    req = GapAnalysisRequest(policy_document_id="policy-2", run_async=False, save_results=False)

    response = asyncio.run(service.run(req))

    assert len(response.gaps) == 1
    assert response.gaps[0].status == "partial"
    assert response.gaps[0].citation_binding_failed is None
    assert response.summary.partial == 1
    assert response.summary.total_requirements == 1


@pytest.fixture
def app() -> Flask:
    app = Flask(__name__)
    app.register_blueprint(compliance_v4_bp, url_prefix="/api/v4/compliance")
    return app


@pytest.fixture
def client(app: Flask):
    return app.test_client()


def test_v4_gap_analysis_route_defaults_to_async_job(client) -> None:
    with patch("compliance_routes_v4.authorize_request"), patch(
        "compliance_routes_v4._get_config", return_value=object()
    ), patch("compliance_routes_v4._start_v4_gap_analysis_job", return_value="job-123") as start_job:
        response = client.post(
            "/api/v4/compliance/gap-analysis",
            json={"policy_document_id": "policy-1"},
        )

    assert response.status_code == 202
    payload = response.get_json()
    assert payload["job_id"] == "job-123"
    assert payload["status"] == "pending"
    start_job.assert_called_once()
    request_dict = start_job.call_args[0][0]
    assert request_dict["policy_document_id"] == "policy-1"
    assert "run_async" not in request_dict


def test_v4_gap_analysis_route_runs_inline_when_run_async_false(client) -> None:
    result = MagicMock()
    result.model_dump.return_value = {"policy_document_id": "policy-1", "gaps": [], "summary": {"total_requirements": 0}}
    service = MagicMock()
    service.run = AsyncMock(return_value=result)

    with patch("compliance_routes_v4.authorize_request"), patch(
        "compliance_routes_v4._get_config", return_value=object()
    ), patch("compliance_routes_v4._get_gap_analysis_v4_service", return_value=service), patch(
        "compliance_routes_v4._start_v4_gap_analysis_job"
    ) as start_job:
        response = client.post(
            "/api/v4/compliance/gap-analysis",
            json={"policy_document_id": "policy-1", "run_async": False},
        )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["policy_document_id"] == "policy-1"
    service.run.assert_awaited_once()
    start_job.assert_not_called()
