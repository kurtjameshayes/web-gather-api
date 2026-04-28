"""Service-level regression tests for GapAnalysisServiceV4."""
from __future__ import annotations

import asyncio
import sys
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

sys.modules.setdefault("anthropic", MagicMock())

from compliance_config import load_config
from compliance_suite_schemas import GapAnalysisRequest
from gap_analysis_service_v4 import GapAnalysisServiceV4


class FakeCollection:
    """Small Mongo collection fake for deterministic service tests."""

    def __init__(self, docs: Optional[List[Dict[str, Any]]] = None) -> None:
        self.docs = list(docs or [])
        self.find_calls: List[tuple[Dict[str, Any], Any]] = []
        self.find_one_calls: List[Dict[str, Any]] = []
        self.count_documents_calls: List[Dict[str, Any]] = []
        self.inserted: List[Dict[str, Any]] = []

    def find_one(self, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        self.find_one_calls.append(query)
        for doc in self.docs:
            if _matches(doc, query):
                return dict(doc)
        return None

    def count_documents(self, query: Dict[str, Any]) -> int:
        self.count_documents_calls.append(query)
        return sum(1 for doc in self.docs if _matches(doc, query))

    def find(self, query: Optional[Dict[str, Any]] = None, projection: Any = None) -> List[Dict[str, Any]]:
        query = query or {}
        self.find_calls.append((query, projection))
        return [dict(doc) for doc in self.docs if _matches(doc, query)]

    def insert_one(self, doc: Dict[str, Any]) -> MagicMock:
        self.inserted.append(dict(doc))
        return MagicMock(inserted_id=doc.get("_id"))


class FakeDatabase:
    def __init__(self, collections: Dict[str, FakeCollection]) -> None:
        self.collections = collections

    def __getitem__(self, name: str) -> FakeCollection:
        return self.collections[name]


class FakeMongo:
    def __init__(self, databases: Dict[str, FakeDatabase]) -> None:
        self.databases = databases

    def __getitem__(self, name: str) -> FakeDatabase:
        return self.databases[name]


def _matches(doc: Dict[str, Any], query: Dict[str, Any]) -> bool:
    for key, expected in query.items():
        actual = doc.get(key)
        if isinstance(expected, dict) and "$in" in expected:
            if actual not in expected["$in"]:
                return False
        elif actual != expected:
            return False
    return True


def _config():
    config = load_config()
    config.compliance_database = "privacy-compliance"
    config.statute_database = ""
    config.default_jurisdictions = ["CA"]
    config.adaptive_feedback_enabled = False
    return config


def _mongo_for_v4(policy_text: str, policy_chunk_text: str) -> tuple[FakeMongo, Dict[str, FakeCollection]]:
    config = _config()
    collections = {
        config.policies_collection: FakeCollection(
            [
                {
                    config.policy_document_id_field: "pol-1",
                    "text": policy_text,
                    "company_name": "Acme",
                }
            ]
        ),
        config.policy_legal_embeddings_collection: FakeCollection(
            [
                {
                    config.policy_document_id_field: "pol-1",
                    "category": "privacy_rights",
                    "chunk_text": policy_chunk_text,
                }
            ]
        ),
        config.category_mapping_collection: FakeCollection(
            [
                {
                    "statute_category": "consumer_rights",
                    "policy_categories": ["privacy_rights"],
                    "sub_topic": "delete",
                }
            ]
        ),
        config.statute_sub_topic_embeddings_collection: FakeCollection(
            [
                {
                    "_id": "stat-1",
                    "document_id": "ccpa",
                    "category": "consumer_rights",
                    "sub_topic": "delete",
                    "jurisdiction": "CA",
                    "header_text": "CCPA deletion right",
                    "subtopic_text": "Consumers may request deletion of personal information.",
                    "requirement_summary": "Right to delete",
                },
                {
                    "_id": "ctx-1",
                    "document_id": "ccpa",
                    "category": "definitions",
                    "header_text": "Definitions",
                    "subtopic_text": "Consumer means a California resident.",
                },
            ]
        ),
        config.compliance_results_collection: FakeCollection(),
        config.compliance_run_log_collection: FakeCollection(),
    }
    return FakeMongo({"privacy-compliance": FakeDatabase(collections)}), collections


def test_run_demotes_addressed_gap_when_policy_quote_is_not_bound_to_policy_text() -> None:
    """Hallucinated policy quotes must not let a requirement pass as addressed."""
    config = _config()
    mongo, _collections = _mongo_for_v4(
        policy_text="You may request deletion of personal information.",
        policy_chunk_text="You may request deletion of personal information.",
    )
    llm = MagicMock()
    llm.gap_check_v4 = AsyncMock(
        return_value={
            "status": "addressed",
            "policy_quote": "We delete every record within one day.",
            "confidence": "high",
            "requirement_summary": "Right to delete",
            "statute_quote": "Consumers may request deletion of personal information.",
        }
    )
    rate_limiter = MagicMock()
    rate_limiter.allow = AsyncMock(return_value=True)

    service = GapAnalysisServiceV4(mongo, config, llm, rate_limiter)
    response = asyncio.run(
        service.run(
            GapAnalysisRequest(
                policy_document_id="pol-1",
                applicable_jurisdictions=["CA"],
                run_async=False,
                save_results=False,
            )
        )
    )

    assert response.summary.total_requirements == 1
    assert response.summary.missing == 1
    assert response.summary.addressed == 0
    gap = response.gaps[0]
    assert gap.status == "missing"
    assert gap.policy_quote is None
    assert gap.citation_binding_failed is True
    assert "does not contain provisions" in gap.conflict_description
    llm.gap_check_v4.assert_awaited_once()
    assert llm.gap_check_v4.await_args.kwargs["adaptive_feedback"] == ""


def test_run_injects_feedback_records_usage_and_triggers_critic_for_persisted_runs() -> None:
    """Adaptive feedback should be injected, audited, and evaluated on saved v4 runs."""
    config = _config()
    config.adaptive_feedback_enabled = True
    policy_text = "You may request deletion of personal information."
    mongo, collections = _mongo_for_v4(policy_text=policy_text, policy_chunk_text=policy_text)
    llm = MagicMock()
    llm.gap_check_v4 = AsyncMock(
        return_value={
            "status": "addressed",
            "policy_quote": "You may request deletion of personal information.",
            "confidence": "medium",
            "requirement_summary": "Right to delete",
            "statute_quote": "Consumers may request deletion of personal information.",
        }
    )
    rate_limiter = MagicMock()
    rate_limiter.allow = AsyncMock(return_value=True)
    critic = MagicMock()
    critic.get_active_feedback.return_value = [
        {"_id": "fb-1", "suggestions": [{"instruction": "Require verbatim quotes."}]}
    ]
    critic.format_feedback_for_prompt.return_value = "feedback-text"
    critic.record_feedback_usage = AsyncMock()
    critic.evaluate = AsyncMock()

    service = GapAnalysisServiceV4(mongo, config, llm, rate_limiter, critic=critic)
    response = asyncio.run(
        service.run(
            GapAnalysisRequest(
                policy_document_id="pol-1",
                applicable_jurisdictions=["CA"],
                run_async=False,
                save_results=True,
            )
        )
    )

    assert response.summary.addressed == 1
    assert llm.gap_check_v4.await_args.kwargs["adaptive_feedback"] == "feedback-text"
    assert len(collections[config.compliance_results_collection].inserted) == 1
    persisted_run = collections[config.compliance_results_collection].inserted[0]
    critic.record_feedback_usage.assert_awaited_once_with(
        run_id=persisted_run["_id"],
        feedback_ids=["fb-1"],
        rendered_text="feedback-text",
    )
    critic.evaluate.assert_awaited_once()
    evaluated_response, evaluated_run_id = critic.evaluate.await_args.args
    assert evaluated_response is response
    assert evaluated_run_id == persisted_run["_id"]
