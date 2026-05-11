"""Focused regression tests for GapAnalysisServiceV4 orchestration."""
from __future__ import annotations

import asyncio
import copy
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

# Keep imports deterministic in lightweight test environments.
sys.modules["anthropic"] = MagicMock()

from compliance_config import load_config
from compliance_suite_schemas import GapAnalysisRequest
from gap_analysis_service_v4 import GapAnalysisServiceV4, GapAnalysisServiceV4Error


class FakeCollection:
    def __init__(self, docs: list[dict] | None = None) -> None:
        self.docs = [copy.deepcopy(doc) for doc in (docs or [])]
        self.find_calls: list[tuple[dict, tuple, dict]] = []
        self.find_one_calls: list[dict] = []
        self.count_calls: list[dict] = []
        self.inserted: list[dict] = []

    def find_one(self, query: dict, *args, **kwargs):
        self.find_one_calls.append(copy.deepcopy(query))
        for doc in self.docs:
            if _matches(doc, query):
                return copy.deepcopy(doc)
        return None

    def find(self, query: dict | None = None, *args, **kwargs):
        query = query or {}
        self.find_calls.append((copy.deepcopy(query), args, copy.deepcopy(kwargs)))
        return [copy.deepcopy(doc) for doc in self.docs if _matches(doc, query)]

    def count_documents(self, query: dict) -> int:
        self.count_calls.append(copy.deepcopy(query))
        return sum(1 for doc in self.docs if _matches(doc, query))

    def insert_one(self, doc: dict):
        self.inserted.append(copy.deepcopy(doc))
        return MagicMock(inserted_id=doc.get("_id"))


class FakeDb:
    def __init__(self, collections: dict[str, FakeCollection]) -> None:
        self.collections = collections

    def __getitem__(self, name: str) -> FakeCollection:
        return self.collections[name]


class FakeMongo:
    def __init__(self, databases: dict[str, dict[str, FakeCollection]]) -> None:
        self.databases = {name: FakeDb(collections) for name, collections in databases.items()}

    def __getitem__(self, name: str) -> FakeDb:
        return self.databases[name]


def _matches(doc: dict, query: dict) -> bool:
    for key, expected in query.items():
        actual = doc.get(key)
        if isinstance(expected, dict) and "$in" in expected:
            if actual not in expected["$in"]:
                return False
        elif actual != expected:
            return False
    return True


def _config():
    cfg = load_config()
    cfg.compliance_database = "privacy-db"
    cfg.statute_database = ""
    cfg.default_jurisdictions = ["CA"]
    cfg.category_mapping_collection = "category_mapping"
    return cfg


def _service(mongo: FakeMongo, cfg=None, *, rate_allowed: bool = True):
    cfg = cfg or _config()
    llm = MagicMock()
    llm.gap_check_v4 = AsyncMock(
        return_value={
            "status": "addressed",
            "policy_quote": "This quote is not actually in the policy.",
            "confidence": "high",
            "requirement_summary": "Deletion rights",
            "statute_quote": "Consumers may request deletion.",
        }
    )
    rate_limiter = MagicMock()
    rate_limiter.allow = AsyncMock(return_value=rate_allowed)
    return GapAnalysisServiceV4(mongo, cfg, llm, rate_limiter), llm, rate_limiter


def test_run_uses_plural_category_mapping_fallback_and_downgrades_unbound_addressed_quote():
    cfg = _config()
    collections = {
        cfg.policies_collection: FakeCollection(
            [
                {
                    "document_id": "pol-1",
                    "text": "We let consumers delete data on request.",
                    "company_name": "Acme",
                },
                {
                    "document_id": "pol-2",
                    "policy_chunks": [{"chunk_text": "We honor access requests."}],
                },
            ]
        ),
        cfg.policy_legal_embeddings_collection: FakeCollection(
            [
                {"document_id": "pol-1", "category": "rights", "chunk_text": "Consumers can delete data."},
                {"document_id": "pol-2", "category": "rights", "chunk_text": "Consumers can access data."},
            ]
        ),
        cfg.category_mapping_collection: FakeCollection([]),
        "category_mappings": FakeCollection(
            [
                {
                    "statute_category": "consumer_rights",
                    "policy_categories": ["rights"],
                    "sub_topic": "deletion",
                }
            ]
        ),
        cfg.statute_sub_topic_embeddings_collection: FakeCollection(
            [
                {
                    "_id": "stat-1",
                    "document_id": "ccpa",
                    "category": "consumer_rights",
                    "sub_topic": "deletion",
                    "jurisdiction": "CA",
                    "header_text": "CCPA § 1798.105",
                    "subtopic_text": "Consumers may request deletion.",
                    "requirement_summary": "Deletion rights",
                },
                {
                    "_id": "ctx-1",
                    "document_id": "ccpa",
                    "category": "definitions",
                    "header_text": "Definitions",
                    "subtopic_text": "Consumer means a natural person.",
                },
            ]
        ),
        cfg.compliance_results_collection: FakeCollection([]),
        cfg.compliance_run_log_collection: FakeCollection([]),
    }
    mongo = FakeMongo({cfg.compliance_database: collections})
    service, llm, _rate_limiter = _service(mongo, cfg)

    response = asyncio.run(
        service.run(
            GapAnalysisRequest(
                policy_document_ids=["pol-1", "pol-2"],
                applicable_jurisdictions=["CA"],
                save_results=True,
            )
        )
    )

    assert response.policy_document_id == "pol-1"
    assert response.policy_document_ids == ["pol-1", "pol-2"]
    assert response.company_name == "Acme"
    assert response.summary.total_requirements == 1
    assert response.summary.missing == 1
    assert response.summary.addressed == 0
    assert response.retrieval_metadata.statute_items_considered == 1

    gap = response.gaps[0]
    assert gap.status == "missing"
    assert gap.policy_quote is None
    assert gap.citation_binding_failed is True
    assert gap.conflict_description == (
        "The policy does not contain provisions that address this statutory requirement."
    )

    assert collections[cfg.category_mapping_collection].find_calls == [
        ({"statute_category": {"$in": ["consumer_rights", "controller_duties"]}}, (), {})
    ]
    assert collections["category_mappings"].find_calls == [
        ({"statute_category": {"$in": ["consumer_rights", "controller_duties"]}}, (), {})
    ]
    assert collections[cfg.policy_legal_embeddings_collection].count_calls == [
        {"document_id": {"$in": ["pol-1", "pol-2"]}}
    ]
    assert (
        {
            "document_id": {"$in": ["pol-1", "pol-2"]},
            "category": {"$in": ["rights"]},
        },
        ({"chunk_text": 1},),
        {},
    ) in collections[cfg.policy_legal_embeddings_collection].find_calls

    llm.gap_check_v4.assert_awaited_once()
    llm_kwargs = llm.gap_check_v4.await_args.kwargs
    assert "[definitions]" in llm_kwargs["reference_context"]
    assert llm_kwargs["statutory_requirement"] == "CCPA § 1798.105\n\nConsumers may request deletion."
    assert llm_kwargs["policy_text"] == "Consumers can delete data.\n\n---\n\nConsumers can access data."

    result_doc = collections[cfg.compliance_results_collection].inserted[0]
    assert result_doc["policy_document_id"] == "pol-1"
    assert result_doc["policy_document_ids"] == ["pol-1", "pol-2"]
    assert result_doc["gaps"][0]["status"] == "missing"
    run_log_doc = collections[cfg.compliance_run_log_collection].inserted[0]
    assert run_log_doc["policy_document_ids"] == ["pol-1", "pol-2"]
    assert run_log_doc["statute_item_ids"] == ["stat-1"]


def test_run_rejects_unindexed_policy_before_llm_or_persistence():
    cfg = _config()
    collections = {
        cfg.policies_collection: FakeCollection(
            [{"document_id": "pol-1", "text": "We describe consumer rights."}]
        ),
        cfg.policy_legal_embeddings_collection: FakeCollection([]),
        cfg.category_mapping_collection: FakeCollection([]),
        cfg.statute_sub_topic_embeddings_collection: FakeCollection([]),
        cfg.compliance_results_collection: FakeCollection([]),
        cfg.compliance_run_log_collection: FakeCollection([]),
    }
    mongo = FakeMongo({cfg.compliance_database: collections})
    service, llm, _rate_limiter = _service(mongo, cfg)

    with pytest.raises(GapAnalysisServiceV4Error) as exc_info:
        asyncio.run(
            service.run(
                GapAnalysisRequest(policy_document_id="pol-1", save_results=True)
            )
        )

    assert exc_info.value.status_code == 400
    assert "Policy not indexed" in str(exc_info.value)
    llm.gap_check_v4.assert_not_called()
    assert collections[cfg.compliance_results_collection].inserted == []
    assert collections[cfg.compliance_run_log_collection].inserted == []


def test_run_rate_limit_exceeded_short_circuits_database_work():
    cfg = _config()
    mongo = MagicMock()
    service, llm, rate_limiter = _service(mongo, cfg, rate_allowed=False)

    with pytest.raises(GapAnalysisServiceV4Error) as exc_info:
        asyncio.run(
            service.run(
                GapAnalysisRequest(policy_document_id="pol-1", save_results=False)
            )
        )

    assert exc_info.value.status_code == 429
    assert "Rate limit exceeded" in str(exc_info.value)
    rate_limiter.allow.assert_awaited_once()
    mongo.__getitem__.assert_not_called()
    llm.gap_check_v4.assert_not_called()
