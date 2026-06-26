"""Regression tests for the V4 gap analysis service orchestration."""
from __future__ import annotations

import asyncio
from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock

from compliance_config import load_config
from compliance_suite_schemas import GapAnalysisRequest
from gap_analysis_service_v4 import GapAnalysisServiceV4, _citation_binding


class FakeCollection:
    def __init__(self, docs=None):
        self.docs = [deepcopy(doc) for doc in docs or []]
        self.inserted = []

    def find_one(self, query):
        for doc in self.docs:
            if _matches(doc, query):
                return deepcopy(doc)
        return None

    def find(self, query=None, projection=None):
        return [_project(doc, projection) for doc in self.docs if _matches(doc, query or {})]

    def count_documents(self, query):
        return sum(1 for doc in self.docs if _matches(doc, query))

    def insert_one(self, doc):
        stored = deepcopy(doc)
        self.inserted.append(stored)
        self.docs.append(stored)
        return MagicMock(inserted_id=stored.get("_id"))


class FakeDatabase:
    def __init__(self, collections):
        self.collections = collections

    def __getitem__(self, name):
        return self.collections.setdefault(name, FakeCollection())


class FakeMongo:
    def __init__(self, databases):
        self.databases = databases

    def __getitem__(self, name):
        return self.databases[name]


def test_citation_binding_normalizes_whitespace_and_case():
    assert _citation_binding("We collect personal information.", "we  collect\nPERSONAL information.")
    assert not _citation_binding("fabricated quote", "We collect personal information.")
    assert not _citation_binding("", "We collect personal information.")


def test_run_downgrades_addressed_gap_when_policy_quote_is_not_grounded():
    service, _, llm, _ = _build_service(
        llm_result={
            "status": "addressed",
            "policy_quote": "Users may request a portable archive.",
            "confidence": "high",
            "requirement_summary": "Disclose collected personal information",
            "statute_quote": "Businesses must disclose categories collected.",
        }
    )

    result = asyncio.run(
        service.run(
            GapAnalysisRequest(
                policy_document_id="pol-1",
                applicable_jurisdictions=["CA"],
                save_results=False,
            )
        )
    )

    assert result.summary.total_requirements == 1
    assert result.summary.addressed == 0
    assert result.summary.missing == 1
    gap = result.gaps[0]
    assert gap.status == "missing"
    assert gap.policy_quote is None
    assert gap.citation_binding_failed is True
    assert "does not contain provisions" in gap.conflict_description
    llm.gap_check_v4.assert_awaited_once()


def test_run_injects_adaptive_feedback_and_triggers_critic_after_persisting():
    critic = MagicMock()
    critic.get_active_feedback.return_value = [
        {"_id": "fb-old", "suggestions": [{"instruction": "Require verbatim policy quotes."}]}
    ]
    critic.format_feedback_for_prompt.return_value = (
        "PRIOR ANALYSIS FEEDBACK\n- Require verbatim policy quotes."
    )
    critic.record_feedback_usage = AsyncMock()
    critic.evaluate = AsyncMock(return_value="fb-new")
    service, mongo, llm, cfg = _build_service(
        critic=critic,
        llm_result={
            "status": "addressed",
            "policy_quote": "We collect personal information.",
            "confidence": "high",
            "requirement_summary": "Disclose collected personal information",
            "statute_quote": "Businesses must disclose categories collected.",
        },
    )

    result = asyncio.run(
        service.run(
            GapAnalysisRequest(
                policy_document_id="pol-1",
                applicable_jurisdictions=["CA"],
                save_results=True,
            )
        )
    )

    call_kwargs = llm.gap_check_v4.await_args.kwargs
    assert call_kwargs["adaptive_feedback"] == (
        "PRIOR ANALYSIS FEEDBACK\n- Require verbatim policy quotes."
    )
    assert result.summary.addressed == 1

    results_coll = mongo[cfg.compliance_database][cfg.compliance_results_collection]
    assert len(results_coll.inserted) == 1
    run_id = results_coll.inserted[0]["_id"]
    critic.record_feedback_usage.assert_awaited_once_with(
        run_id=run_id,
        feedback_ids=["fb-old"],
        rendered_text="PRIOR ANALYSIS FEEDBACK\n- Require verbatim policy quotes.",
    )
    critic.evaluate.assert_awaited_once()
    evaluated_response, evaluated_run_id = critic.evaluate.await_args.args
    assert evaluated_response is result
    assert evaluated_run_id == run_id


def _build_service(llm_result, critic=None):
    cfg = load_config()
    cfg.auth_required = False
    cfg.adaptive_feedback_enabled = True
    cfg.default_jurisdictions = ["CA"]

    compliance_db = FakeDatabase(
        {
            cfg.policies_collection: FakeCollection(
                [
                    {
                        "document_id": "pol-1",
                        "company_name": "Acme",
                        "text": "We collect personal information.",
                    }
                ]
            ),
            cfg.policy_legal_embeddings_collection: FakeCollection(
                [
                    {
                        "document_id": "pol-1",
                        "category": "notice",
                        "chunk_text": "We collect personal information.",
                    }
                ]
            ),
            cfg.statute_sub_topic_embeddings_collection: FakeCollection(
                [
                    {
                        "_id": "stat-1",
                        "document_id": "ccpa",
                        "category": "consumer_rights",
                        "sub_topic": "right_to_know",
                        "header_text": "CCPA Section 1798.100",
                        "subtopic_text": "Businesses must disclose categories collected.",
                        "requirement_summary": "Right to know disclosures",
                        "jurisdiction": "CA",
                    },
                    {
                        "_id": "ctx-1",
                        "document_id": "ccpa",
                        "category": "definitions",
                        "header_text": "Definitions",
                        "subtopic_text": "Personal information has the statutory meaning.",
                    },
                ]
            ),
            cfg.category_mapping_collection: FakeCollection(
                [
                    {
                        "statute_category": "consumer_rights",
                        "sub_topic": "right_to_know",
                        "policy_categories": ["notice"],
                    }
                ]
            ),
            cfg.compliance_results_collection: FakeCollection(),
            cfg.compliance_run_log_collection: FakeCollection(),
        }
    )
    mongo = FakeMongo({cfg.compliance_database: compliance_db})
    llm = MagicMock()
    llm.gap_check_v4 = AsyncMock(return_value=llm_result)
    rate_limiter = MagicMock()
    rate_limiter.allow = AsyncMock(return_value=True)
    service = GapAnalysisServiceV4(mongo, cfg, llm, rate_limiter, critic=critic)
    return service, mongo, llm, cfg


def _matches(doc, query):
    for key, expected in query.items():
        value = doc.get(key)
        if isinstance(expected, dict):
            if "$in" in expected:
                allowed = expected["$in"]
                if isinstance(value, list):
                    if not any(item in allowed for item in value):
                        return False
                elif value not in allowed:
                    return False
            else:
                return False
        elif value != expected:
            return False
    return True


def _project(doc, projection):
    if not projection:
        return deepcopy(doc)
    projected = {}
    for key, include in projection.items():
        if include and key in doc:
            projected[key] = deepcopy(doc[key])
    return projected
