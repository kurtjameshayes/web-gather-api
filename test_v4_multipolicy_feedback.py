"""Regression tests for v4 gap analysis when several policies are analyzed together.

The request list is the source of truth, including when a single policy id is
also present. Adaptive feedback is loaded only for the first listed policy and
is the text injected into every statute comparison for that run.
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from adaptive_feedback_service import CriticService
from compliance_config import load_config
from compliance_suite_schemas import GapAnalysisRequest
from gap_analysis_service_v4 import GapAnalysisServiceV4

FIRST_INSTRUCTION = "Quote the deletion sentence verbatim."
SECOND_INSTRUCTION = "Apply the second policy's retention exception."
DELETION_QUOTE = "We delete personal information when you ask."


def _config():
    config = load_config()
    config.adaptive_feedback_enabled = True
    config.statute_database = ""
    return config


def _mongo(config):
    colls = {
        name: MagicMock(name=name)
        for name in (
            config.policies_collection,
            config.statute_sub_topic_embeddings_collection,
            config.policy_legal_embeddings_collection,
            config.category_mapping_collection,
            config.compliance_results_collection,
            config.compliance_run_log_collection,
        )
    }

    def get_db(db_name: str):
        assert db_name == config.compliance_database
        db = MagicMock()
        db.__getitem__.side_effect = lambda name: colls[name]
        return db

    mongo = MagicMock()
    mongo.__getitem__.side_effect = get_db
    return mongo, colls


def _policy_docs() -> dict[str, dict[str, Any]]:
    return {
        "pol-a": {
            "document_id": "pol-a",
            "text": f"Alpha policy. {DELETION_QUOTE}",
            "company_name": "Alpha Co",
        },
        "pol-b": {
            "document_id": "pol-b",
            "text": "Beta policy retains billing records for one year.",
            "company_name": "Beta Co",
        },
    }


def _wire_corpus(config, colls, policies: dict[str, dict[str, Any]]) -> None:
    loaded: list[str] = []

    def find_policy(query, *args, **kwargs):
        policy_id = query.get(config.policy_document_id_field)
        loaded.append(policy_id)
        return policies.get(policy_id)

    colls[config.policies_collection].find_one.side_effect = find_policy
    colls[config.category_mapping_collection].find.return_value = [
        {
            "statute_category": "consumer_rights",
            "policy_categories": ["deletion"],
            "sub_topic": "erasure",
        }
    ]
    colls[config.statute_sub_topic_embeddings_collection].find.return_value = [
        {
            "_id": "stat-1",
            "header_text": "Right to delete",
            "subtopic_text": "A business shall delete personal information upon request.",
            "jurisdiction": "CA",
            "category": "consumer_rights",
        }
    ]
    colls[config.policy_legal_embeddings_collection].count_documents.return_value = 2
    colls[config.policy_legal_embeddings_collection].find.return_value = [
        {"chunk_text": DELETION_QUOTE},
        {"chunk_text": "Billing records follow the second policy."},
    ]
    colls["_loaded_policy_ids"] = loaded


def _critic(feedback_by_policy: dict[str, list[dict[str, Any]]]) -> MagicMock:
    critic = MagicMock()

    def get_active(policy_id: str, limit: int | None = None):
        if policy_id not in feedback_by_policy:
            raise AssertionError(f"unexpected feedback lookup for {policy_id}")
        return feedback_by_policy[policy_id]

    critic.get_active_feedback.side_effect = get_active
    critic.format_feedback_for_prompt.side_effect = (
        lambda docs: CriticService.format_feedback_for_prompt(critic, docs)
    )
    critic.record_feedback_usage = AsyncMock()
    critic.evaluate = AsyncMock()
    return critic


def _service(config, mongo, critic) -> tuple[GapAnalysisServiceV4, AsyncMock]:
    llm = MagicMock()
    llm.gap_check_v4 = AsyncMock(
        return_value={
            "status": "addressed",
            "policy_quote": DELETION_QUOTE,
            "statute_quote": "A business shall delete personal information upon request.",
            "requirement_summary": "Delete on request",
            "conflict_description": None,
            "confidence": "high",
            "_analysis_failed": False,
        }
    )
    limiter = MagicMock()
    limiter.allow = AsyncMock(return_value=True)
    service = GapAnalysisServiceV4(
        mongo_client=mongo,
        config=config,
        llm_client=llm,
        rate_limiter=limiter,
        critic=critic,
    )
    return service, llm


def _request() -> GapAnalysisRequest:
    return GapAnalysisRequest(
        policy_document_id="pol-ignored",
        policy_document_ids=["pol-a", "pol-b"],
        applicable_jurisdictions=["CA"],
        save_results=True,
        run_async=False,
    )


def test_multipolicy_run_injects_only_the_first_policys_feedback() -> None:
    config = _config()
    mongo, colls = _mongo(config)
    policies = _policy_docs()
    _wire_corpus(config, colls, policies)
    critic = _critic(
        {
            "pol-a": [
                {
                    "_id": "fb-a",
                    "suggestions": [
                        {"instruction": "   "},
                        {"instruction": FIRST_INSTRUCTION},
                    ],
                }
            ],
            "pol-b": [
                {
                    "_id": "fb-b",
                    "suggestions": [{"instruction": SECOND_INSTRUCTION}],
                }
            ],
        }
    )
    service, llm = _service(config, mongo, critic)

    result = asyncio.run(service.run(_request()))

    assert colls["_loaded_policy_ids"] == ["pol-a", "pol-b"]
    assert critic.get_active_feedback.call_args_list[0].args == ("pol-a",)
    assert all(call.args[0] != "pol-b" for call in critic.get_active_feedback.call_args_list)
    assert critic.get_active_feedback.call_count == 1

    llm.gap_check_v4.assert_awaited_once()
    prompt_kwargs = llm.gap_check_v4.await_args.kwargs
    assert FIRST_INSTRUCTION in prompt_kwargs["adaptive_feedback"]
    assert SECOND_INSTRUCTION not in prompt_kwargs["adaptive_feedback"]
    assert "\n-    \n" not in prompt_kwargs["adaptive_feedback"]
    assert prompt_kwargs["adaptive_feedback"].count("\n- ") == 1
    assert prompt_kwargs["policy_text"] == (
        f"{DELETION_QUOTE}\n\n---\n\nBilling records follow the second policy."
    )
    statute_query = colls[config.statute_sub_topic_embeddings_collection].find.call_args.args[0]
    assert statute_query == {"category": "consumer_rights", "sub_topic": "erasure"}
    id_field = config.policy_document_id_field
    legal_query = colls[config.policy_legal_embeddings_collection].find.call_args.args[0]
    assert legal_query[id_field] == {"$in": ["pol-a", "pol-b"]}
    assert legal_query["category"] == {"$in": ["deletion"]}
    indexed_query = colls[config.policy_legal_embeddings_collection].count_documents.call_args.args[0]
    assert indexed_query == {id_field: {"$in": ["pol-a", "pol-b"]}}

    assert result.policy_document_id == "pol-a"
    assert result.policy_document_ids == ["pol-a", "pol-b"]
    assert result.company_name == "Alpha Co"
    assert result.summary.addressed == 1
    assert result.summary.missing == 0
    assert result.gaps[0].policy_quote == DELETION_QUOTE
    assert result.gaps[0].status == "addressed"

    saved = colls[config.compliance_results_collection].insert_one.call_args.args[0]
    assert saved["policy_document_id"] == "pol-a"
    assert saved["policy_document_ids"] == ["pol-a", "pol-b"]
    assert saved["company_name"] == "Alpha Co"
    assert "pol-ignored" not in str(saved)
    log_doc = colls[config.compliance_run_log_collection].insert_one.call_args.args[0]
    assert log_doc["policy_document_id"] == "pol-a"
    assert log_doc["policy_document_ids"] == ["pol-a", "pol-b"]

    critic.record_feedback_usage.assert_awaited_once()
    usage = critic.record_feedback_usage.await_args.kwargs
    assert usage["feedback_ids"] == ["fb-a"]
    assert FIRST_INSTRUCTION in usage["rendered_text"]
    assert SECOND_INSTRUCTION not in usage["rendered_text"]
    assert usage["run_id"] == saved["_id"]
    evaluated = critic.evaluate.await_args.args
    assert evaluated[0].policy_document_id == "pol-a"
    assert evaluated[0].policy_document_ids == ["pol-a", "pol-b"]
    assert evaluated[1] == saved["_id"]


def test_empty_feedback_on_the_first_policy_does_not_read_later_policies() -> None:
    config = _config()
    mongo, colls = _mongo(config)
    _wire_corpus(config, colls, _policy_docs())
    critic = _critic({"pol-a": []})
    service, llm = _service(config, mongo, critic)

    result = asyncio.run(service.run(_request()))

    critic.get_active_feedback.assert_called_once_with("pol-a")
    assert llm.gap_check_v4.await_args.kwargs["adaptive_feedback"] == ""
    critic.record_feedback_usage.assert_not_awaited()
    critic.evaluate.assert_awaited_once()
    assert critic.evaluate.await_args.args[0].policy_document_ids == ["pol-a", "pol-b"]
    assert result.company_name == "Alpha Co"
    assert result.summary.addressed == 1
