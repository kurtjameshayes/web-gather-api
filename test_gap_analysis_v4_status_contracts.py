"""V4 leftover status, citation, and persist/critic coupling tests.

Open coverage PRs pin addressed-quote demotion, empty-mapping persist,
policy-load guards, and adaptive-feedback injection when results are saved.
They do not pin non-addressed citation flags, analysis_failed summary
accounting, default missing conflict text, or save_results=False skipping
critic evaluation (the path used by async jobs and statute-policy-compliance).
"""
from __future__ import annotations

import asyncio
import sys
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.modules.setdefault("anthropic", MagicMock())
sys.modules.setdefault("sentence_transformers", MagicMock())

from compliance_config import load_config
from compliance_suite_schemas import GapAnalysisRequest
from gap_analysis_service_v4 import GapAnalysisServiceV4


DEFAULT_MISSING_CONFLICT = (
    "The policy does not contain provisions that address this statutory requirement."
)


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _req(**overrides):
    payload = {
        "policy_document_id": "pol-1",
        "save_results": False,
        "run_async": False,
    }
    payload.update(overrides)
    return GapAnalysisRequest.model_validate(payload)


def _service(
    *,
    llm_result: Optional[Dict[str, Any]] = None,
    critic: Any = None,
):
    config = load_config()
    mongo = MagicMock()
    db = MagicMock()
    colls: Dict[str, MagicMock] = {}

    def _coll(name: str):
        if name not in colls:
            colls[name] = MagicMock()
        return colls[name]

    db.__getitem__.side_effect = _coll
    mongo.__getitem__.return_value = db
    for name in (
        config.policies_collection,
        config.statute_sub_topic_embeddings_collection,
        config.policy_legal_embeddings_collection,
        config.category_mapping_collection,
        "category_mappings",
        config.compliance_results_collection,
        config.compliance_run_log_collection,
    ):
        _coll(name)

    limiter = MagicMock()
    limiter.allow = AsyncMock(return_value=True)
    llm = MagicMock()
    llm.gap_check_v4 = AsyncMock(
        return_value=llm_result
        or {
            "status": "missing",
            "policy_quote": None,
            "statute_quote": "Obtain consent.",
            "requirement_summary": "Consent",
            "conflict_description": None,
            "confidence": "high",
            "_analysis_failed": False,
        }
    )
    svc = GapAnalysisServiceV4(
        mongo_client=mongo,
        config=config,
        llm_client=llm,
        rate_limiter=limiter,
        critic=critic,
    )
    return svc, config, colls, llm


def _wire_mapped_run(
    colls: Dict[str, MagicMock],
    config,
    *,
    policy_text: str = "We obtain consent before sharing.",
    statutes: Optional[List[Dict[str, Any]]] = None,
    mappings: Optional[List[Dict[str, Any]]] = None,
) -> None:
    colls[config.policies_collection].find_one.return_value = {
        "document_id": "pol-1",
        "text": policy_text,
        "company_name": "Acme Corp",
    }
    colls[config.policy_legal_embeddings_collection].count_documents.return_value = 1
    colls[config.policy_legal_embeddings_collection].find.return_value = []
    colls[config.category_mapping_collection].find.return_value = mappings or [
        {
            "statute_category": "consumer_rights",
            "policy_categories": ["consent"],
            "sub_topic": "consent",
        }
    ]
    statute_rows = statutes or [
        {
            "_id": "stat-1",
            "document_id": "statute-doc-1",
            "header_text": "Consent",
            "subtopic_text": "Controllers must obtain consent.",
            "requirement_summary": "Obtain consent",
            "jurisdiction": "CA",
        }
    ]

    def _statute_find(query, *args, **kwargs):
        if "document_id" in query and "category" in query:
            return []
        return list(statute_rows)

    colls[config.statute_sub_topic_embeddings_collection].find.side_effect = _statute_find


@pytest.mark.parametrize("status", ["conflict", "partial", "ambiguous"])
def test_v4_unbound_non_addressed_quote_is_flagged_not_demoted(status: str) -> None:
    """Only addressed + unbound quotes are demoted; other statuses keep the quote."""
    svc, config, colls, llm = _service(
        llm_result={
            "status": status,
            "policy_quote": "this phrase is not in the policy",
            "statute_quote": "Obtain consent.",
            "requirement_summary": "Consent",
            "conflict_description": "Policy language is incomplete.",
            "confidence": "high",
            "_analysis_failed": False,
        }
    )
    _wire_mapped_run(colls, config)

    result = _run(svc.run(_req()))
    gap = result.gaps[0]
    assert gap.status == status
    assert gap.citation_binding_failed is True
    assert gap.policy_quote == "this phrase is not in the policy"
    assert getattr(result.summary, status if status != "conflict" else "conflicts") == 1
    assert result.summary.missing == 0


def test_v4_missing_status_gets_default_conflict_description() -> None:
    svc, config, colls, _ = _service(
        llm_result={
            "status": "missing",
            "policy_quote": None,
            "statute_quote": "Obtain consent.",
            "requirement_summary": "Consent",
            "conflict_description": None,
            "confidence": "medium",
            "_analysis_failed": False,
        }
    )
    _wire_mapped_run(colls, config)

    result = _run(svc.run(_req()))
    assert result.gaps[0].status == "missing"
    assert result.gaps[0].conflict_description == DEFAULT_MISSING_CONFLICT
    assert result.summary.missing == 1
    assert result.summary.analysis_failures == 0


def test_v4_analysis_failed_counts_separately_from_missing() -> None:
    svc, config, colls, _ = _service(
        llm_result={
            "status": "missing",
            "policy_quote": None,
            "statute_quote": None,
            "requirement_summary": "Consent",
            "conflict_description": "LLM timed out.",
            "confidence": "low",
            "_analysis_failed": True,
        }
    )
    _wire_mapped_run(colls, config)

    result = _run(svc.run(_req()))
    assert result.gaps[0].analysis_failed is True
    assert result.gaps[0].status == "missing"
    assert result.summary.total_requirements == 1
    assert result.summary.analysis_failures == 1
    assert result.summary.missing == 0


def test_v4_invalid_confidence_is_stored_as_none() -> None:
    svc, config, colls, _ = _service(
        llm_result={
            "status": "partial",
            "policy_quote": "We obtain consent before sharing.",
            "statute_quote": "Obtain consent.",
            "requirement_summary": "Consent",
            "conflict_description": None,
            "confidence": "EXTREME",
            "_analysis_failed": False,
        }
    )
    _wire_mapped_run(colls, config)

    result = _run(svc.run(_req()))
    assert result.gaps[0].status == "partial"
    assert result.gaps[0].confidence is None


def test_v4_duplicate_statute_keys_are_skipped() -> None:
    svc, config, colls, llm = _service()
    _wire_mapped_run(
        colls,
        config,
        mappings=[
            {
                "statute_category": "consumer_rights",
                "policy_categories": ["consent"],
                "sub_topic": "consent",
            },
            {
                "statute_category": "consumer_rights",
                "policy_categories": ["notice"],
                "sub_topic": "consent",
            },
        ],
    )

    result = _run(svc.run(_req()))
    assert len(result.gaps) == 1
    assert llm.gap_check_v4.await_count == 1
    assert result.statute_chunk_ids_used == ["stat-1"]


def test_v4_save_results_false_injects_feedback_but_skips_persist_and_critic() -> None:
    """Async jobs and statute-policy force save_results=False, which also skips critic."""
    critic = MagicMock()
    critic.get_active_feedback.return_value = [
        {"_id": "fb-1", "suggestions": [{"instruction": "Require verbatim quotes"}]}
    ]
    critic.format_feedback_for_prompt.return_value = (
        "PRIOR ANALYSIS FEEDBACK (incorporate these lessons into your analysis):\n\n"
        "- Require verbatim quotes"
    )
    critic.record_feedback_usage = AsyncMock()
    critic.evaluate = AsyncMock(return_value="fb-2")

    svc, config, colls, llm = _service(critic=critic)
    svc._cfg_enabled = True
    config.adaptive_feedback_enabled = True
    _wire_mapped_run(colls, config)

    result = _run(svc.run(_req(save_results=False)))
    assert result.gaps
    llm.gap_check_v4.assert_awaited()
    assert "Require verbatim quotes" in llm.gap_check_v4.await_args.kwargs["adaptive_feedback"]
    critic.record_feedback_usage.assert_not_awaited()
    critic.evaluate.assert_not_awaited()
    colls[config.compliance_results_collection].insert_one.assert_not_called()
    colls[config.compliance_run_log_collection].insert_one.assert_not_called()
