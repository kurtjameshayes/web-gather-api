"""Regression tests for v4 empty-mapping vs persist contracts.

When category_mapping (and the plural fallback) have no ANALYZE_CATEGORIES
rows, run() returns an empty 200-equivalent response and does not persist.
Mappings that exist but are skipped (empty policy_categories) still persist.
A persist exception must not fail a completed analysis.
"""
from __future__ import annotations

import asyncio
import sys
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

sys.modules.setdefault("anthropic", MagicMock())
sys.modules.setdefault("sentence_transformers", MagicMock())

from compliance_config import load_config
from compliance_suite_schemas import GapAnalysisRequest
from gap_analysis_service_v4 import GapAnalysisServiceV4
from rate_limiter import RateLimiter


def _run(coro):
    return asyncio.run(coro)


def _build_service(collections: Dict[str, MagicMock]) -> GapAnalysisServiceV4:
    config = load_config()
    config.adaptive_feedback_enabled = False
    config.auth_required = False
    config.statute_database = ""

    db = MagicMock()
    db.__getitem__.side_effect = lambda name: collections[name]
    mongo = MagicMock()
    mongo.__getitem__.return_value = db

    llm = MagicMock()
    llm.gap_check_v4 = AsyncMock(side_effect=AssertionError("LLM should not be called"))
    limiter = MagicMock(spec=RateLimiter)
    limiter.allow = AsyncMock(return_value=True)

    return GapAnalysisServiceV4(
        mongo_client=mongo,
        config=config,
        llm_client=llm,
        rate_limiter=limiter,
        critic=None,
    )


def _indexed_policy_collections() -> Dict[str, MagicMock]:
    policies = MagicMock(name="policies")
    policies.find_one.return_value = {
        "document_id": "pol-1",
        "text": "Consumers may request deletion of personal data.",
        "company_name": "Acme",
    }

    policy_legal = MagicMock(name="policy_legal_embeddings")
    policy_legal.count_documents.return_value = 2

    primary_map = MagicMock(name="category_mapping")
    fallback_map = MagicMock(name="category_mappings")
    statutes = MagicMock(name="statute_sub_topic_embeddings")
    results = MagicMock(name="compliance_results")
    run_log = MagicMock(name="compliance_run_log")

    return {
        "policies": policies,
        "policy_legal_embeddings": policy_legal,
        "category_mapping": primary_map,
        "category_mappings": fallback_map,
        "statute_sub_topic_embeddings": statutes,
        "compliance_results": results,
        "compliance_run_log": run_log,
    }


def test_v4_empty_mappings_returns_empty_gaps_without_persisting() -> None:
    """No mappings in singular or plural collection: empty gaps, no writes."""
    colls = _indexed_policy_collections()
    colls["category_mapping"].find.return_value = []
    colls["category_mappings"].find.return_value = []
    service = _build_service(colls)

    response = _run(
        service.run(
            GapAnalysisRequest(
                policy_document_id="pol-1",
                applicable_jurisdictions=["CA"],
                run_async=False,
            )
        )
    )

    assert response.gaps == []
    assert response.summary.total_requirements == 0
    assert response.retrieval_metadata.statute_items_considered == 0
    assert response.run_type == "gap_analysis_v4"
    assert response.company_name == "Acme"
    assert response.applicable_jurisdictions == ["CA"]
    colls["category_mapping"].find.assert_called_once()
    colls["category_mappings"].find.assert_called_once()
    colls["statute_sub_topic_embeddings"].find.assert_not_called()
    colls["compliance_results"].insert_one.assert_not_called()
    colls["compliance_run_log"].insert_one.assert_not_called()
    service.llm.gap_check_v4.assert_not_called()


def test_v4_skipped_mappings_still_persist_empty_result() -> None:
    """Mappings with empty policy_categories are skipped but the empty run is saved."""
    colls = _indexed_policy_collections()
    colls["category_mapping"].find.return_value = [
        {"statute_category": "consumer_rights", "policy_categories": [], "sub_topic": "right_to_delete"},
        {"statute_category": "controller_duties", "policy_categories": None, "sub_topic": "privacy_notice"},
    ]
    service = _build_service(colls)

    response = _run(
        service.run(
            GapAnalysisRequest(
                policy_document_id="pol-1",
                applicable_jurisdictions=["CA"],
                run_async=False,
            )
        )
    )

    assert response.gaps == []
    assert response.summary.total_requirements == 0
    colls["category_mappings"].find.assert_not_called()
    colls["statute_sub_topic_embeddings"].find.assert_not_called()
    colls["compliance_results"].insert_one.assert_called_once()
    persisted = colls["compliance_results"].insert_one.call_args.args[0]
    assert persisted["gaps"] == []
    assert persisted["run_type"] == "gap_analysis_v4"
    assert persisted["policy_document_id"] == "pol-1"
    colls["compliance_run_log"].insert_one.assert_called_once()
    service.llm.gap_check_v4.assert_not_called()


def test_v4_persist_failure_does_not_fail_completed_empty_run() -> None:
    """A results-collection write error is non-fatal after mappings are skipped."""
    colls = _indexed_policy_collections()
    colls["category_mapping"].find.return_value = [
        {"statute_category": "consumer_rights", "policy_categories": []},
    ]
    colls["compliance_results"].insert_one.side_effect = RuntimeError("write concern")
    service = _build_service(colls)

    response = _run(
        service.run(
            GapAnalysisRequest(
                policy_document_id="pol-1",
                applicable_jurisdictions=["VA"],
                run_async=False,
            )
        )
    )

    assert response.gaps == []
    assert response.policy_document_id == "pol-1"
    assert response.applicable_jurisdictions == ["VA"]
    colls["compliance_results"].insert_one.assert_called_once()
