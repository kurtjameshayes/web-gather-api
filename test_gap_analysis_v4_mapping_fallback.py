"""V4 category-mapping plural fallback and skip contracts."""
from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock

sys.modules.setdefault("anthropic", MagicMock())
sys.modules.setdefault("sentence_transformers", MagicMock())

from compliance_config import load_config
from compliance_suite_schemas import GapAnalysisRequest
from gap_analysis_service_v4 import ANALYZE_CATEGORIES, GapAnalysisServiceV4


def test_v4_falls_back_to_plural_mappings_and_skips_non_analyze() -> None:
    """Singular category_mapping miss falls back to category_mappings; only ANALYZE rows run."""
    asyncio.run(_run())


async def _run() -> None:
    config = load_config()
    config.adaptive_feedback_enabled = False
    config.category_mapping_collection = "category_mapping"

    policies = MagicMock()
    policy_legal = MagicMock()
    statute_coll = MagicMock()
    cat_map = MagicMock()
    cat_maps = MagicMock()
    results = MagicMock()
    run_log = MagicMock()

    colls = {
        config.policies_collection: policies,
        config.policy_legal_embeddings_collection: policy_legal,
        config.statute_sub_topic_embeddings_collection: statute_coll,
        "category_mapping": cat_map,
        "category_mappings": cat_maps,
        config.compliance_results_collection: results,
        config.compliance_run_log_collection: run_log,
    }

    db = MagicMock()
    db.__getitem__.side_effect = lambda name: colls[name]
    mongo = MagicMock()
    mongo.__getitem__.return_value = db

    policies.find_one.return_value = {
        "document_id": "pol-1",
        "text": "You may request deletion of personal information.",
        "company_name": "Acme",
    }
    policy_legal.count_documents.return_value = 1
    cat_map.find.return_value = []
    cat_maps.find.return_value = [
        {
            "statute_category": "definitions",
            "policy_categories": ["definitions"],
            "sub_topic": "pi",
        },
        {
            "statute_category": "consumer_rights",
            "policy_categories": [],
            "sub_topic": "access",
        },
        {
            "statute_category": "consumer_rights",
            "policy_categories": ["rights"],
            "sub_topic": "delete",
        },
    ]

    def statute_find(filt, *args, **kwargs):
        category = filt.get("category")
        if category == "consumer_rights":
            return [
                {
                    "_id": "s1",
                    "document_id": "stat-1",
                    "header_text": "Right to delete",
                    "subtopic_text": "Must allow deletion",
                    "requirement_summary": "Deletion",
                    "jurisdiction": "CA",
                }
            ]
        if isinstance(category, dict) and set(category.get("$in", [])) == {
            "definitions",
            "applicability",
        }:
            return [
                {
                    "category": "definitions",
                    "header_text": "Definitions",
                    "subtopic_text": "Personal information means data.",
                }
            ]
        return []

    statute_coll.find.side_effect = statute_find
    policy_legal.find.return_value = [{"chunk_text": "You may request deletion of personal information."}]

    llm = MagicMock()
    llm.gap_check_v4 = AsyncMock(
        return_value={
            "status": "addressed",
            "policy_quote": "You may request deletion of personal information.",
            "requirement_summary": "Deletion right",
            "statute_quote": "Must allow deletion",
            "conflict_description": None,
            "confidence": "high",
        }
    )
    limiter = MagicMock()
    limiter.allow = AsyncMock(return_value=True)

    svc = GapAnalysisServiceV4(
        mongo_client=mongo,
        config=config,
        llm_client=llm,
        rate_limiter=limiter,
        critic=None,
    )
    req = GapAnalysisRequest(policy_document_id="pol-1", run_async=False, save_results=True)
    result = await svc.run(req)

    cat_map.find.assert_called()
    assert cat_map.find.call_args[0][0]["statute_category"]["$in"] == list(ANALYZE_CATEGORIES)
    cat_maps.find.assert_called()
    assert llm.gap_check_v4.await_count == 1
    kwargs = llm.gap_check_v4.await_args.kwargs
    assert "Personal information means data." in kwargs["reference_context"]
    assert "Must allow deletion" in kwargs["statutory_requirement"]
    assert result.summary.total_requirements == 1
    assert result.summary.addressed == 1
    results.insert_one.assert_called_once()
    run_log.insert_one.assert_called_once()
