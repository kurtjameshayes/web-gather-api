"""Jurisdiction fallback for v3 and v4 gap analysis.

Empty configured defaults become ``["CA"]``. A request list that only contains
a blank string is truthy, so it blocks that default. v4 then labels a statute
row with no usable jurisdiction as the first resolved value. v3 leaves the gap
label blank and, when the request entry expands to nothing, scans the
California constant instead of the configured default.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

from compliance_config import load_config
from compliance_suite_schemas import GapAnalysisRequest
from compliance_utils import jurisdiction_filter_values, normalize_jurisdiction
from gap_analysis_service_v3 import VECTOR_SEARCH_JURISDICTION, GapAnalysisServiceV3
from gap_analysis_service_v4 import GapAnalysisServiceV4
from rate_limiter import RateLimiter


def _config(defaults: Optional[List[str]]):
    config = load_config()
    config.default_jurisdictions = defaults
    config.statute_database = ""
    config.adaptive_feedback_enabled = False
    return config


def _mongo(specs: Dict[str, Dict[str, Any]]) -> MagicMock:
    mongo = MagicMock()
    databases: Dict[str, MagicMock] = {}

    def get_db(name: str) -> MagicMock:
        if name not in databases:
            db = MagicMock()
            collections: Dict[str, MagicMock] = {}

            def get_coll(coll_name: str) -> MagicMock:
                if coll_name not in collections:
                    coll = MagicMock()
                    spec = specs.get(coll_name, {})
                    if "find_one" in spec:
                        coll.find_one.return_value = spec["find_one"]
                    if "find" in spec:
                        find_spec = spec["find"]
                        if callable(find_spec):
                            coll.find.side_effect = find_spec
                        else:
                            coll.find.return_value = find_spec
                    if "count" in spec:
                        coll.count_documents.return_value = spec["count"]
                    if "aggregate" in spec:
                        coll.aggregate.return_value = spec["aggregate"]
                    collections[coll_name] = coll
                return collections[coll_name]

            db.__getitem__.side_effect = get_coll
            databases[name] = db
        return databases[name]

    mongo.__getitem__.side_effect = get_db
    return mongo


def _policy_doc() -> Dict[str, Any]:
    return {
        "document_id": "pol-1",
        "text": "We retain records.",
        "company_name": "Acme",
    }


def _v4(defaults: Optional[List[str]], specs: Dict[str, Dict[str, Any]]) -> GapAnalysisServiceV4:
    llm = MagicMock()
    llm.gap_check_v4 = AsyncMock(
        return_value={
            "status": "addressed",
            "policy_quote": "We retain records.",
            "confidence": "high",
            "requirement_summary": "Retain records",
            "conflict_description": "none",
            "statute_quote": "Keep records.",
        }
    )
    return GapAnalysisServiceV4(
        mongo_client=_mongo(specs),
        config=_config(defaults),
        llm_client=llm,
        rate_limiter=RateLimiter(60),
        critic=None,
    )


def _v4_specs(statute_docs: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Dict[str, Any]]:
    def statute_find(query: Dict[str, Any], projection: Any = None):
        category = query.get("category")
        if isinstance(category, dict) and "definitions" in category.get("$in", []):
            return []
        return statute_docs or []

    return {
        "policies": {"find_one": _policy_doc()},
        "policy_legal_embeddings": {
            "count": 1,
            "find": [{"chunk_text": "We retain records."}],
        },
        "category_mapping": {
            "find": [
                {
                    "statute_category": "consumer_rights",
                    "policy_categories": ["retention"],
                }
            ]
        },
        "statute_sub_topic_embeddings": {"find": statute_find},
    }


def _statute_row(row_id: str, jurisdiction: Any = ...) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "_id": row_id,
        "header_text": "Sec",
        "subtopic_text": f"Requirement {row_id}",
        "document_id": "stat-1",
        "category": "consumer_rights",
    }
    if jurisdiction is not ...:
        row["jurisdiction"] = jurisdiction
    return row


def test_v4_empty_defaults_label_unlabeled_statutes_ca() -> None:
    rows = [
        _statute_row("missing"),
        _statute_row("none", None),
        _statute_row("zero", 0),
        _statute_row("false", False),
        _statute_row("blank", ""),
        _statute_row("texas", "TX"),
    ]
    service = _v4([], _v4_specs(rows))
    result = asyncio.run(
        service.run(
            GapAnalysisRequest(policy_document_id="pol-1", save_results=False)
        )
    )

    assert result.applicable_jurisdictions == ["CA"]
    assert [gap.jurisdiction for gap in result.gaps] == [
        "CA",
        "CA",
        "CA",
        "CA",
        "CA",
        "TX",
    ]
    assert service.llm.gap_check_v4.await_count == 6


def test_v4_request_list_wins_over_configured_default() -> None:
    rows = [_statute_row("missing"), _statute_row("texas", "TX")]
    service = _v4(["CA"], _v4_specs(rows))
    result = asyncio.run(
        service.run(
            GapAnalysisRequest(
                policy_document_id="pol-1",
                applicable_jurisdictions=["VA", "CO"],
                save_results=False,
            )
        )
    )

    assert result.applicable_jurisdictions == ["VA", "CO"]
    assert [gap.jurisdiction for gap in result.gaps] == ["VA", "TX"]


def test_v4_blank_request_entry_blocks_ca_default() -> None:
    service = _v4(["VA"], _v4_specs([_statute_row("missing")]))
    result = asyncio.run(
        service.run(
            GapAnalysisRequest(
                policy_document_id="pol-1",
                applicable_jurisdictions=[""],
                save_results=False,
            )
        )
    )

    assert result.applicable_jurisdictions == [""]
    assert result.gaps[0].jurisdiction == ""


def test_v4_empty_mappings_still_report_ca_when_defaults_blank() -> None:
    specs = _v4_specs()
    specs["category_mapping"] = {"find": []}
    specs["category_mappings"] = {"find": []}
    service = _v4([], specs)
    result = asyncio.run(
        service.run(
            GapAnalysisRequest(policy_document_id="pol-1", save_results=False)
        )
    )

    assert result.applicable_jurisdictions == ["CA"]
    assert result.gaps == []
    service.llm.gap_check_v4.assert_not_called()


def _v3(defaults: Optional[List[str]], statute_docs: List[Dict[str, Any]]) -> GapAnalysisServiceV3:
    llm = MagicMock()
    llm.gap_check_v3 = AsyncMock(
        return_value={
            "status": "missing",
            "policy_quote": None,
            "confidence": "low",
            "requirement_summary": "Retain records",
            "conflict_description": "none",
            "statute_quote": "Keep records.",
        }
    )
    specs = {
        "policies": {"find_one": _policy_doc()},
        "policy_sub_embeddings": {"count": 1, "aggregate": []},
        "statute_sub_embeddings": {"find": statute_docs},
    }
    return GapAnalysisServiceV3(
        mongo_client=_mongo(specs),
        config=_config(defaults),
        retriever=MagicMock(),
        llm_client=llm,
        storage=MagicMock(),
        rate_limiter=RateLimiter(60),
    )


def _v3_statute_filter(service: GapAnalysisServiceV3) -> Any:
    coll = service.mongo[service.db][service.cfg.statute_sub_embeddings_collection]
    return coll.find.call_args[0][0][service.cfg.statute_jurisdiction_field]


def test_v3_empty_defaults_query_ca_but_leave_gap_label_blank() -> None:
    service = _v3(
        [],
            [
                {"_id": "missing", "embedding": [0.2, 0.4], "sub_chunk_text": "Keep records."},
                {
                    "_id": "texas",
                    "embedding": [0.2, 0.4],
                    "sub_chunk_text": "Keep records.",
                    "jurisdiction": "TX",
                },
            ],
    )
    result = asyncio.run(
        service.run(
            GapAnalysisRequest(policy_document_id="pol-1", save_results=False)
        )
    )

    assert result.applicable_jurisdictions == ["CA"]
    assert [gap.jurisdiction for gap in result.gaps] == ["", "TX"]
    expected = jurisdiction_filter_values(normalize_jurisdiction("CA"))
    assert _v3_statute_filter(service) == {"$in": expected}
    assert "VA" not in expected


def test_v3_blank_request_entry_scans_california_constant() -> None:
    service = _v3(["VA"], [])
    result = asyncio.run(
        service.run(
            GapAnalysisRequest(
                policy_document_id="pol-1",
                applicable_jurisdictions=[""],
                save_results=False,
            )
        )
    )

    assert result.applicable_jurisdictions == [""]
    assert result.gaps == []
    expected = jurisdiction_filter_values(VECTOR_SEARCH_JURISDICTION)
    assert expected == ["California"]
    assert _v3_statute_filter(service) == "California"
    service.llm.gap_check_v3.assert_not_called()
