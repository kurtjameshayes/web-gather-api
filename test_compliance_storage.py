"""Regression tests for compliance result and alert storage queries."""
from __future__ import annotations

import asyncio
import re
from typing import Any, Dict, List
from unittest.mock import MagicMock, call, patch

from compliance_config import load_config
from compliance_storage import ComplianceStorage


async def _sync_run(func, *args, **kwargs):
    """Run storage thread-pool callbacks inline for deterministic assertions."""
    return func(*args, **kwargs)


def _storage_with_collection() -> tuple[ComplianceStorage, Any, MagicMock, MagicMock]:
    config = load_config()
    mongo_client = MagicMock()
    db = MagicMock()
    collection = MagicMock()
    mongo_client.__getitem__.return_value = db
    db.__getitem__.return_value = collection
    return ComplianceStorage(mongo_client, config), config, db, collection


def _cursor_returning(docs: List[Dict[str, Any]]) -> MagicMock:
    cursor = MagicMock()
    cursor.sort.return_value = cursor
    cursor.skip.return_value = cursor
    cursor.limit.return_value = docs
    return cursor


def test_list_runs_builds_filters_and_infers_run_types() -> None:
    storage, config, db, collection = _storage_with_collection()
    docs = [
        {
            "_id": "run-1",
            "policy_document_id": "policy-1",
            "company_name": "Example Co",
            "analyzed_at": "2026-03-21T10:00:00+00:00",
            "gaps": [],
            "privacy_health_score": 94,
            "summary": {"missing": 0},
        },
        {
            "_id": "run-2",
            "policy_document_id": "policy-1",
            "analyzed_at": "2026-03-20T10:00:00+00:00",
            "applicable_jurisdictions": ["CA"],
        },
    ]
    cursor = _cursor_returning(docs)
    collection.count_documents.return_value = 2
    collection.find.return_value = cursor

    with patch("compliance_storage.run_in_thread", side_effect=_sync_run):
        summaries, total = asyncio.run(
            storage.list_runs(
                policy_document_id="policy-1",
                since="2026-03-01T00:00:00+00:00",
                until="2026-03-31T23:59:59+00:00",
                types=["gap", "health_score"],
                limit=25,
                offset=5,
            )
        )

    expected_query = {
        "policy_document_id": "policy-1",
        "run_types": {"$in": ["gap", "health_score"]},
        "analyzed_at": {
            "$gte": "2026-03-01T00:00:00+00:00",
            "$lte": "2026-03-31T23:59:59+00:00",
        },
    }
    db.__getitem__.assert_has_calls([call(config.compliance_results_collection)])
    collection.count_documents.assert_called_once_with(expected_query)
    collection.find.assert_called_once_with(expected_query)
    cursor.sort.assert_called_once_with("analyzed_at", -1)
    cursor.skip.assert_called_once_with(5)
    cursor.limit.assert_called_once_with(25)
    assert total == 2
    assert summaries == [
        {
            "run_id": "run-1",
            "policy_document_id": "policy-1",
            "company_name": "Example Co",
            "run_at": "2026-03-21T10:00:00+00:00",
            "types": ["gap", "health_score"],
            "privacy_health_score": 94,
            "score_assessment": None,
            "summary": {"missing": 0},
        },
        {
            "run_id": "run-2",
            "policy_document_id": "policy-1",
            "company_name": None,
            "run_at": "2026-03-20T10:00:00+00:00",
            "types": ["applicability"],
            "privacy_health_score": None,
            "score_assessment": None,
            "summary": None,
        },
    ]


def test_list_alerts_escapes_company_filter_and_strips_mongo_ids() -> None:
    storage, config, db, collection = _storage_with_collection()
    docs = [
        {
            "_id": "mongo-object-id",
            "alert_id": "alert-1",
            "policy_document_id": "policy-1",
            "company_name": "Acme.* (CA)",
            "affected_jurisdictions": ["CA"],
            "detected_at": "2026-03-21T10:00:00+00:00",
        }
    ]
    cursor = _cursor_returning(docs)
    collection.count_documents.return_value = 1
    collection.find.return_value = cursor

    with patch("compliance_storage.run_in_thread", side_effect=_sync_run):
        alerts, total = asyncio.run(
            storage.list_alerts(
                policy_document_id="policy-1",
                company_name="Acme.* (CA)",
                jurisdiction="CA",
                since="2026-03-01T00:00:00+00:00",
                limit=10,
                offset=3,
            )
        )

    expected_query = {
        "policy_document_id": "policy-1",
        "company_name": {"$regex": re.escape("Acme.* (CA)"), "$options": "i"},
        "affected_jurisdictions": "CA",
        "detected_at": {"$gte": "2026-03-01T00:00:00+00:00"},
    }
    db.__getitem__.assert_has_calls([call(config.compliance_alerts_collection)])
    collection.count_documents.assert_called_once_with(expected_query)
    collection.find.assert_called_once_with(expected_query)
    cursor.sort.assert_called_once_with("detected_at", -1)
    cursor.skip.assert_called_once_with(3)
    cursor.limit.assert_called_once_with(10)
    assert total == 1
    assert alerts == [
        {
            "alert_id": "alert-1",
            "policy_document_id": "policy-1",
            "company_name": "Acme.* (CA)",
            "affected_jurisdictions": ["CA"],
            "detected_at": "2026-03-21T10:00:00+00:00",
        }
    ]
