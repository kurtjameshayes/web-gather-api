"""Leftover list_runs type-inference contracts.

Open storage coverage pins gap+health_score inference and applicability-only
rows. These tests pin the remaining branches operators actually rely on when
legacy results omit `run_types`.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

from compliance_config import load_config
from compliance_storage import ComplianceStorage


async def _sync_run(func, *args, **kwargs):
    return func(*args, **kwargs)


def _storage_with_collection() -> tuple[ComplianceStorage, MagicMock]:
    config = load_config()
    mongo_client = MagicMock()
    db = MagicMock()
    collection = MagicMock()
    mongo_client.__getitem__.return_value = db
    db.__getitem__.return_value = collection
    return ComplianceStorage(mongo_client, config), collection


def _cursor_returning(docs: List[Dict[str, Any]]) -> MagicMock:
    cursor = MagicMock()
    cursor.sort.return_value = cursor
    cursor.skip.return_value = cursor
    cursor.limit.return_value = docs
    return cursor


def test_list_runs_infers_multi_jurisdictional_from_strictest_denominator() -> None:
    """A stored strictest-common-denominator result is typed multi_jurisdictional."""
    storage, collection = _storage_with_collection()
    collection.count_documents.return_value = 1
    collection.find.return_value = _cursor_returning(
        [
            {
                "_id": "run-mj",
                "policy_document_id": "policy-1",
                "analyzed_at": "2026-09-17T10:00:00+00:00",
                "strictest_common_denominator": [{"requirement": "opt-out"}],
            }
        ]
    )

    with patch("compliance_storage.run_in_thread", side_effect=_sync_run):
        summaries, total = asyncio.run(storage.list_runs())

    assert total == 1
    assert summaries[0]["types"] == ["multi_jurisdictional"]
    assert summaries[0]["run_id"] == "run-mj"


def test_list_runs_skips_inference_when_run_types_present() -> None:
    """An explicit run_types list is the source of truth even if other fields exist."""
    storage, collection = _storage_with_collection()
    collection.count_documents.return_value = 1
    collection.find.return_value = _cursor_returning(
        [
            {
                "_id": "run-explicit",
                "policy_document_id": "policy-1",
                "analyzed_at": "2026-09-17T10:00:00+00:00",
                "run_types": ["gap_v4"],
                "gaps": [],
                "privacy_health_score": 91,
                "strictest_common_denominator": [],
                "applicable_jurisdictions": ["CA"],
            }
        ]
    )

    with patch("compliance_storage.run_in_thread", side_effect=_sync_run):
        summaries, _ = asyncio.run(storage.list_runs())

    assert summaries[0]["types"] == ["gap_v4"]


def test_list_runs_empty_run_types_still_infers() -> None:
    """Empty run_types is falsy (`or []`) so inference still runs."""
    storage, collection = _storage_with_collection()
    collection.count_documents.return_value = 1
    collection.find.return_value = _cursor_returning(
        [
            {
                "_id": "run-empty-types",
                "policy_document_id": "policy-1",
                "analyzed_at": "2026-09-17T10:00:00+00:00",
                "run_types": [],
                "gaps": [{"statute_id": "s-1"}],
            }
        ]
    )

    with patch("compliance_storage.run_in_thread", side_effect=_sync_run):
        summaries, _ = asyncio.run(storage.list_runs())

    assert summaries[0]["types"] == ["gap"]


def test_list_runs_does_not_add_applicability_when_other_types_inferred() -> None:
    """applicable_jurisdictions is only used when no other type was inferred."""
    storage, collection = _storage_with_collection()
    collection.count_documents.return_value = 1
    collection.find.return_value = _cursor_returning(
        [
            {
                "_id": "run-gap-and-jurisdictions",
                "policy_document_id": "policy-1",
                "analyzed_at": "2026-09-17T10:00:00+00:00",
                "gaps": [],
                "applicable_jurisdictions": ["CA", "VA"],
            }
        ]
    )

    with patch("compliance_storage.run_in_thread", side_effect=_sync_run):
        summaries, _ = asyncio.run(storage.list_runs())

    assert summaries[0]["types"] == ["gap"]
