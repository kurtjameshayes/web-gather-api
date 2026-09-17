"""Caller-supplied compliance result ids and last-result lookup filters.

Open storage coverage pins UUID minting when `_id` is omitted. Drift-check and
report load the latest result by policy id; overwriting a caller `_id` or
adding a statute-index filter when none was requested would hide stored runs.
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

from compliance_config import load_config
from compliance_storage import ComplianceStorage


async def _sync_run(func, *args, **kwargs):
    return func(*args, **kwargs)


def _storage_with_results() -> tuple[ComplianceStorage, MagicMock]:
    config = load_config()
    mongo_client = MagicMock()
    db = MagicMock()
    results = MagicMock()
    mongo_client.__getitem__.return_value = db
    db.__getitem__.return_value = results
    return ComplianceStorage(mongo_client, config), results


def test_write_compliance_result_preserves_caller_id_and_timestamp() -> None:
    """Provided `_id` and `analyzed_at` must not be replaced on insert."""
    storage, results = _storage_with_results()
    doc = {
        "_id": "caller-run-id",
        "policy_document_id": "policy-1",
        "analyzed_at": "2026-01-01T00:00:00+00:00",
    }

    with patch("compliance_storage.run_in_thread", side_effect=_sync_run):
        inserted_id = asyncio.run(storage.write_compliance_result(doc))

    assert inserted_id == "caller-run-id"
    results.insert_one.assert_called_once_with(doc)
    stored = results.insert_one.call_args[0][0]
    assert stored["_id"] == "caller-run-id"
    assert stored["analyzed_at"] == "2026-01-01T00:00:00+00:00"


def test_get_last_compliance_result_omits_version_filter_when_unspecified() -> None:
    """Without statute_index_version the query is policy id only, newest first."""
    storage, results = _storage_with_results()
    cursor = MagicMock()
    cursor.sort.return_value = cursor
    cursor.limit.return_value = iter(
        [{"_id": "abc", "policy_document_id": "policy-1"}]
    )
    results.find.return_value = cursor

    with patch("compliance_storage.run_in_thread", side_effect=_sync_run):
        doc = asyncio.run(storage.get_last_compliance_result("policy-1"))

    results.find.assert_called_once_with({"policy_document_id": "policy-1"})
    cursor.sort.assert_called_once_with("analyzed_at", -1)
    cursor.limit.assert_called_once_with(1)
    assert doc == {"_id": "abc", "policy_document_id": "policy-1"}


def test_get_last_compliance_result_missing_is_none_and_skips_id_without_id() -> None:
    """No match returns None; a row without `_id` is returned unchanged."""
    storage, results = _storage_with_results()
    empty = MagicMock()
    empty.sort.return_value = empty
    empty.limit.return_value = iter([])
    results.find.return_value = empty

    with patch("compliance_storage.run_in_thread", side_effect=_sync_run):
        missing = asyncio.run(storage.get_last_compliance_result("missing"))
    assert missing is None

    no_id = MagicMock()
    no_id.sort.return_value = no_id
    no_id.limit.return_value = iter([{"policy_document_id": "policy-1"}])
    results.find.return_value = no_id

    with patch("compliance_storage.run_in_thread", side_effect=_sync_run):
        doc = asyncio.run(storage.get_last_compliance_result("policy-1"))
    assert doc == {"policy_document_id": "policy-1"}
