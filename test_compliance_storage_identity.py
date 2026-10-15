"""Regression tests for compliance result and alert identity rules.

Drift comparison reads the newest result for a policy. An explicit statute
index version, including an empty string, narrows that lookup. Alert company
filters are literal, case-insensitive matches.
"""
from __future__ import annotations

import asyncio
import re

from unittest.mock import MagicMock

from compliance_config import load_config
from compliance_storage import ComplianceStorage


def _storage():
    results = MagicMock()
    alerts = MagicMock()
    client = MagicMock()

    def _collection(name: str):
        if name == "compliance_results":
            return results
        if name == "compliance_alerts":
            return alerts
        return MagicMock()

    client.__getitem__.return_value.__getitem__.side_effect = _collection
    return ComplianceStorage(client, load_config()), results, alerts


def test_last_result_keeps_caller_identity_and_optional_version() -> None:
    storage, results, _alerts = _storage()
    limited = MagicMock()
    limited.__next__.return_value = {"_id": 15, "policy_document_id": "pol-1", "analyzed_at": "t2"}
    results.find.return_value.sort.return_value.limit.return_value = limited

    async def run():
        newest = await storage.get_last_compliance_result("pol-1")
        versioned = await storage.get_last_compliance_result("pol-1", statute_index_version="")
        kept = {"_id": "run-7", "analyzed_at": "2020-01-01T00:00:00+00:00"}
        kept_id = await storage.write_compliance_result(kept)
        generated = {}
        generated_id = await storage.write_compliance_result(generated)
        return newest, versioned, kept, kept_id, generated, generated_id

    newest, _versioned, kept, kept_id, generated, generated_id = asyncio.run(run())

    assert newest["_id"] == "15"
    assert newest["policy_document_id"] == "pol-1"
    first_query = results.find.call_args_list[0][0][0]
    assert first_query == {"policy_document_id": "pol-1"}
    results.find.return_value.sort.assert_called_with("analyzed_at", -1)
    second_query = results.find.call_args_list[1][0][0]
    assert second_query == {"policy_document_id": "pol-1", "statute_index_version": ""}

    assert kept_id == "run-7"
    assert kept["analyzed_at"] == "2020-01-01T00:00:00+00:00"
    assert generated_id == generated["_id"]
    assert len(generated_id) == 36
    assert "T" in generated["analyzed_at"]
    inserted = [call.args[0] for call in results.insert_one.call_args_list]
    assert inserted[0]["_id"] == "run-7"
    assert inserted[1]["_id"] == generated_id


def test_alert_company_filter_is_escaped_and_ids_are_preserved() -> None:
    storage, _results, alerts = _storage()
    alerts.count_documents.return_value = 1
    alerts.find.return_value.sort.return_value.skip.return_value.limit.return_value = [
        {"_id": "mongo-id", "alert_id": "a1", "company_name": "Acme.*"}
    ]

    async def run():
        rows, total = await storage.list_alerts(
            company_name="  Acme.*  ",
            policy_document_id="pol-1",
            jurisdiction="CA",
            since="2026-01-01T00:00:00Z",
            limit=2,
            offset=3,
        )
        kept = await storage.write_compliance_alert(
            {"alert_id": "alert-9", "type": "manual", "detected_at": "already-set"}
        )
        fresh = {}
        generated = await storage.write_compliance_alert(fresh)
        return rows, total, kept, fresh, generated

    rows, total, kept, fresh, generated = asyncio.run(run())

    query = alerts.find.call_args[0][0]
    assert query["company_name"] == {"$regex": re.escape("Acme.*"), "$options": "i"}
    assert query["policy_document_id"] == "pol-1"
    assert query["affected_jurisdictions"] == "CA"
    assert query["detected_at"] == {"$gte": "2026-01-01T00:00:00Z"}
    alerts.count_documents.assert_called_once_with(query)
    alerts.find.return_value.sort.assert_called_with("detected_at", -1)
    alerts.find.return_value.sort.return_value.skip.assert_called_with(3)
    alerts.find.return_value.sort.return_value.skip.return_value.limit.assert_called_with(2)
    assert total == 1
    assert rows == [{"alert_id": "a1", "company_name": "Acme.*"}]
    assert kept == "alert-9"
    written = alerts.insert_one.call_args_list[0].args[0]
    assert written["type"] == "manual"
    assert written["detected_at"] == "already-set"
    assert generated == fresh["alert_id"]
    assert len(generated) == 36
    generated_doc = alerts.insert_one.call_args_list[1].args[0]
    assert generated_doc["type"] == "regulatory_drift"
    assert "T" in generated_doc["detected_at"]
