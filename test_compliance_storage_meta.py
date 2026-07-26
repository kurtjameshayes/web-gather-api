"""Regression tests for drift checkpoint, alert write, and run lookup storage."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

from compliance_config import load_config
from compliance_storage import ComplianceStorage


async def _sync_run(func, *args, **kwargs):
    """Run storage thread-pool callbacks inline for deterministic assertions."""
    return func(*args, **kwargs)


def _storage_with_db() -> tuple[ComplianceStorage, Any, MagicMock]:
    config = load_config()
    mongo_client = MagicMock()
    db = MagicMock()
    mongo_client.__getitem__.return_value = db
    return ComplianceStorage(mongo_client, config), config, db


def test_get_last_drift_check_at_parses_iso_z_and_rejects_invalid() -> None:
    """Drift checkpoint reads must tolerate datetime objects, Z-suffix ISO, and bad values."""
    storage, config, db = _storage_with_db()
    run_log = MagicMock()
    db.__getitem__.return_value = run_log

    aware = datetime(2026, 3, 21, 12, 0, tzinfo=timezone.utc)
    run_log.find_one.return_value = {"type": "drift_meta", "last_drift_check_at": aware}
    with patch("compliance_storage.run_in_thread", side_effect=_sync_run):
        assert asyncio.run(storage.get_last_drift_check_at()) == aware

    run_log.find_one.return_value = {
        "type": "drift_meta",
        "last_drift_check_at": "2026-03-21T12:00:00Z",
    }
    with patch("compliance_storage.run_in_thread", side_effect=_sync_run):
        parsed = asyncio.run(storage.get_last_drift_check_at())
    assert parsed == datetime(2026, 3, 21, 12, 0, tzinfo=timezone.utc)

    run_log.find_one.return_value = {
        "type": "drift_meta",
        "last_drift_check_at": "not-a-timestamp",
    }
    with patch("compliance_storage.run_in_thread", side_effect=_sync_run):
        assert asyncio.run(storage.get_last_drift_check_at()) is None

    run_log.find_one.return_value = {"type": "drift_meta"}
    with patch("compliance_storage.run_in_thread", side_effect=_sync_run):
        assert asyncio.run(storage.get_last_drift_check_at()) is None

    run_log.find_one.assert_called_with(
        {"type": "drift_meta"}, sort=[("last_drift_check_at", -1)]
    )


def test_set_last_drift_check_at_writes_drift_meta_marker() -> None:
    """Completing a drift check must append a typed checkpoint to the run log."""
    storage, config, db = _storage_with_db()
    run_log = MagicMock()
    db.__getitem__.return_value = run_log
    fixed = datetime(2026, 7, 26, 10, 0, tzinfo=timezone.utc)

    with patch("compliance_storage.run_in_thread", side_effect=_sync_run):
        asyncio.run(storage.set_last_drift_check_at(fixed))

    db.__getitem__.assert_called_with(config.compliance_run_log_collection)
    run_log.insert_one.assert_called_once()
    doc = run_log.insert_one.call_args[0][0]
    assert doc["type"] == "drift_meta"
    assert doc["last_drift_check_at"] == fixed.isoformat()


def test_write_compliance_alert_defaults_id_type_and_timestamp() -> None:
    """Alert persistence must mint ids and default type/timestamp when omitted."""
    storage, config, db = _storage_with_db()
    alerts = MagicMock()
    db.__getitem__.return_value = alerts

    with patch("compliance_storage.run_in_thread", side_effect=_sync_run):
        alert_id = asyncio.run(
            storage.write_compliance_alert(
                {
                    "policy_document_id": "policy-1",
                    "company_name": "Acme",
                    "affected_jurisdictions": ["CA"],
                }
            )
        )

    db.__getitem__.assert_called_with(config.compliance_alerts_collection)
    alerts.insert_one.assert_called_once()
    doc = alerts.insert_one.call_args[0][0]
    assert alert_id == doc["alert_id"]
    assert len(alert_id) == 36
    assert doc["type"] == "regulatory_drift"
    assert doc["policy_document_id"] == "policy-1"
    assert "detected_at" in doc


def test_get_run_by_id_normalizes_mongo_id_fields() -> None:
    """Run detail responses should expose run_id and stringify Mongo _id."""
    storage, config, db = _storage_with_db()
    results = MagicMock()
    db.__getitem__.return_value = results
    results.find_one.return_value = {
        "_id": "run-123",
        "policy_document_id": "policy-1",
        "privacy_health_score": 88,
    }

    with patch("compliance_storage.run_in_thread", side_effect=_sync_run):
        doc = asyncio.run(storage.get_run_by_id("run-123"))

    results.find_one.assert_called_once_with({"_id": "run-123"})
    assert doc == {
        "_id": "run-123",
        "run_id": "run-123",
        "policy_document_id": "policy-1",
        "privacy_health_score": 88,
    }

    results.find_one.return_value = None
    with patch("compliance_storage.run_in_thread", side_effect=_sync_run):
        assert asyncio.run(storage.get_run_by_id("missing")) is None


def test_list_policy_document_ids_stringifies_ids_from_policies_collection() -> None:
    """Drift rebaseline must enumerate policy ids from the configured collection."""
    storage, config, db = _storage_with_db()
    policies = MagicMock()
    db.__getitem__.return_value = policies
    policies.find.return_value = [
        {"_id": "policy-1"},
        {"_id": 42},
        {"name": "no-id"},
    ]

    with patch("compliance_storage.run_in_thread", side_effect=_sync_run):
        ids = asyncio.run(storage.list_policy_document_ids("custom-db"))

    storage._client.__getitem__.assert_called_with("custom-db")
    db.__getitem__.assert_called_with(config.policies_collection)
    policies.find.assert_called_once_with({}, {"_id": 1})
    assert ids == ["policy-1", "42"]
