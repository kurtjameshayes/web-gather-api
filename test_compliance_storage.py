"""Regression tests for compliance suite persistence helpers."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from compliance_config import load_config
from compliance_storage import ComplianceStorage


class _Cursor:
    def __init__(self, docs):
        self._docs = docs
        self.sort_calls = []
        self.skip_calls = []
        self.limit_calls = []

    def sort(self, *args):
        self.sort_calls.append(args)
        return self

    def skip(self, *args):
        self.skip_calls.append(args)
        return self

    def limit(self, *args):
        self.limit_calls.append(args)
        return self

    def __iter__(self):
        return iter(self._docs)

    def __next__(self):
        return next(iter(self._docs))


@pytest.fixture
def storage_harness(monkeypatch):
    async def run_sync(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr("compliance_storage.run_in_thread", run_sync)
    config = load_config()
    results = MagicMock()
    alerts = MagicMock()
    run_log = MagicMock()
    policies = MagicMock()
    database = {
        config.compliance_results_collection: results,
        config.compliance_alerts_collection: alerts,
        config.compliance_run_log_collection: run_log,
        config.policies_collection: policies,
    }
    client = {config.compliance_database: database}
    storage = ComplianceStorage(client, config)
    return SimpleNamespace(
        storage=storage,
        results=results,
        alerts=alerts,
        run_log=run_log,
        policies=policies,
        config=config,
    )


def test_write_compliance_result_adds_id_and_timestamp(storage_harness) -> None:
    doc = {"policy_document_id": "policy-1"}

    inserted_id = asyncio.run(storage_harness.storage.write_compliance_result(doc))

    assert inserted_id == doc["_id"]
    assert doc["policy_document_id"] == "policy-1"
    assert isinstance(datetime.fromisoformat(doc["analyzed_at"]), datetime)
    storage_harness.results.insert_one.assert_called_once_with(doc)


def test_get_last_compliance_result_filters_and_normalizes_id(storage_harness) -> None:
    latest_doc = {
        "_id": 12345,
        "policy_document_id": "policy-1",
        "statute_index_version": "idx-2026",
    }
    cursor = _Cursor([latest_doc])
    storage_harness.results.find.return_value = cursor

    result = asyncio.run(
        storage_harness.storage.get_last_compliance_result(
            "policy-1",
            statute_index_version="idx-2026",
        )
    )

    assert result == {
        "_id": "12345",
        "policy_document_id": "policy-1",
        "statute_index_version": "idx-2026",
    }
    storage_harness.results.find.assert_called_once_with(
        {
            "policy_document_id": "policy-1",
            "statute_index_version": "idx-2026",
        }
    )
    assert cursor.sort_calls == [("analyzed_at", -1)]
    assert cursor.limit_calls == [(1,)]


@pytest.mark.parametrize(
    ("stored_value", "expected"),
    [
        (
            datetime(2026, 4, 26, 10, 0, tzinfo=timezone.utc),
            datetime(2026, 4, 26, 10, 0, tzinfo=timezone.utc),
        ),
        (
            "2026-04-26T10:00:00Z",
            datetime(2026, 4, 26, 10, 0, tzinfo=timezone.utc),
        ),
        ("not-a-date", None),
        (42, None),
    ],
)
def test_get_last_drift_check_at_parses_stored_timestamps(
    storage_harness,
    stored_value,
    expected,
) -> None:
    storage_harness.run_log.find_one.return_value = {
        "type": "drift_meta",
        "last_drift_check_at": stored_value,
    }

    result = asyncio.run(storage_harness.storage.get_last_drift_check_at())

    assert result == expected
    storage_harness.run_log.find_one.assert_called_once_with(
        {"type": "drift_meta"},
        sort=[("last_drift_check_at", -1)],
    )


def test_list_runs_builds_filters_and_infers_summary_types(storage_harness) -> None:
    cursor = _Cursor([
        {
            "_id": 9876,
            "policy_document_id": "policy-1",
            "company_name": "Acme",
            "analyzed_at": "2026-04-26T10:00:00+00:00",
            "gaps": [],
            "privacy_health_score": 82,
            "score_assessment": "good",
            "summary": {"gap_count": 0},
        }
    ])
    storage_harness.results.count_documents.return_value = 1
    storage_harness.results.find.return_value = cursor

    runs, total = asyncio.run(
        storage_harness.storage.list_runs(
            policy_document_id="policy-1",
            since="2026-04-01T00:00:00+00:00",
            until="2026-04-30T00:00:00+00:00",
            types=["gap"],
            limit=10,
            offset=5,
        )
    )

    expected_query = {
        "policy_document_id": "policy-1",
        "run_types": {"$in": ["gap"]},
        "analyzed_at": {
            "$gte": "2026-04-01T00:00:00+00:00",
            "$lte": "2026-04-30T00:00:00+00:00",
        },
    }
    assert total == 1
    assert runs == [
        {
            "run_id": "9876",
            "policy_document_id": "policy-1",
            "company_name": "Acme",
            "run_at": "2026-04-26T10:00:00+00:00",
            "types": ["gap", "health_score"],
            "privacy_health_score": 82,
            "score_assessment": "good",
            "summary": {"gap_count": 0},
        }
    ]
    storage_harness.results.count_documents.assert_called_once_with(expected_query)
    storage_harness.results.find.assert_called_once_with(expected_query)
    assert cursor.sort_calls == [("analyzed_at", -1)]
    assert cursor.skip_calls == [(5,)]
    assert cursor.limit_calls == [(10,)]
