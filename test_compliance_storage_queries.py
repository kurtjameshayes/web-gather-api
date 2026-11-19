"""Compliance run labeling, drift timestamps, and alert query escaping."""
from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from unittest.mock import MagicMock

from compliance_config import load_config
from compliance_storage import ComplianceStorage


def _storage() -> tuple[ComplianceStorage, Dict[str, MagicMock]]:
    config = load_config()
    results = MagicMock()
    alerts = MagicMock()
    run_log = MagicMock()
    policies = MagicMock()
    collections = {
        config.compliance_results_collection: results,
        config.compliance_alerts_collection: alerts,
        config.compliance_run_log_collection: run_log,
        config.policies_collection: policies,
    }
    database = MagicMock()
    database.__getitem__.side_effect = lambda name: collections[name]
    mongo = MagicMock()
    mongo.__getitem__.return_value = database
    return ComplianceStorage(mongo, config), {
        "results": results,
        "alerts": alerts,
        "run_log": run_log,
        "policies": policies,
    }


def _chain(find_target: MagicMock, docs: List[Dict[str, Any]], total: Optional[int] = None) -> MagicMock:
    chain = MagicMock()
    find_target.find.return_value = chain
    chain.sort.return_value = chain
    chain.skip.return_value = chain
    chain.limit.return_value = docs
    find_target.count_documents.return_value = len(docs) if total is None else total
    return chain


def test_list_runs_builds_filters_and_infers_missing_run_types() -> None:
    storage, cols = _storage()
    docs = [
        {
            "_id": "run-a",
            "policy_document_id": "pol-1",
            "company_name": "Acme",
            "analyzed_at": "2026-03-02T00:00:00+00:00",
            "gaps": [],
            "privacy_health_score": 0,
            "applicable_jurisdictions": ["CA"],
            "summary": {"missing": 0},
        },
        {
            "_id": "run-b",
            "analyzed_at": "2026-03-01T00:00:00+00:00",
            "strictest_common_denominator": {},
            "applicable_jurisdictions": ["CA"],
        },
        {
            "analyzed_at": "2026-02-01T00:00:00+00:00",
            "applicable_jurisdictions": [],
        },
        {
            "_id": "run-d",
            "run_types": ["custom"],
            "gaps": [{"status": "missing"}],
            "analyzed_at": "2026-01-01T00:00:00+00:00",
        },
        {
            "_id": "run-e",
            "run_types": [],
            "analyzed_at": "2025-01-01T00:00:00+00:00",
        },
    ]
    chain = _chain(cols["results"], docs, total=11)

    summaries, total = asyncio.run(
        storage.list_runs(
            policy_document_id="  pol  ",
            since="2026-01-01T00:00:00+00:00",
            until="2026-04-01T00:00:00+00:00",
            types=["gap", "health_score"],
            limit=10,
            offset=3,
        )
    )

    assert cols["results"].find.call_args.args[0] == {
        "policy_document_id": "  pol  ",
        "run_types": {"$in": ["gap", "health_score"]},
        "analyzed_at": {
            "$gte": "2026-01-01T00:00:00+00:00",
            "$lte": "2026-04-01T00:00:00+00:00",
        },
    }
    chain.sort.assert_called_once_with("analyzed_at", -1)
    chain.skip.assert_called_once_with(3)
    chain.limit.assert_called_once_with(10)
    assert total == 11
    assert summaries[0]["types"] == ["gap", "health_score"]
    assert summaries[0]["privacy_health_score"] == 0
    assert summaries[0]["company_name"] == "Acme"
    assert summaries[0]["summary"] == {"missing": 0}
    assert summaries[1]["types"] == ["multi_jurisdictional"]
    assert summaries[2]["run_id"] == ""
    assert summaries[2]["policy_document_id"] == ""
    assert summaries[2]["company_name"] is None
    assert summaries[2]["types"] == ["applicability"]
    assert summaries[3]["types"] == ["custom"]
    assert summaries[4]["types"] == []


def test_list_runs_omits_falsy_filters() -> None:
    storage, cols = _storage()
    _chain(cols["results"], [])

    summaries, total = asyncio.run(
        storage.list_runs(policy_document_id="", since="", types=[], limit=5, offset=0)
    )

    assert cols["results"].find.call_args.args[0] == {}
    assert summaries == []
    assert total == 0


def test_last_result_treats_blank_statute_version_as_a_real_filter() -> None:
    storage, cols = _storage()
    chain = MagicMock()
    cols["results"].find.return_value = chain
    chain.sort.return_value = chain
    chain.limit.return_value = iter([{"_id": 15, "policy_document_id": "pol-1"}])

    found = asyncio.run(storage.get_last_compliance_result("pol-1", statute_index_version=""))

    assert cols["results"].find.call_args.args[0] == {
        "policy_document_id": "pol-1",
        "statute_index_version": "",
    }
    chain.sort.assert_called_once_with("analyzed_at", -1)
    chain.limit.assert_called_once_with(1)
    assert found is not None
    assert found["_id"] == "15"

    chain.limit.return_value = iter([])
    assert asyncio.run(storage.get_last_compliance_result("pol-1")) is None
    assert cols["results"].find.call_args.args[0] == {"policy_document_id": "pol-1"}


def test_drift_checkpoint_parses_datetimes_and_rejects_bad_values() -> None:
    storage, cols = _storage()
    aware = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    cases = [
        (None, None),
        ({}, None),
        ({"last_drift_check_at": ""}, None),
        ({"last_drift_check_at": aware}, aware),
        ({"last_drift_check_at": "2026-01-02T03:04:05Z"}, datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)),
        ({"last_drift_check_at": "not-a-date"}, None),
        ({"last_drift_check_at": 1700000000}, None),
    ]
    for doc, expected in cases:
        cols["run_log"].find_one.return_value = doc
        assert asyncio.run(storage.get_last_drift_check_at()) == expected
    cols["run_log"].find_one.assert_called_with({"type": "drift_meta"}, sort=[("last_drift_check_at", -1)])


def test_alert_company_filter_escapes_regex_and_drops_mongo_id() -> None:
    storage, cols = _storage()
    docs = [{"_id": "mongo-id", "alert_id": "alert-1", "company_name": "Acme.Co (US)"}]
    chain = _chain(cols["alerts"], docs, total=4)

    alerts, total = asyncio.run(
        storage.list_alerts(
            policy_document_id="pol-1",
            company_name="  Acme.Co (US)  ",
            jurisdiction="ca",
            since="2026-01-01T00:00:00+00:00",
            limit=7,
            offset=2,
        )
    )

    query = cols["alerts"].find.call_args.args[0]
    assert query["policy_document_id"] == "pol-1"
    assert query["company_name"]["$options"] == "i"
    assert query["company_name"]["$regex"] == re.escape("Acme.Co (US)")
    assert query["affected_jurisdictions"] == "ca"
    assert query["detected_at"] == {"$gte": "2026-01-01T00:00:00+00:00"}
    chain.sort.assert_called_once_with("detected_at", -1)
    chain.skip.assert_called_once_with(2)
    chain.limit.assert_called_once_with(7)
    assert total == 4
    assert alerts == [{"alert_id": "alert-1", "company_name": "Acme.Co (US)"}]

    _chain(cols["alerts"], [])
    asyncio.run(storage.list_alerts(company_name="   "))
    assert cols["alerts"].find.call_args.args[0] == {}


def test_writes_keep_existing_ids_and_timestamps() -> None:
    storage, cols = _storage()
    result = {"_id": "keep", "analyzed_at": "2020-01-01T00:00:00+00:00"}
    stored_id = asyncio.run(storage.write_compliance_result(result))
    assert stored_id == "keep"
    inserted = cols["results"].insert_one.call_args.args[0]
    assert inserted["_id"] == "keep"
    assert inserted["analyzed_at"] == "2020-01-01T00:00:00+00:00"

    generated = asyncio.run(storage.write_compliance_result({}))
    created = cols["results"].insert_one.call_args.args[0]
    assert created["_id"] == generated
    assert len(generated) == 36
    assert "T" in created["analyzed_at"]

    alert = {
        "alert_id": "alert-9",
        "type": "custom",
        "detected_at": "2024-05-01T00:00:00+00:00",
    }
    assert asyncio.run(storage.write_compliance_alert(alert)) == "alert-9"
    assert cols["alerts"].insert_one.call_args.args[0]["type"] == "custom"
    assert cols["alerts"].insert_one.call_args.args[0]["detected_at"] == "2024-05-01T00:00:00+00:00"

    fresh: Dict[str, Any] = {}
    new_alert = asyncio.run(storage.write_compliance_alert(fresh))
    assert fresh["alert_id"] == new_alert
    assert fresh["type"] == "regulatory_drift"
    assert "T" in fresh["detected_at"]


def test_policy_id_listing_skips_falsy_ids() -> None:
    storage, cols = _storage()
    cols["policies"].find.return_value = [
        {"_id": "pol-1"},
        {"_id": ""},
        {"_id": 0},
        {"_id": None},
        {},
        {"_id": "pol-2"},
    ]

    ids = asyncio.run(storage.list_policy_document_ids())

    assert ids == ["pol-1", "pol-2"]
    cols["policies"].find.assert_called_once_with({}, {"_id": 1})
