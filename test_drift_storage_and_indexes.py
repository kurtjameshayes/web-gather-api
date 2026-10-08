"""Startup unique indexes and drift storage selection.

Drift re-checks every returned policy id and uses the last checkpoint
timestamp to decide the window. A bad id list or a mis-parsed checkpoint
scans the wrong corpus.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from compliance_storage import ComplianceStorage
from db import PRIVACY_COMPLIANCE_DB, ensure_privacy_compliance_indexes


_EXPECTED_INDEXES = [
    ("statutes", [("document_id", 1)], True),
    ("policies", [("document_id", 1)], True),
    ("statute_chunks", [("document_id", 1), ("chunk_index", 1)], True),
    ("policy_chunks", [("document_id", 1), ("chunk_index", 1)], True),
    ("policy_sub_chunks", [("document_id", 1), ("subchunk_id", 1)], True),
    ("statute_sub_chunks", [("document_id", 1), ("subchunk_id", 1)], True),
    ("policy_sub_embeddings", [("document_id", 1), ("subchunk_id", 1)], True),
    ("statute_sub_embeddings", [("document_id", 1), ("subchunk_id", 1)], True),
]


def _index_client(fail_on: str | None = None):
    calls = []
    databases = {}

    def get_db(name):
        if name not in databases:
            db = MagicMock()

            def get_coll(coll_name):
                coll = MagicMock()

                def create_index(keys, unique=False):
                    calls.append((name, coll_name, list(keys), unique))
                    if fail_on and coll_name == fail_on:
                        raise RuntimeError("index conflict")

                coll.create_index.side_effect = create_index
                return coll

            db.__getitem__.side_effect = get_coll
            databases[name] = db
        return databases[name]

    client = MagicMock()
    client.__getitem__.side_effect = get_db
    return client, calls


def test_privacy_indexes_are_unique_on_the_compliance_database() -> None:
    client, calls = _index_client()
    ensure_privacy_compliance_indexes(client)
    assert [(coll, keys, unique) for _db, coll, keys, unique in calls] == _EXPECTED_INDEXES
    assert {db for db, _coll, _keys, _unique in calls} == {PRIVACY_COMPLIANCE_DB}


def test_privacy_indexes_continue_after_one_collection_fails() -> None:
    client, calls = _index_client(fail_on="policies")
    ensure_privacy_compliance_indexes(client, database_name="tenant-compliance")
    assert [coll for _db, coll, _keys, _unique in calls] == [spec[0] for spec in _EXPECTED_INDEXES]
    assert {db for db, _coll, _keys, _unique in calls} == {"tenant-compliance"}


def _storage():
    cfg = MagicMock()
    cfg.compliance_database = "privacy-compliance"
    cfg.policies_collection = "policies"
    cfg.compliance_results_collection = "compliance_results"
    cfg.compliance_alerts_collection = "compliance_alerts"
    cfg.compliance_run_log_collection = "compliance_run_log"
    collections = {}

    def collection(db_name, coll_name):
        key = (db_name, coll_name)
        if key not in collections:
            coll = MagicMock()
            coll.find.return_value = []
            collections[key] = coll
        return collections[key]

    def get_db(name):
        db = MagicMock()
        db.__getitem__.side_effect = lambda coll_name, name=name: collection(name, coll_name)
        return db

    client = MagicMock()
    client.__getitem__.side_effect = get_db
    storage = ComplianceStorage(client, cfg)
    return storage, collection


async def _inline(func, *args, **kwargs):
    return func(*args, **kwargs)


def _run(coro):
    return asyncio.run(coro)


def test_list_policy_document_ids_skips_falsy_and_selects_database() -> None:
    storage, collection = _storage()
    policies = collection("privacy-compliance", "policies")
    policies.find.return_value = [
        {"_id": "pol-1"},
        {},
        {"_id": None},
        {"_id": ""},
        {"_id": 0},
        {"_id": 7},
        {"_id": "  "},
    ]
    other = collection("other-db", "policies")
    other.find.return_value = [{"_id": "only-other"}]
    padded = collection("  ", "policies")

    with patch("compliance_storage.run_in_thread", _inline):
        default_ids = _run(storage.list_policy_document_ids())
        blank_ids = _run(storage.list_policy_document_ids(""))
        other_ids = _run(storage.list_policy_document_ids("other-db"))
        padded_ids = _run(storage.list_policy_document_ids("  "))

    assert default_ids == ["pol-1", "7", "  "]
    assert blank_ids == ["pol-1", "7", "  "]
    assert other_ids == ["only-other"]
    assert padded_ids == []
    policies.find.assert_called_with({}, {"_id": 1})
    other.find.assert_called_once_with({}, {"_id": 1})
    padded.find.assert_called_once_with({}, {"_id": 1})


def test_get_last_drift_check_at_parses_checkpoint() -> None:
    storage, collection = _storage()
    run_log = collection("privacy-compliance", "compliance_run_log")
    aware = datetime(2026, 10, 8, 10, 12, 10, tzinfo=timezone.utc)

    cases = [
        (None, None),
        ({}, None),
        ({"last_drift_check_at": ""}, None),
        ({"last_drift_check_at": 0}, None),
        ({"last_drift_check_at": 1_700_000_000}, None),
        ({"last_drift_check_at": aware}, aware),
        ({"last_drift_check_at": "2026-10-08T10:12:10.750Z"}, datetime.fromisoformat("2026-10-08T10:12:10.750+00:00")),
        ({"last_drift_check_at": "yesterday"}, None),
    ]
    with patch("compliance_storage.run_in_thread", _inline):
        for doc, expected in cases:
            run_log.find_one.return_value = doc
            assert _run(storage.get_last_drift_check_at()) == expected

    run_log.find_one.assert_called_with(
        {"type": "drift_meta"},
        sort=[("last_drift_check_at", -1)],
    )


def test_set_last_drift_check_at_writes_iso_checkpoint() -> None:
    storage, collection = _storage()
    run_log = collection("privacy-compliance", "compliance_run_log")
    at = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    with patch("compliance_storage.run_in_thread", _inline):
        _run(storage.set_last_drift_check_at(at))
    inserted = run_log.insert_one.call_args[0][0]
    assert inserted == {
        "type": "drift_meta",
        "last_drift_check_at": at.isoformat(),
    }
