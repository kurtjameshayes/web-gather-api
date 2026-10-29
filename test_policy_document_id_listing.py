"""Drift re-analysis selects policies from Mongo ``_id``, not the document id field."""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from bson import ObjectId

from compliance_config import load_config
from compliance_storage import ComplianceStorage


def _storage(policies: MagicMock, database_name: str = "privacy-compliance") -> ComplianceStorage:
    config = load_config()
    config.compliance_database = "privacy-compliance"
    config.policies_collection = "policies"
    mongo = MagicMock()

    def get_db(name: str) -> MagicMock:
        db = MagicMock()

        def get_coll(coll: str) -> MagicMock:
            if name == database_name and coll == "policies":
                return policies
            return MagicMock()

        db.__getitem__.side_effect = get_coll
        return db

    mongo.__getitem__.side_effect = get_db
    return ComplianceStorage(mongo, config)


def test_list_policy_document_ids_uses_mongo_id_and_drops_falsy() -> None:
    """A missing or falsy ``_id`` is skipped even when ``document_id`` is set."""
    policies = MagicMock()
    oid = ObjectId("507f1f77bcf86cd799439011")
    policies.find.return_value = [
        {"_id": "pol-1", "document_id": "ignored-field"},
        {"_id": oid, "document_id": "also-ignored"},
        {"_id": 7, "document_id": "numeric"},
        {"_id": 0, "document_id": "zero-id"},
        {"_id": "", "document_id": "blank-id"},
        {"_id": False, "document_id": "false-id"},
        {"_id": None, "document_id": "none-id"},
        {"document_id": "no-mongo-id"},
        {"_id": "pol-2"},
    ]
    storage = _storage(policies)

    ids = asyncio.run(storage.list_policy_document_ids())

    assert ids == ["pol-1", str(oid), "7", "pol-2"]
    policies.find.assert_called_once_with({}, {"_id": 1})


def test_list_policy_document_ids_database_override() -> None:
    """An explicit database is queried; blank falls back; whitespace is kept."""
    config = load_config()
    config.compliance_database = "privacy-compliance"
    config.policies_collection = "policies"
    seen: list[tuple[str, str]] = []
    mongo = MagicMock()

    def get_db(name: str) -> MagicMock:
        db = MagicMock()

        def get_coll(coll: str) -> MagicMock:
            coll_mock = MagicMock()
            coll_mock.find.return_value = [{"_id": f"{name}-row"}]
            seen.append((name, coll))
            return coll_mock

        db.__getitem__.side_effect = get_coll
        return db

    mongo.__getitem__.side_effect = get_db
    storage = ComplianceStorage(mongo, config)

    assert asyncio.run(storage.list_policy_document_ids("archive-db")) == ["archive-db-row"]
    assert asyncio.run(storage.list_policy_document_ids("")) == ["privacy-compliance-row"]
    assert asyncio.run(storage.list_policy_document_ids("   ")) == ["   -row"]
    assert seen == [
        ("archive-db", "policies"),
        ("privacy-compliance", "policies"),
        ("   ", "policies"),
    ]
