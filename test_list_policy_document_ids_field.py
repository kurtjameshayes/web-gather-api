"""ComplianceStorage.list_policy_document_ids uses Mongo _id, not document_id."""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from bson import ObjectId

from compliance_config import load_config
from compliance_storage import ComplianceStorage


def test_list_policy_document_ids_projects_and_returns_id_field() -> None:
    """Drift policy enumeration finds {}, {_id: 1} and stringifies _id, ignoring document_id."""
    oid = ObjectId()
    mongo = MagicMock()
    coll = MagicMock()
    coll.find.return_value = [
        {"_id": oid, "document_id": "pol-should-not-be-used"},
        {"_id": "pol-string-id", "document_id": "other"},
        {"document_id": "missing-id-skipped"},
    ]
    mongo.__getitem__.return_value.__getitem__.return_value = coll
    config = load_config()
    storage = ComplianceStorage(mongo, config)

    ids = asyncio.run(storage.list_policy_document_ids())

    assert ids == [str(oid), "pol-string-id"]
    mongo.__getitem__.assert_called_with(config.compliance_database)
    mongo.__getitem__.return_value.__getitem__.assert_called_with(config.policies_collection)
    coll.find.assert_called_once_with({}, {"_id": 1})


def test_list_policy_document_ids_honors_database_override() -> None:
    """An explicit database argument is used instead of the compliance database."""
    mongo = MagicMock()
    coll = MagicMock()
    coll.find.return_value = [{"_id": "p1"}]
    mongo.__getitem__.return_value.__getitem__.return_value = coll
    storage = ComplianceStorage(mongo, load_config())

    ids = asyncio.run(storage.list_policy_document_ids("other-db"))

    assert ids == ["p1"]
    mongo.__getitem__.assert_called_with("other-db")
