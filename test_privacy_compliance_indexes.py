"""Regression tests for privacy-compliance unique index bootstrap.

ensure_privacy_compliance_indexes runs at compliance startup. If the unique
keys or unique=True flags drift, statute/policy collections can silently
accept duplicates that later gap-analysis and sub-vector-index jobs treat as
canonical rows.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from db import PRIVACY_COMPLIANCE_DB, ensure_privacy_compliance_indexes


EXPECTED_UNIQUE_INDEXES = {
    "statutes": [("document_id", 1)],
    "policies": [("document_id", 1)],
    "statute_chunks": [("document_id", 1), ("chunk_index", 1)],
    "policy_chunks": [("document_id", 1), ("chunk_index", 1)],
    "policy_sub_chunks": [("document_id", 1), ("subchunk_id", 1)],
    "statute_sub_chunks": [("document_id", 1), ("subchunk_id", 1)],
    "policy_sub_embeddings": [("document_id", 1), ("subchunk_id", 1)],
    "statute_sub_embeddings": [("document_id", 1), ("subchunk_id", 1)],
}


def test_ensure_privacy_compliance_indexes_creates_unique_keys() -> None:
    """Startup must create unique indexes on the eight compliance collections."""
    collections: dict[str, MagicMock] = {}

    def _get_coll(name: str) -> MagicMock:
        collections[name] = MagicMock(name=name)
        return collections[name]

    db = MagicMock()
    db.__getitem__.side_effect = _get_coll
    client = MagicMock()
    client.__getitem__.return_value = db

    ensure_privacy_compliance_indexes(client)

    client.__getitem__.assert_called_once_with(PRIVACY_COMPLIANCE_DB)
    assert set(collections) == set(EXPECTED_UNIQUE_INDEXES)
    for name, keys in EXPECTED_UNIQUE_INDEXES.items():
        collections[name].create_index.assert_called_once_with(keys, unique=True)


def test_ensure_privacy_compliance_indexes_custom_database() -> None:
    """Callers can target a non-default compliance database name."""
    db = MagicMock()
    client = MagicMock()
    client.__getitem__.return_value = db

    ensure_privacy_compliance_indexes(client, database_name="compliance-alt")

    client.__getitem__.assert_called_once_with("compliance-alt")
    assert db.__getitem__.call_count == len(EXPECTED_UNIQUE_INDEXES)


def test_ensure_privacy_compliance_indexes_continues_after_collection_failure() -> None:
    """A create_index failure on one collection must not skip the rest."""
    collections: dict[str, MagicMock] = {}

    def _get_coll(name: str) -> MagicMock:
        coll = MagicMock(name=name)
        if name == "statute_chunks":
            coll.create_index.side_effect = RuntimeError("index already exists with different options")
        collections[name] = coll
        return coll

    db = MagicMock()
    db.__getitem__.side_effect = _get_coll
    client = MagicMock()
    client.__getitem__.return_value = db

    ensure_privacy_compliance_indexes(client)

    assert set(collections) == set(EXPECTED_UNIQUE_INDEXES)
    for name, keys in EXPECTED_UNIQUE_INDEXES.items():
        collections[name].create_index.assert_called_once_with(keys, unique=True)
