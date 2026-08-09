"""Regression tests for db embedding-model lookup helpers."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from db import (
    EMBEDDING_MODEL_COLLECTION,
    WEB_GATHER_DB,
    get_embedding_model_name,
    get_embedding_model_record,
    init_db,
)


@pytest.fixture
def mock_mongo() -> MagicMock:
    mock_mongo = MagicMock()
    init_db(mock_mongo)
    return mock_mongo


def test_get_embedding_model_name_returns_model(mock_mongo: MagicMock) -> None:
    wg_db = MagicMock()
    coll = MagicMock()
    coll.find_one.return_value = {
        "database_name": "privacy-compliance",
        "model_name": "all-MiniLM-L6-v2",
    }
    wg_db.__getitem__.return_value = coll
    mock_mongo.__getitem__.return_value = wg_db

    name = get_embedding_model_name("privacy-compliance")

    assert name == "all-MiniLM-L6-v2"
    mock_mongo.__getitem__.assert_called_with(WEB_GATHER_DB)
    wg_db.__getitem__.assert_called_with(EMBEDDING_MODEL_COLLECTION)
    coll.find_one.assert_called_once_with({"database_name": "privacy-compliance"})


def test_get_embedding_model_name_missing_returns_none(mock_mongo: MagicMock) -> None:
    wg_db = MagicMock()
    coll = MagicMock()
    coll.find_one.return_value = None
    wg_db.__getitem__.return_value = coll
    mock_mongo.__getitem__.return_value = wg_db

    assert get_embedding_model_name("unknown-db") is None


def test_get_embedding_model_record_returns_full_document(mock_mongo: MagicMock) -> None:
    record = {
        "database_name": "index_db",
        "model_name": "all-MiniLM-L6-v2",
        "fields": [{"type": "vector", "path": "embedding", "numDimensions": 384}],
    }
    wg_db = MagicMock()
    coll = MagicMock()
    coll.find_one.return_value = record
    wg_db.__getitem__.return_value = coll
    mock_mongo.__getitem__.return_value = wg_db

    assert get_embedding_model_record("index_db") is record
    coll.find_one.assert_called_once_with({"database_name": "index_db"})


def test_get_embedding_model_record_without_client_returns_none() -> None:
    # init_db(None) clears the module-level client used by the helper.
    init_db(None)
    assert get_embedding_model_record("any") is None
