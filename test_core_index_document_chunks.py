"""Regression tests for document chunk indexing and DB/collection warnings."""
from __future__ import annotations

import logging
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Avoid loading sentence_transformers in unit tests.
sys.modules["sentence_transformers"] = MagicMock()

import core
from conftest import DummyModel
from db import (
    PRIVACY_COMPLIANCE_DB,
    WEB_GATHER_DB,
    get_application_embedding_model,
    set_application_embedding_model,
)


@pytest.fixture(autouse=True)
def _reset_application_embedding_model():
    previous = get_application_embedding_model()
    set_application_embedding_model(None)
    yield
    set_application_embedding_model(previous)


@pytest.fixture
def mock_mongo():
    mock_client = MagicMock()
    core.init_core(mock_client, MagicMock(), MagicMock())
    return mock_client


def test_application_embedding_model_roundtrip() -> None:
    assert get_application_embedding_model() is None
    set_application_embedding_model("all-MiniLM-L6-v2")
    assert get_application_embedding_model() == "all-MiniLM-L6-v2"
    set_application_embedding_model(None)
    assert get_application_embedding_model() is None


def test_warn_if_database_or_collection_not_found_handles_missing_targets(
    mock_mongo, caplog: pytest.LogCaptureFixture
) -> None:
    mock_mongo.list_database_names.return_value = ["other-db"]
    with caplog.at_level(logging.WARNING, logger="web-gather-api"):
        core._warn_if_database_or_collection_not_found(
            "missing-db", "chunks", "POST /create-chunks"
        )
    assert any("Database 'missing-db' not found" in r.message for r in caplog.records)

    mock_mongo.list_database_names.return_value = ["source-db"]
    source_db = MagicMock()
    source_db.list_collection_names.return_value = ["policies"]
    mock_mongo.__getitem__.return_value = source_db
    with caplog.at_level(logging.WARNING, logger="web-gather-api"):
        core._warn_if_database_or_collection_not_found(
            "source-db", "chunks", "POST /create-chunks"
        )
    assert any(
        "Source collection 'chunks' not found in database 'source-db'" in r.message
        for r in caplog.records
    )


def test_warn_if_database_or_collection_not_found_skips_without_client(
    caplog: pytest.LogCaptureFixture,
) -> None:
    core.mongo_client = None
    with caplog.at_level(logging.WARNING, logger="web-gather-api"):
        core._warn_if_database_or_collection_not_found("db", "coll", "prefix")
    assert caplog.records == []


def test_warn_if_database_or_collection_not_found_swallows_list_errors(
    mock_mongo, caplog: pytest.LogCaptureFixture
) -> None:
    mock_mongo.list_database_names.side_effect = RuntimeError("mongo down")
    with caplog.at_level(logging.WARNING, logger="web-gather-api"):
        core._warn_if_database_or_collection_not_found("db", "coll", "POST /index")
    assert any("Could not verify database/collections" in r.message for r in caplog.records)


def test_index_document_chunks_skips_when_no_embedding_model(mock_mongo) -> None:
    with patch("core.get_embedding_model_name", return_value=None):
        result = core.index_document_chunks(
            document_id="doc-1",
            text="Some policy text about retention.",
            database_name="source-db",
            collection_name="policies",
        )
    assert result is None
    mock_mongo.__getitem__.assert_not_called()


def test_index_document_chunks_uses_application_model_for_privacy_compliance(
    mock_mongo,
) -> None:
    set_application_embedding_model("app-default-model")
    index_db = MagicMock()
    chunk_coll = MagicMock()
    docs_coll = MagicMock()
    wg_db = MagicMock()

    def _get_db(name: str):
        if name == PRIVACY_COMPLIANCE_DB:
            return index_db
        if name == WEB_GATHER_DB:
            return wg_db
        return MagicMock()

    mock_mongo.__getitem__.side_effect = _get_db
    index_db.__getitem__.return_value = chunk_coll
    wg_db.__getitem__.return_value = docs_coll
    chunk_coll.delete_many.return_value = MagicMock(deleted_count=1)

    model = DummyModel()
    with (
        patch("core.get_embedding_model_name") as get_db_model,
        patch("core.get_model", return_value=model) as get_model,
        patch("core.chunk_text", return_value=["chunk-a", "chunk-b"]),
        patch.object(
            DummyModel,
            "encode",
            return_value=np.array([[0.1, 0.2], [0.3, 0.4]]),
        ),
    ):
        result = core.index_document_chunks(
            document_id="pol-1",
            text="Policy body",
            database_name=PRIVACY_COMPLIANCE_DB,
            collection_name="policies",
            index_database_name=PRIVACY_COMPLIANCE_DB,
            index_collection_name="policy_chunks",
            chunk_size=100,
            chunk_overlap=10,
            splitting_strategy="sentence",
        )

    get_db_model.assert_not_called()
    get_model.assert_called_once_with("app-default-model")
    chunk_coll.delete_many.assert_called_once_with({"document_id": "pol-1"})
    chunk_coll.insert_many.assert_called_once()
    inserted = chunk_coll.insert_many.call_args.args[0]
    assert len(inserted) == 2
    assert inserted[0]["document_id"] == "pol-1"
    assert inserted[0]["chunk_index"] == 0
    assert inserted[0]["text"] == "chunk-a"
    assert inserted[0]["embedding"] == [0.1, 0.2]
    docs_coll.update_one.assert_called_once()
    update_filter, update_doc = docs_coll.update_one.call_args.args
    assert update_filter == {"document_id": "pol-1"}
    assert update_doc["$set"]["chunk_count"] == 2
    assert update_doc["$set"]["embedding_model"] == "app-default-model"
    assert update_doc["$set"]["chunk_collection"] == "policy_chunks"
    assert result == {
        "chunks_indexed": 2,
        "embedding_model": "app-default-model",
        "chunk_collection": "policy_chunks",
        "chunk_size": 100,
        "chunk_overlap": 10,
        "splitting_strategy": "sentence",
    }


def test_index_document_chunks_returns_none_for_empty_chunk_output(mock_mongo) -> None:
    with (
        patch("core.get_embedding_model_name", return_value="all-MiniLM-L6-v2"),
        patch("core.get_model", return_value=DummyModel()),
        patch("core.chunk_text", return_value=[]),
    ):
        result = core.index_document_chunks(
            document_id="doc-empty",
            text="   ",
            database_name="source-db",
            collection_name="docs",
        )
    assert result is None
