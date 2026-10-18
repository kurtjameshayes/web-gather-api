"""POST /create-embeddings model selection, text column coercion, and delete scope."""
from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from flask import Flask

sys.modules["sentence_transformers"] = MagicMock()

from core import WEB_GATHER_DB, core_bp, init_core
from db import PRIVACY_COMPLIANCE_DB


class RecordingModel:
    def __init__(self) -> None:
        self.inputs: list[str] | None = None

    def encode(self, inputs: Any, **kwargs: object) -> np.ndarray:
        self.inputs = list(inputs)
        return np.array([[1.0, 0.0] for _ in inputs])


@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app.test_client()


@pytest.fixture
def mongo():
    mock_mongo = MagicMock()
    init_core(mock_mongo, MagicMock(), MagicMock())
    return mock_mongo


def _wire(mock_mongo: MagicMock, index_name: str) -> dict[str, MagicMock]:
    source_db = MagicMock()
    index_db = MagicMock()
    wg_db = MagicMock()

    def _get_db(name: str) -> MagicMock:
        if name == WEB_GATHER_DB:
            return wg_db
        if name == index_name:
            return index_db
        return source_db

    mock_mongo.__getitem__.side_effect = _get_db
    source_db.__getitem__.return_value = MagicMock()
    index_db.__getitem__.return_value = MagicMock()
    wg_db.__getitem__.return_value = MagicMock()
    return {"source_db": source_db, "index_db": index_db, "wg_db": wg_db}


def _post(client, **overrides: Any):
    body = {
        "source_database_name": "src",
        "source_collection_name": "docs",
        "index_database_name": "index_db",
        "index_collection_name": "chunks",
    }
    body.update(overrides)
    return client.post("/create-embeddings", json=body)


def test_invalid_source_query_is_rejected_before_missing_params(client, mongo) -> None:
    response = _post(
        client,
        source_database_name="",
        source_query="{not-json}",
    )
    assert response.status_code == 400
    assert "source_query" in response.get_json()["error"]
    assert "Missing required parameters" not in response.get_json()["error"]


def test_missing_params_win_over_blank_text_column(client, mongo) -> None:
    dbs = _wire(mongo, "index_db")
    response = _post(client, source_database_name="", text_column="   ")
    assert response.status_code == 400
    assert "Missing required parameters" in response.get_json()["error"]
    dbs["source_db"].__getitem__.return_value.find.assert_not_called()


@pytest.mark.parametrize("text_column", ["   ", 12, ["chunk_text"]])
def test_text_column_must_be_a_non_empty_string(client, mongo, text_column: Any) -> None:
    dbs = _wire(mongo, "index_db")
    with patch("core.get_embedding_model_name", return_value="db-model") as model_name, patch(
        "core.get_model"
    ) as get_model:
        response = _post(client, text_column=text_column)
    assert response.status_code == 400
    assert response.get_json()["error"] == "text_column must be a non-empty string"
    dbs["source_db"].__getitem__.return_value.find.assert_not_called()
    model_name.assert_not_called()
    get_model.assert_not_called()


@pytest.mark.parametrize("text_column", [None, "", False])
def test_falsy_text_column_reads_chunk_text(client, mongo, text_column: Any) -> None:
    dbs = _wire(mongo, "index_db")
    source = dbs["source_db"].__getitem__.return_value
    source.find.return_value = [
        {"_id": "row-1", "chunk_text": "from-chunk", "body": "from-body"},
    ]
    model = RecordingModel()
    body: dict[str, Any] = {}
    if text_column is not None:
        body["text_column"] = text_column
    with patch("core.get_embedding_model_name", return_value="db-model"), patch(
        "core.get_model", return_value=model
    ):
        response = _post(client, **body)
    assert response.status_code == 200
    assert response.get_json()["text_column"] == "chunk_text"
    assert model.inputs == ["from-chunk"]
    inserted = dbs["index_db"].__getitem__.return_value.insert_many.call_args[0][0]
    assert inserted[0]["chunk_text"] == "from-chunk"
    assert inserted[0]["body"] == "from-body"


def test_padded_text_column_is_stripped_before_read(client, mongo) -> None:
    dbs = _wire(mongo, "index_db")
    source = dbs["source_db"].__getitem__.return_value
    source.find.return_value = [{"_id": "row-1", "body": "kept", "chunk_text": "ignored"}]
    model = RecordingModel()
    with patch("core.get_embedding_model_name", return_value="db-model"), patch(
        "core.get_model", return_value=model
    ):
        response = _post(client, text_column="  body  ")
    assert response.status_code == 200
    assert response.get_json()["text_column"] == "body"
    assert model.inputs == ["kept"]
    inserted = dbs["index_db"].__getitem__.return_value.insert_many.call_args[0][0]
    assert inserted[0]["body"] == "kept"


def test_numeric_zero_false_and_null_category_are_copied_on_insert(client, mongo) -> None:
    dbs = _wire(mongo, "index_db")
    source = dbs["source_db"].__getitem__.return_value
    source.find.return_value = [
        {"_id": 15, "chunk_text": 0, "category": 0},
        {"_id": "row-2", "chunk_text": False, "category": None},
        {"_id": "row-3", "chunk_text": "ok"},
    ]
    model = RecordingModel()
    with patch("core.get_embedding_model_name", return_value="db-model"), patch(
        "core.get_model", return_value=model
    ):
        response = _post(client)
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["chunks_indexed"] == 3
    assert payload["skipped_rows"] == 0
    assert model.inputs == ["0", "False", "ok"]
    inserted = dbs["index_db"].__getitem__.return_value.insert_many.call_args[0][0]
    assert inserted[0]["chunk_text"] == "0"
    assert inserted[0]["category"] == 0
    assert inserted[0]["source_id"] == "15"
    assert inserted[1]["chunk_text"] == "False"
    assert inserted[1]["category"] is None
    assert "category" not in inserted[2]
    assert "_id" not in inserted[0]


def test_same_target_update_keeps_zero_category_and_omits_null(client, mongo) -> None:
    dbs = _wire(mongo, "src")
    collection = dbs["index_db"].__getitem__.return_value
    collection.find.return_value = [
        {"_id": "a", "chunk_text": "one", "category": 0},
        {"_id": "b", "chunk_text": "two", "category": None},
    ]
    with patch("core.get_embedding_model_name", return_value="db-model"), patch(
        "core.get_model", return_value=RecordingModel()
    ):
        response = _post(
            client,
            source_database_name="src",
            source_collection_name="docs",
            index_database_name="src",
            index_collection_name="docs",
        )
    assert response.status_code == 200
    collection.delete_many.assert_not_called()
    collection.insert_many.assert_not_called()
    assert collection.update_one.call_count == 2
    first = collection.update_one.call_args_list[0][0]
    second = collection.update_one.call_args_list[1][0]
    assert first[0] == {"_id": "a"}
    assert first[1]["$set"]["category"] == 0
    assert first[1]["$set"]["chunk_text"] == "one"
    assert second[0] == {"_id": "b"}
    assert "category" not in second[1]["$set"]
    assert second[1]["$set"]["chunk_text"] == "two"


def test_source_query_overwrites_delete_database_scope(client, mongo) -> None:
    dbs = _wire(mongo, "index_db")
    source = dbs["source_db"].__getitem__.return_value
    index = dbs["index_db"].__getitem__.return_value
    source.find.return_value = [{"_id": "row-1", "chunk_text": "kept"}]
    index.delete_many.return_value.deleted_count = 1
    query = {"source_database_name": "other-db", "document_id": "doc-9"}
    with patch("core.get_embedding_model_name", return_value="db-model"), patch(
        "core.get_model", return_value=RecordingModel()
    ):
        response = _post(client, source_query=query)
    assert response.status_code == 200
    source.find.assert_called_once_with(query)
    index.delete_many.assert_called_once_with(
        {
            "source_database_name": "other-db",
            "source_collection_name": "docs",
            "document_id": "doc-9",
        }
    )


def test_privacy_index_keeps_whitespace_application_model(client, mongo) -> None:
    dbs = _wire(mongo, PRIVACY_COMPLIANCE_DB)
    dbs["source_db"].__getitem__.return_value.find.return_value = [
        {"_id": "row-1", "chunk_text": "policy"},
    ]
    model = RecordingModel()
    with patch("core.get_application_embedding_model", return_value="  mini  ") as app_model, patch(
        "core.get_embedding_model_name"
    ) as db_model, patch("core.get_model", return_value=model) as get_model:
        response = _post(client, index_database_name=PRIVACY_COMPLIANCE_DB)
    assert response.status_code == 200
    assert response.get_json()["embedding_model"] == "  mini  "
    app_model.assert_called_once()
    db_model.assert_not_called()
    get_model.assert_called_once_with("  mini  ")
    assert model.inputs == ["policy"]


@pytest.mark.parametrize("application_model", [None, ""])
def test_privacy_index_falls_back_when_application_model_is_blank(
    client, mongo, application_model: str | None
) -> None:
    dbs = _wire(mongo, PRIVACY_COMPLIANCE_DB)
    dbs["source_db"].__getitem__.return_value.find.return_value = [
        {"_id": "row-1", "chunk_text": "policy"},
    ]
    with patch(
        "core.get_application_embedding_model", return_value=application_model
    ), patch(
        "core.get_embedding_model_name", return_value="db-model"
    ) as db_model, patch("core.get_model", return_value=RecordingModel()) as get_model:
        response = _post(client, index_database_name=PRIVACY_COMPLIANCE_DB)
    assert response.status_code == 200
    assert response.get_json()["embedding_model"] == "db-model"
    db_model.assert_called_once_with(PRIVACY_COMPLIANCE_DB)
    get_model.assert_called_once_with("db-model")


def test_privacy_index_without_any_model_does_not_read_rows(client, mongo) -> None:
    dbs = _wire(mongo, PRIVACY_COMPLIANCE_DB)
    with patch("core.get_application_embedding_model", return_value=None), patch(
        "core.get_embedding_model_name", return_value=None
    ), patch("core.get_model") as get_model:
        response = _post(client, index_database_name=PRIVACY_COMPLIANCE_DB)
    assert response.status_code == 400
    assert "embedding model" in response.get_json()["error"]
    dbs["source_db"].__getitem__.return_value.find.assert_not_called()
    get_model.assert_not_called()


def test_non_privacy_index_ignores_application_model(client, mongo) -> None:
    dbs = _wire(mongo, "index_db")
    dbs["source_db"].__getitem__.return_value.find.return_value = []
    with patch(
        "core.get_application_embedding_model", return_value="app-model"
    ) as app_model, patch(
        "core.get_embedding_model_name", return_value="db-model"
    ) as db_model, patch("core.get_model", return_value=RecordingModel()) as get_model:
        # find happens after model selection; an empty source is enough to prove the name.
        response = _post(client)
    assert response.status_code == 404
    app_model.assert_not_called()
    db_model.assert_called_once_with("index_db")
    get_model.assert_not_called()
