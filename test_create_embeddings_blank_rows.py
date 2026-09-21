"""Leftover create-embeddings all-blank source-row 400 vs empty-collection 404."""
from __future__ import annotations

import sys
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("pyppeteer", MagicMock())

from conftest import DummyModel, _wire_mongo_core
from core import core_bp, init_core


@pytest.fixture
def app() -> Flask:
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app


@pytest.fixture
def client(app: Flask):
    return app.test_client()


@pytest.fixture
def mock_clients() -> Dict[str, Any]:
    mock_mongo = MagicMock()
    mock_firecrawl = MagicMock()
    mock_anthropic = MagicMock()
    init_core(mock_mongo, mock_firecrawl, mock_anthropic)
    return {
        "mongo": mock_mongo,
        "firecrawl": mock_firecrawl,
        "anthropic": mock_anthropic,
    }


def _index_payload(**overrides: Any) -> Dict[str, Any]:
    payload = {
        "source_database_name": "src",
        "source_collection_name": "docs",
        "index_database_name": "index_db",
        "index_collection_name": "chunks",
    }
    payload.update(overrides)
    return payload


def test_create_embeddings_all_blank_text_is_400_not_404(client, mock_clients) -> None:
    """find() hits rows but every text_column is blank: 400, no encode/wipe/insert."""
    dbs = _wire_mongo_core(mock_clients["mongo"])
    source_coll = dbs["source_db"].__getitem__.return_value
    index_coll = dbs["index_db"].__getitem__.return_value
    source_coll.find.return_value = [
        {"_id": "row-1", "chunk_text": ""},
        {"_id": "row-2", "chunk_text": "   "},
        {"_id": "row-3", "chunk_text": None},
        {"_id": "row-4"},
    ]
    dummy = DummyModel()

    with patch("core.get_embedding_model_name", return_value="model"), patch(
        "core.get_model", return_value=dummy
    ) as get_model:
        response = client.post("/create-embeddings", json=_index_payload())

    assert response.status_code == 400
    assert "no rows with 'chunk_text'" in response.get_json()["error"]
    source_coll.find.assert_called_once_with({})
    get_model.assert_not_called()
    index_coll.delete_many.assert_not_called()
    index_coll.insert_many.assert_not_called()
    index_coll.update_one.assert_not_called()


def test_create_embeddings_empty_find_is_404_before_blank_check(
    client, mock_clients
) -> None:
    """No source documents is 404, distinct from all-blank 400 after a successful find()."""
    dbs = _wire_mongo_core(mock_clients["mongo"])
    source_coll = dbs["source_db"].__getitem__.return_value
    index_coll = dbs["index_db"].__getitem__.return_value
    source_coll.find.return_value = []

    with patch("core.get_embedding_model_name", return_value="model"), patch(
        "core.get_model", return_value=DummyModel()
    ) as get_model:
        response = client.post("/create-embeddings", json=_index_payload())

    assert response.status_code == 404
    err = response.get_json()["error"]
    assert "No documents found" in err
    assert "src.docs" in err
    get_model.assert_not_called()
    index_coll.delete_many.assert_not_called()
    index_coll.insert_many.assert_not_called()
