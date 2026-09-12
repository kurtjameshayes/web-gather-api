"""GET /search reads chunk['text'], not chunk_text.

create-embeddings stores the embedded column as chunk_text by default.
/search currently KeyErrors (generic HTTP 500) when indexed rows only have
chunk_text. Pin that so a silent fallback or field rename is intentional.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from flask import Flask

sys.modules.setdefault("sentence_transformers", MagicMock())

from core import WEB_GATHER_DB, core_bp, init_core


@pytest.fixture
def mock_clients():
    mock_mongo = MagicMock()
    init_core(mock_mongo, MagicMock(), MagicMock())
    return mock_mongo


@pytest.fixture
def client(mock_clients):
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app.test_client()


class _DummyModel:
    def encode(self, inputs, **kwargs: object) -> np.ndarray:
        return np.array([[1.0, 0.0]])


def _wire_search_chunks(mock_mongo: MagicMock, chunks: list[dict]) -> None:
    wg_db = MagicMock()
    index_db = MagicMock()

    def _get_db(name: str) -> MagicMock:
        return wg_db if name == WEB_GATHER_DB else index_db

    mock_mongo.__getitem__.side_effect = _get_db
    wg_db.__getitem__.return_value.find_one.return_value = {
        "document_id": "doc-1",
        "database_name": "index_db",
        "collection_name": "docs",
        "index_database_name": "index_db",
        "chunk_collection": "chunks",
    }
    index_db.__getitem__.return_value.find.return_value = chunks


def test_search_chunk_text_only_is_generic_500(client, mock_clients) -> None:
    _wire_search_chunks(
        mock_clients,
        [{"chunk_text": "indexed via create-embeddings", "embedding": [1.0, 0.0]}],
    )

    with patch("core.get_embedding_model_name", return_value="model"), patch(
        "core.get_model", return_value=_DummyModel()
    ):
        response = client.get("/search?document_id=doc-1&query=privacy")

    assert response.status_code == 500
    assert response.get_json() is None or "text" not in (response.get_json() or {})


def test_search_uses_text_not_chunk_text_when_both_present(client, mock_clients) -> None:
    _wire_search_chunks(
        mock_clients,
        [
            {
                "text": "from text field",
                "chunk_text": "from chunk_text field",
                "embedding": [1.0, 0.0],
            }
        ],
    )

    with patch("core.get_embedding_model_name", return_value="model"), patch(
        "core.get_model", return_value=_DummyModel()
    ):
        response = client.get("/search?document_id=doc-1&query=privacy")

    assert response.status_code == 200
    results = response.get_json()["results"]
    assert results[0]["text"] == "from text field"
    assert "from chunk_text field" not in results[0]["text"]
