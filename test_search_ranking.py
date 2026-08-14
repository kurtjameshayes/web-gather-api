"""Regression tests for GET /search ranking, clamping, and model selection.

Base tests cover missing params / not-found / success of the top hit. Ranking
order, percent_match clamps, top-5 truncation, and privacy-compliance model
preference were still untested on v0_1.
"""
from __future__ import annotations

import sys
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from flask import Flask

sys.modules["sentence_transformers"] = MagicMock()
sys.modules.setdefault("pyppeteer", MagicMock())

from core import PRIVACY_COMPLIANCE_DB, WEB_GATHER_DB, core_bp, init_core


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


def _wire_search_mongo(
    mock_mongo: MagicMock, *, index_db_name: str, chunks: List[Dict[str, Any]]
) -> None:
    wg_db = MagicMock()
    index_db = MagicMock()
    other_db = MagicMock()

    def _get_db(name: str) -> MagicMock:
        if name == WEB_GATHER_DB:
            return wg_db
        if name == index_db_name:
            return index_db
        return other_db

    mock_mongo.__getitem__.side_effect = _get_db
    wg_db.__getitem__.return_value.find_one.return_value = {
        "document_id": "doc-1",
        "database_name": index_db_name,
        "collection_name": "docs",
        "chunk_collection": "chunks",
        "index_database_name": index_db_name,
    }
    index_db.__getitem__.return_value.find.return_value = chunks


class _QueryModel:
    def encode(self, inputs, **kwargs: object) -> np.ndarray:
        return np.array([[1.0, 0.0]])


def test_search_ranks_clamps_and_returns_top_five(client, mock_clients) -> None:
    """Scores are sorted desc, percent_match is clamped to 0–100, and only 5 hits return."""
    chunks = [
        {"text": "weak", "embedding": [0.1, 0.0]},
        {"text": "over-one", "embedding": [2.0, 0.0]},
        {"text": "mid", "embedding": [0.5, 0.0]},
        {"text": "neg", "embedding": [-1.0, 0.0]},
        {"text": "zero", "embedding": [0.0, 1.0]},
        {"text": "sixth", "embedding": [0.05, 0.0]},
        {"text": "strong", "embedding": [0.9, 0.0]},
    ]
    _wire_search_mongo(mock_clients["mongo"], index_db_name="index_db", chunks=chunks)

    with patch("core.get_embedding_model_name", return_value="model"), patch(
        "core.get_model", return_value=_QueryModel()
    ):
        response = client.get("/search?document_id=doc-1&query=retention")

    assert response.status_code == 200
    results = response.get_json()["results"]
    assert len(results) == 5
    assert [row["text"] for row in results] == [
        "over-one",
        "strong",
        "mid",
        "weak",
        "sixth",
    ]
    assert results[0]["percent_match"] == 100.0
    assert results[0]["score"] == 2.0
    assert results[2]["percent_match"] == 50.0
    # Negative/zero scores are truncated out of the top 5; clamping is still the contract.
    assert all(0.0 <= row["percent_match"] <= 100.0 for row in results)


def test_search_privacy_compliance_prefers_application_model(client, mock_clients) -> None:
    """privacy-compliance searches must use the application embedding model when set."""
    chunks = [{"text": "match", "embedding": [1.0, 0.0]}]
    _wire_search_mongo(
        mock_clients["mongo"], index_db_name=PRIVACY_COMPLIANCE_DB, chunks=chunks
    )
    chosen: list[str] = []

    def _get_model(name: str) -> _QueryModel:
        chosen.append(name)
        return _QueryModel()

    with patch("core.get_application_embedding_model", return_value="app-model"), patch(
        "core.get_embedding_model_name", return_value="db-model"
    ), patch("core.get_model", side_effect=_get_model):
        response = client.get("/search?document_id=doc-1&query=deletion")

    assert response.status_code == 200
    assert chosen == ["app-model"]
    assert response.get_json()["results"][0]["text"] == "match"
