"""Search ranking uses a raw dot product and then clamps the displayed percent.

Stored chunk vectors are not renormalized. Unit-vector fixtures cannot tell a
true cosine from that dot product, and they never exercise percent clamping or
the five-result cap.
"""
from __future__ import annotations

import sys
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from flask import Flask

sys.modules["sentence_transformers"] = MagicMock()

from core import WEB_GATHER_DB, core_bp, cosine_similarity, init_core


class _QueryModel:
    """Return a unit query so only the stored chunk vectors affect magnitude."""

    def encode(self, inputs, **kwargs: object) -> np.ndarray:
        return np.array([[1.0, 0.0]])


@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app.test_client()


@pytest.fixture
def mock_clients() -> Dict[str, Any]:
    mock_mongo = MagicMock()
    init_core(mock_mongo, MagicMock(), MagicMock())
    return {"mongo": mock_mongo}


def _wire_chunks(mock_mongo: MagicMock, chunks: List[dict]) -> None:
    index_db = MagicMock()
    wg_db = MagicMock()

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


def _search(client, mock_clients):
    with patch("core.get_embedding_model_name", return_value="model"), patch(
        "core.get_model", return_value=_QueryModel()
    ):
        return client.get("/search?document_id=doc-1&query=privacy")


def test_cosine_similarity_keeps_vector_magnitude() -> None:
    """A true cosine would return 1 and -1. This helper returns the dot product."""
    query = np.array([2.0, 0.0])
    scores = cosine_similarity(
        query,
        [np.array([3.0, 0.0]), np.array([-1.0, 0.0])],
    )
    assert scores.tolist() == [6.0, -2.0]


def test_search_clamps_percent_and_keeps_negative_scores(client, mock_clients) -> None:
    _wire_chunks(
        mock_clients["mongo"],
        [
            {"text": "opposed", "embedding": [-2.0, 0.0]},
            {"text": "oversized", "embedding": [4.0, 0.0]},
        ],
    )

    response = _search(client, mock_clients)

    assert response.status_code == 200
    results = response.get_json()["results"]
    assert [item["text"] for item in results] == ["oversized", "opposed"]
    assert results[0]["score"] == 4.0
    assert results[0]["percent_match"] == 100.0
    assert results[1]["score"] == -2.0
    assert results[1]["percent_match"] == 0.0


def test_search_returns_five_results_ordered_by_raw_score(client, mock_clients) -> None:
    _wire_chunks(
        mock_clients["mongo"],
        [
            {"text": "opposed", "embedding": [-2.0, 0.0]},
            {"text": "orthogonal", "embedding": [0.0, 3.0]},
            {"text": "oversized", "embedding": [4.0, 0.0]},
            {"text": "aligned", "embedding": [1.0, 0.0]},
            {"text": "partial", "embedding": [0.5, 0.0]},
            {"text": "weak", "embedding": [0.25, 0.0]},
        ],
    )

    response = _search(client, mock_clients)

    assert response.status_code == 200
    results = response.get_json()["results"]
    assert [(item["text"], item["score"], item["percent_match"]) for item in results] == [
        ("oversized", 4.0, 100.0),
        ("aligned", 1.0, 100.0),
        ("partial", 0.5, 50.0),
        ("weak", 0.25, 25.0),
        ("orthogonal", 0.0, 0.0),
    ]
