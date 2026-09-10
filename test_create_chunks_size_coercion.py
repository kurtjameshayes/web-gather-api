"""POST /create-chunks integer coercion for chunk_size and overlap."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest
from flask import Flask

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("pyppeteer", MagicMock())
sys.modules.setdefault("pypdf", MagicMock())

from core import core_bp, init_core


@pytest.fixture
def client():
    init_core(MagicMock(), MagicMock(), MagicMock())
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app.test_client()


_REQUIRED = {
    "database": "privacy-compliance",
    "source_collection": "documents",
    "destination_collection": "chunks",
    "source_column": "text",
    "chunk_column": "chunk_text",
}


def test_null_chunk_size_is_400(client) -> None:
    """Explicit JSON null is not the omitted default; int(None) is 400."""
    response = client.post("/create-chunks", json={**_REQUIRED, "chunk_size": None})
    assert response.status_code == 400
    assert "chunk_size and overlap must be integers" in response.get_json()["error"]


def test_non_integer_overlap_is_400(client) -> None:
    """Non-numeric overlap strings fail before any source read."""
    response = client.post("/create-chunks", json={**_REQUIRED, "overlap": "abc"})
    assert response.status_code == 400
    assert "chunk_size and overlap must be integers" in response.get_json()["error"]
