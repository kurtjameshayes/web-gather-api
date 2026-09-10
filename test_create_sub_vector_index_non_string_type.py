"""POST /create-sub-vector-index non-string document_type validation."""
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


def test_numeric_document_type_is_required_error(client) -> None:
    """Non-strings use the 'required and must be policy or statute' message, not the enum-only message."""
    response = client.post("/create-sub-vector-index", json={"document_type": 1})
    assert response.status_code == 400
    err = response.get_json()["error"]
    assert "required" in err
    assert "policy" in err and "statute" in err


def test_boolean_document_type_is_required_error(client) -> None:
    """JSON true is not stripped/lowercased into a valid document_type."""
    response = client.post("/create-sub-vector-index", json={"document_type": True})
    assert response.status_code == 400
    err = response.get_json()["error"]
    assert "required" in err
    assert "must be 'policy' or 'statute'" in err
