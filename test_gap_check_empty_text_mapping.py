"""Regression tests for /gap-check remapping of empty-document fetch errors.

_fetch_document_text returns HTTP 404 for both a missing document and a
document with no extractable text. The route currently maps every 404 to
"Statute/Policy document not found", so an existing empty document is
indistinguishable from a missing one. Pin that composition; do not patch
the helper.
"""
from __future__ import annotations

import sys
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest
from flask import Flask

sys.modules["sentence_transformers"] = MagicMock()
sys.modules["pyppeteer"] = MagicMock()

from core import core_bp, init_core


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


@pytest.fixture
def client(mock_clients) -> Any:
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app.test_client()


def _gap_payload() -> dict:
    return {
        "policy_database_name": "pdb",
        "policy_collection_name": "policies",
        "policy_document_id": "pol-1",
        "statute_database_name": "sdb",
        "statute_collection_name": "statutes",
        "statute_document_id": "stat-1",
    }


def _wire_collections(mock_mongo: MagicMock) -> Dict[str, MagicMock]:
    statute_coll = MagicMock(name="statute_coll")
    policy_coll = MagicMock(name="policy_coll")
    statute_db = MagicMock(name="statute_db")
    policy_db = MagicMock(name="policy_db")
    statute_db.__getitem__.return_value = statute_coll
    policy_db.__getitem__.return_value = policy_coll

    def _get_db(name: str) -> MagicMock:
        return statute_db if name == "sdb" else policy_db

    mock_mongo.__getitem__.side_effect = _get_db
    return {"statute_coll": statute_coll, "policy_coll": policy_coll}


def test_gap_check_existing_statute_without_text_is_reported_not_found(
    client, mock_clients
) -> None:
    """A statute document that exists but has no text currently surfaces as not found."""
    colls = _wire_collections(mock_clients["mongo"])
    colls["statute_coll"].find_one.return_value = {
        "document_id": "stat-1",
        "policy_chunks": [],
    }

    response = client.post("/gap-check", json=_gap_payload())

    assert response.status_code == 404
    assert response.get_json()["error"] == "Statute document not found"
    mock_clients["anthropic"].messages.create.assert_not_called()
    colls["policy_coll"].find_one.assert_not_called()


def test_gap_check_existing_policy_without_text_is_reported_not_found(
    client, mock_clients
) -> None:
    """After a valid statute, an empty policy document is also remapped to not found."""
    colls = _wire_collections(mock_clients["mongo"])
    colls["statute_coll"].find_one.return_value = {
        "document_id": "stat-1",
        "text": "Consumers have the right to delete personal information.",
    }
    colls["policy_coll"].find_one.return_value = {
        "document_id": "pol-1",
        "text": "   ",
        "chunk_text": "",
        "section_text": None,
        "policy_chunks": [{"chunk_text": "  "}],
    }

    response = client.post("/gap-check", json=_gap_payload())

    assert response.status_code == 404
    assert response.get_json()["error"] == "Policy document not found"
    mock_clients["anthropic"].messages.create.assert_not_called()
