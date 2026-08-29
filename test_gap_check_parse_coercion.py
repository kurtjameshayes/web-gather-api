"""Regression tests for POST /gap-check LLM JSON coercion.

Pins bool() / empty-string contracts used when shaping gap_check results.
Do not "fix" these in a coverage-only change.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("pyppeteer", MagicMock())
sys.modules.setdefault("pypdf", MagicMock())

from core import core_bp, init_core


@pytest.fixture
def mock_anthropic() -> MagicMock:
    return MagicMock()


@pytest.fixture
def client(mock_anthropic: MagicMock):
    init_core(MagicMock(), MagicMock(), mock_anthropic)
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app.test_client()


def _llm_text(text: str, mock_anthropic: MagicMock) -> None:
    block = MagicMock()
    block.type = "text"
    block.text = text
    mock_anthropic.messages.create.return_value.content = [block]


def _valid_body() -> dict:
    return {
        "policy_database_name": "db",
        "policy_collection_name": "policies",
        "policy_document_id": "pol-1",
        "statute_database_name": "db",
        "statute_collection_name": "statutes",
        "statute_document_id": "stat-1",
    }


def test_gap_check_bool_and_empty_quote_coercion(client, mock_anthropic) -> None:
    """String 'false' is True; empty quotes become None; omitted missing defaults True."""
    _llm_text(
        '{"addressed": "false", "policy_quote": "", "conflict": 0}',
        mock_anthropic,
    )
    with patch("core._fetch_document_text", side_effect=[("statute", None), ("policy", None)]):
        response = client.post("/gap-check", json=_valid_body())

    assert response.status_code == 200
    result = response.get_json()["gap_check"]
    # bool("false") is True — pin the current coercion, do not treat as False.
    assert result["addressed"] is True
    assert result["policy_quote"] is None
    assert result["missing"] is True
    assert result["conflict"] is False
    assert result["conflict_description"] is None
    mock_anthropic.messages.create.assert_called_once()


def test_gap_check_empty_conflict_description_is_null(client, mock_anthropic) -> None:
    """Empty conflict_description is normalized to null like empty policy_quote."""
    _llm_text(
        '{"addressed": false, "policy_quote": "keep this", "missing": false, '
        '"conflict": true, "conflict_description": ""}',
        mock_anthropic,
    )
    with patch("core._fetch_document_text", side_effect=[("statute", None), ("policy", None)]):
        response = client.post("/gap-check", json=_valid_body())

    assert response.status_code == 200
    result = response.get_json()["gap_check"]
    assert result["addressed"] is False
    assert result["policy_quote"] == "keep this"
    assert result["missing"] is False
    assert result["conflict"] is True
    assert result["conflict_description"] is None
