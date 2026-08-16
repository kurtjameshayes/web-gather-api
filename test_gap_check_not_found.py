"""Regression tests for /gap-check 404 mapping, truncation, and LLM failures.

Base tests cover missing params and a happy-path parse. Open coverage PRs
cover fetch-helper internals and parse-retry, not these route contracts.
"""
from __future__ import annotations

import sys
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

sys.modules["sentence_transformers"] = MagicMock()
sys.modules["pyppeteer"] = MagicMock()

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


def _gap_payload() -> dict:
    return {
        "policy_database_name": "pdb",
        "policy_collection_name": "policies",
        "policy_document_id": "pol-1",
        "statute_database_name": "sdb",
        "statute_collection_name": "statutes",
        "statute_document_id": "stat-1",
    }


def test_gap_check_statute_not_found(client, mock_clients) -> None:
    with patch("core._fetch_document_text") as mock_fetch:
        mock_fetch.return_value = (None, (MagicMock(), 404))
        response = client.post("/gap-check", json=_gap_payload())

    assert response.status_code == 404
    assert response.get_json()["error"] == "Statute document not found"
    assert mock_fetch.call_count == 1


def test_gap_check_policy_not_found(client, mock_clients) -> None:
    with patch("core._fetch_document_text") as mock_fetch:
        mock_fetch.side_effect = [
            ("Statute text stays available.", None),
            (None, (MagicMock(), 404)),
        ]
        response = client.post("/gap-check", json=_gap_payload())

    assert response.status_code == 404
    assert response.get_json()["error"] == "Policy document not found"
    assert mock_fetch.call_count == 2


def test_gap_check_truncates_statute_start_and_policy_end(client, mock_clients) -> None:
    statute = "STATUTE_START" + ("S" * 4480) + "STATUTE_END"
    policy = "POLICY_START" + ("Q" * 8978) + "POLICY_END"
    captured: dict[str, str] = {}

    def _capture_create(**kwargs):
        captured["prompt"] = kwargs["messages"][0]["content"]
        mock_response = MagicMock()
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = (
            '{"addressed": false, "policy_quote": null, "missing": true, '
            '"conflict": false, "conflict_description": null}'
        )
        mock_response.content = [text_block]
        return mock_response

    with patch("core._fetch_document_text") as mock_fetch:
        mock_fetch.side_effect = [(statute, None), (policy, None)]
        mock_clients["anthropic"].messages.create.side_effect = _capture_create
        response = client.post(
            "/gap-check",
            json={**_gap_payload(), "max_policy_chars": 8000},
        )

    assert response.status_code == 200
    prompt = captured["prompt"]
    # Statute keeps the start (first 4000 chars); policy keeps the end.
    assert "STATUTE_START" in prompt
    assert "STATUTE_END" not in prompt
    assert "POLICY_START" not in prompt
    assert "POLICY_END" in prompt
    assert len(statute) > 4000
    assert len(policy) > 8000


def test_gap_check_llm_exception_returns_500(client, mock_clients) -> None:
    with patch("core._fetch_document_text") as mock_fetch:
        mock_fetch.side_effect = [("statute", None), ("policy", None)]
        mock_clients["anthropic"].messages.create.side_effect = RuntimeError("boom")
        response = client.post("/gap-check", json=_gap_payload())

    assert response.status_code == 500
    assert "LLM parsing failed" in response.get_json()["error"]
