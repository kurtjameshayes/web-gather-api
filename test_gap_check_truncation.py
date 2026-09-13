"""Regression tests for POST /gap-check statute/policy truncation and LLM retry."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("pyppeteer", MagicMock())

from core import core_bp, init_core


@pytest.fixture
def mock_clients():
    mock_mongo = MagicMock()
    mock_firecrawl = MagicMock()
    mock_anthropic = MagicMock()
    init_core(mock_mongo, mock_firecrawl, mock_anthropic)
    return {"mongo": mock_mongo, "firecrawl": mock_firecrawl, "anthropic": mock_anthropic}


@pytest.fixture
def client(mock_clients):
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app.test_client()


def _valid_gap_json() -> str:
    return (
        '{"addressed": false, "policy_quote": null, "missing": true, '
        '"conflict": false, "conflict_description": null}'
    )


def _text_response(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    resp = MagicMock()
    resp.content = [block]
    return resp


def _gap_payload(**overrides):
    body = {
        "policy_database_name": "pdb",
        "policy_collection_name": "policies",
        "policy_document_id": "pol-1",
        "statute_database_name": "sdb",
        "statute_collection_name": "statutes",
        "statute_document_id": "stat-1",
    }
    body.update(overrides)
    return body


def test_gap_check_keeps_statute_start_and_policy_end(client, mock_clients) -> None:
    """Statute is truncated from the start (4000 chars); policy keeps the last N chars."""
    statute = "STATUTE_HEAD" + ("X" * 4200) + "STATUTE_TAIL"
    policy = "POLICY_HEAD" + ("Y" * 200) + "POLICY_TAIL"
    mock_clients["anthropic"].messages.create.return_value = _text_response(_valid_gap_json())

    with patch("core._fetch_document_text") as mock_fetch:
        mock_fetch.side_effect = [(statute, None), (policy, None)]
        response = client.post("/gap-check", json=_gap_payload(max_policy_chars=20))

    assert response.status_code == 200
    prompt = mock_clients["anthropic"].messages.create.call_args.kwargs["messages"][0]["content"]
    assert "STATUTE_HEAD" in prompt
    assert "STATUTE_TAIL" not in prompt
    assert "POLICY_TAIL" in prompt
    assert "POLICY_HEAD" not in prompt
    assert len(statute) > 4000
    # First 4000 of statute is what the route keeps.
    assert statute[:4000] in prompt


def test_gap_check_retries_unparseable_llm_then_succeeds(client, mock_clients) -> None:
    mock_clients["anthropic"].messages.create.side_effect = [
        _text_response("not-json"),
        _text_response(_valid_gap_json()),
    ]

    with patch("core._fetch_document_text") as mock_fetch:
        mock_fetch.side_effect = [("statute body", None), ("policy body", None)]
        response = client.post("/gap-check", json=_gap_payload())

    assert response.status_code == 200
    assert response.get_json()["gap_check"]["missing"] is True
    assert mock_clients["anthropic"].messages.create.call_count == 2
    retry_prompt = mock_clients["anthropic"].messages.create.call_args_list[1].kwargs["messages"][0]["content"]
    assert "IMPORTANT" in retry_prompt
    assert "ONLY a valid JSON object" in retry_prompt


def test_gap_check_returns_500_when_retry_still_unparseable(client, mock_clients) -> None:
    mock_clients["anthropic"].messages.create.side_effect = [
        _text_response("still not json"),
        _text_response("also not json"),
    ]

    with patch("core._fetch_document_text") as mock_fetch:
        mock_fetch.side_effect = [("statute body", None), ("policy body", None)]
        response = client.post("/gap-check", json=_gap_payload())

    assert response.status_code == 500
    assert "Could not parse LLM response" in response.get_json()["error"]
