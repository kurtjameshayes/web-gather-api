"""Regression tests for POST /gather search and next-step contracts.

Base tests only check that a score exists and next_step.endpoint is /ingest.
These pin Firecrawl search bounds, next-step parameter inventory, empty
results, and the current whitespace-query contract.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest
from flask import Flask

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("pyppeteer", MagicMock())

from core import core_bp, init_core  # noqa: E402


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


def test_gather_calls_firecrawl_search_with_limit_10(client, mock_clients) -> None:
    """POST /gather currently hardcodes Firecrawl search limit=10."""
    mock_clients["firecrawl"].search.return_value = [
        {
            "url": "https://example.com/privacy",
            "title": "Privacy Policy",
            "description": "Our privacy policy describes data retention.",
        }
    ]

    response = client.post("/gather", json={"query": "privacy policy"})

    assert response.status_code == 200
    mock_clients["firecrawl"].search.assert_called_once_with(
        query="privacy policy",
        limit=10,
    )
    payload = response.get_json()
    result = payload["results"][0]
    assert result["url"] == "https://example.com/privacy"
    assert result["score"] > 0
    assert result["percent_match"] == round(result["score"] * 100, 1)


def test_gather_next_step_lists_ingest_required_and_optional_params(
    client, mock_clients
) -> None:
    """next_step is the gather->ingest handoff; required/optional keys must stay stable."""
    mock_clients["firecrawl"].search.return_value = []

    response = client.post("/gather", json={"query": "ccpa"})

    assert response.status_code == 200
    next_step = response.get_json()["next_step"]
    assert next_step["endpoint"] == "/ingest"
    assert next_step["method"] == "POST"
    assert set(next_step["required_parameters"]) == {"url", "database", "collection"}
    assert set(next_step["optional_parameters"]) == {
        "depth",
        "breadth",
        "mode",
        "index_database",
        "index_collection",
    }
    assert next_step["optional_parameters"]["mode"]["enum"] == ["append", "overwrite"]
    assert next_step["optional_parameters"]["mode"]["default"] == "append"


def test_gather_empty_results_still_include_query_and_next_step(
    client, mock_clients
) -> None:
    """An empty Firecrawl response is 200 with results=[], not a 500."""
    mock_clients["firecrawl"].search.return_value = SimpleNamespace(other="ignored")

    response = client.post("/gather", json={"query": "no hits"})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["query"] == "no hits"
    assert payload["results"] == []
    assert payload["next_step"]["endpoint"] == "/ingest"


def test_gather_whitespace_query_is_currently_accepted(client, mock_clients) -> None:
    """Whitespace-only query is truthy, so it is not treated as missing."""
    mock_clients["firecrawl"].search.return_value = []

    response = client.post("/gather", json={"query": "   "})

    assert response.status_code == 200
    mock_clients["firecrawl"].search.assert_called_once_with(query="   ", limit=10)
