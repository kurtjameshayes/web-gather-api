"""Regression tests for /ingest request validation and PDF error mapping."""
from __future__ import annotations

import sys
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
import requests
from flask import Flask

sys.modules.setdefault("sentence_transformers", MagicMock())

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


@pytest.mark.parametrize(
    "payload",
    [
        {"database": "test_db", "collection": "docs"},
        {"url": "https://example.com", "collection": "docs"},
        {"url": "https://example.com", "database": "test_db"},
        {},
    ],
)
def test_ingest_missing_required_params_returns_400(client, mock_clients, payload: dict) -> None:
    """url, database, and collection are all required before any crawl or write."""
    response = client.post("/ingest", json=payload)

    assert response.status_code == 400
    assert "url, database, and collection are required" in response.get_json()["error"]
    mock_clients["firecrawl"].crawl.assert_not_called()
    mock_clients["mongo"].__getitem__.assert_not_called()


def test_ingest_pdf_download_failure_returns_500_without_writes(client, mock_clients) -> None:
    """RequestException while downloading a PDF maps to 500 and must not write."""
    with patch("core.is_pdf_url", return_value=True), patch(
        "core.download_pdf", side_effect=requests.RequestException("timeout")
    ):
        response = client.post(
            "/ingest",
            json={
                "url": "https://example.com/file.pdf",
                "database": "test_db",
                "collection": "docs",
            },
        )

    assert response.status_code == 500
    assert "Failed to download PDF" in response.get_json()["error"]
    mock_clients["mongo"].__getitem__.assert_not_called()


def test_ingest_pdf_parse_failure_returns_500_without_writes(client, mock_clients) -> None:
    """Non-network PDF parse errors map to a distinct 500 and must not write."""
    with patch("core.is_pdf_url", return_value=True), patch(
        "core.download_pdf", return_value=b"%PDF-1.0"
    ), patch(
        "core.extract_text_from_pdf", side_effect=ValueError("corrupted")
    ):
        response = client.post(
            "/ingest",
            json={
                "url": "https://example.com/file.pdf",
                "database": "test_db",
                "collection": "docs",
            },
        )

    assert response.status_code == 500
    assert "Failed to parse PDF" in response.get_json()["error"]
    mock_clients["mongo"].__getitem__.assert_not_called()
