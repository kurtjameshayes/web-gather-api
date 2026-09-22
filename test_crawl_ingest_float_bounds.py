"""POST /crawl and /ingest accept JSON float depth/breadth via int() truncation.

#205 pins integer clamps and non-integer string 400s. #238 pins JSON null as
400. Floats still pass int() (2.7 -> 2, 50.5 -> 50) and then the same clamps,
so callers can silently crawl/ingest with a different bound than requested.
"""
from __future__ import annotations

import sys
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("pyppeteer", MagicMock())
sys.modules.setdefault("pypdf", MagicMock())

from core import WEB_GATHER_DB, core_bp, init_core


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
def client(mock_clients):
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app.test_client()


def _wire_mongo(mock_mongo: MagicMock) -> None:
    db = MagicMock()
    wg_db = MagicMock()

    def _get_db(name: str) -> MagicMock:
        return wg_db if name == WEB_GATHER_DB else db

    mock_mongo.__getitem__.side_effect = _get_db
    db.__getitem__.return_value = MagicMock()
    wg_db.__getitem__.return_value = MagicMock()


def test_crawl_float_depth_and_breadth_are_truncated(client, mock_clients) -> None:
    """JSON floats are truncated by int() then clamped (2.7 -> 2, 150.9 -> 100)."""
    with patch(
        "core.playwright_crawl",
        return_value=[{"url": "https://example.com", "title": "T", "markdown": "Body"}],
    ) as playwright_crawl:
        response = client.post(
            "/crawl",
            json={"url": "https://example.com", "depth": 2.7, "breadth": 150.9},
        )

    assert response.status_code == 200
    playwright_crawl.assert_called_once_with("https://example.com", 2, 100)


def test_ingest_float_depth_and_breadth_are_truncated(client, mock_clients) -> None:
    """Ingest truncates float depth/breadth; breadth still clamps to 100 for Firecrawl."""
    _wire_mongo(mock_clients["mongo"])
    mock_clients["firecrawl"].crawl.return_value = [
        {"url": "https://example.com", "title": "T", "markdown": "Hello"}
    ]

    with patch("core.detect_content_type", return_value=""):
        response = client.post(
            "/ingest",
            json={
                "url": "https://example.com",
                "database": "test_db",
                "collection": "docs",
                "depth": 9.8,
                "breadth": 50.5,
            },
        )

    assert response.status_code == 200
    # Ingest does not pass depth to Firecrawl; breadth becomes limit after int()+clamp.
    mock_clients["firecrawl"].crawl.assert_called_with(
        url="https://example.com",
        limit=50,
    )
