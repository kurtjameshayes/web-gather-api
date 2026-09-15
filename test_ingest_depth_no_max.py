"""POST /ingest does not clamp depth (unlike POST /crawl, which caps at 10).

Open PR #205 pins crawl depth<=10 / ingest breadth<=100 and invalid types.
Ingest still accepts depth=999: Firecrawl is called with url + limit only
(no max_discovery_depth), matching the leftover contract in #216 kwargs
without asserting the missing depth cap.
"""
from __future__ import annotations

import sys
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

sys.modules["sentence_transformers"] = MagicMock()
sys.modules["pyppeteer"] = MagicMock()
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


def test_ingest_accepts_depth_above_crawl_max(client, mock_clients) -> None:
    """depth=999 is not 400; Firecrawl still receives only url and limit=breadth."""
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
                "depth": 999,
                "breadth": 7,
            },
        )

    assert response.status_code == 200
    mock_clients["firecrawl"].crawl.assert_called_once_with(
        url="https://example.com",
        limit=7,
    )
    assert "max_discovery_depth" not in mock_clients["firecrawl"].crawl.call_args.kwargs
