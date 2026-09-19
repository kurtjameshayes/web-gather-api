"""POST /ingest treats a HEAD failure as 'not a PDF' and crawls as web.

detect_content_type swallows requests exceptions and returns \"\". URLs
without a .pdf path / 'pdf' query then take the Firecrawl path instead of
download_pdf. Distinct from #213 (HEAD application/pdf) and #227 (query
substring 'pdf' skips HEAD).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests
from flask import Flask

from core import core_bp, init_core
from conftest import _wire_mongo


@pytest.fixture
def mock_clients():
    mock_mongo = MagicMock()
    mock_firecrawl = MagicMock()
    init_core(mock_mongo, mock_firecrawl, MagicMock())
    return {"mongo": mock_mongo, "firecrawl": mock_firecrawl}


@pytest.fixture
def client(mock_clients):
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app.test_client()


def test_ingest_head_exception_uses_web_crawl(client, mock_clients) -> None:
    """HEAD ConnectionError is non-fatal; Firecrawl runs and PDF download does not."""
    _wire_mongo(mock_clients["mongo"])
    mock_clients["firecrawl"].crawl.return_value = [
        {"url": "https://example.com/policy", "title": "Policy", "markdown": "Hello"}
    ]

    with patch("core.requests.head", side_effect=requests.ConnectionError("down")), patch(
        "core.download_pdf"
    ) as download_pdf:
        response = client.post(
            "/ingest",
            json={
                "url": "https://example.com/policy",
                "database": "test_db",
                "collection": "docs",
            },
        )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["document_type"] == "web"
    mock_clients["firecrawl"].crawl.assert_called_once()
    download_pdf.assert_not_called()


def test_detect_content_type_returns_empty_on_head_error() -> None:
    """Helper itself returns \"\" when HEAD raises, so ingest cannot see the error."""
    from core import detect_content_type

    with patch("core.requests.head", side_effect=requests.Timeout("slow")):
        assert detect_content_type("https://example.com/policy") == ""
