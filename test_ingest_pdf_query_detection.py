"""Pin /ingest PDF detection that is not a path suffix or a HEAD Content-Type.

Path suffixes and HEAD application/pdf are covered elsewhere. is_pdf_url also
treats any query string containing "pdf" as a PDF, including a larger token
such as "notpdf", and it ignores the URL fragment.
"""
from __future__ import annotations

import sys
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("pyppeteer", MagicMock())

from core import WEB_GATHER_DB, core_bp, init_core


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


def _wire_mongo(mock_mongo: MagicMock) -> None:
    db = MagicMock()
    wg_db = MagicMock()

    def _get_db(name: str) -> MagicMock:
        return wg_db if name == WEB_GATHER_DB else db

    mock_mongo.__getitem__.side_effect = _get_db
    db.__getitem__.return_value = MagicMock()
    wg_db.__getitem__.return_value = MagicMock()


def _pdf_patches():
    return (
        patch("core.requests.head"),
        patch("core.detect_content_type", return_value="text/html"),
        patch("core.download_pdf", return_value=b"%PDF-1.4"),
        patch(
            "core.extract_text_from_pdf",
            return_value=[{"page_number": 1, "text": "Policy PDF text"}],
        ),
        patch("core.combine_pdf_pages", return_value="Policy PDF text"),
    )


def test_ingest_query_substring_pdf_downloads_without_head(client, mock_clients) -> None:
    _wire_mongo(mock_clients["mongo"])
    url = "https://cdn.example/privacy?topic=notpdf"
    head, detect, download, extract, _combine = _pdf_patches()
    with head as head_mock, detect as detect_mock, download as download_mock, extract:
        response = client.post(
            "/ingest",
            json={"url": url, "database": "test_db", "collection": "docs"},
        )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["document_type"] == "pdf"
    assert payload["pages"] == 1
    download_mock.assert_called_once_with(url)
    head_mock.assert_not_called()
    detect_mock.assert_not_called()
    mock_clients["firecrawl"].crawl.assert_not_called()


def test_ingest_query_pdf_match_is_case_insensitive(client, mock_clients) -> None:
    _wire_mongo(mock_clients["mongo"])
    url = "https://cdn.example/privacy?file=PDF"
    head, detect, download, extract, _combine = _pdf_patches()
    with head as head_mock, detect as detect_mock, download as download_mock, extract:
        response = client.post(
            "/ingest",
            json={"url": url, "database": "test_db", "collection": "docs"},
        )

    assert response.status_code == 200
    assert response.get_json()["document_type"] == "pdf"
    download_mock.assert_called_once_with(url)
    head_mock.assert_not_called()
    detect_mock.assert_not_called()
    mock_clients["firecrawl"].crawl.assert_not_called()


def test_ingest_fragment_pdf_does_not_skip_content_type_check(client, mock_clients) -> None:
    _wire_mongo(mock_clients["mongo"])
    url = "https://cdn.example/privacy#pdf"
    mock_clients["firecrawl"].crawl.return_value = [
        {"url": url, "title": "Privacy", "markdown": "Hello policy"}
    ]
    with patch("core.detect_content_type", return_value="text/html") as detect, patch(
        "core.download_pdf"
    ) as download:
        response = client.post(
            "/ingest",
            json={"url": url, "database": "test_db", "collection": "docs"},
        )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["document_type"] == "web"
    detect.assert_called_once_with(url)
    download.assert_not_called()
    mock_clients["firecrawl"].crawl.assert_called_once()
    assert mock_clients["firecrawl"].crawl.call_args.kwargs["url"] == url
