"""Pin how crawl and ingest forward depth, and how PDF URLs skip HEAD.

POST /crawl clamps depth and breadth, then passes both to every browser crawler
and to the Firecrawl fallback as limit plus max_discovery_depth.

POST /ingest lowercases mode, clamps only breadth, and calls Firecrawl with
limit alone. Depth is accepted (including values crawl would cap) and is not
forwarded. A path ending in .pdf is downloaded without a HEAD request. A path
that merely contains "pdf" is not.
"""
from __future__ import annotations

import sys
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
import requests
from flask import Flask

sys.modules.setdefault("sentence_transformers", MagicMock())

from core import DOCUMENTS_COLLECTION, WEB_GATHER_DB, core_bp, init_core
import security


@pytest.fixture
def clients() -> Dict[str, Any]:
    previous_key = security._app_api_key
    security._app_api_key = None
    mock_mongo = MagicMock()
    mock_firecrawl = MagicMock()
    mock_anthropic = MagicMock()
    init_core(mock_mongo, mock_firecrawl, mock_anthropic)
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    yield {
        "client": app.test_client(),
        "mongo": mock_mongo,
        "firecrawl": mock_firecrawl,
    }
    security._app_api_key = previous_key


def _wire_mongo(mock_mongo: MagicMock) -> Dict[str, MagicMock]:
    target = MagicMock()
    wg = MagicMock()
    target_coll = MagicMock()
    docs_coll = MagicMock()

    def _get_db(name: str) -> MagicMock:
        return wg if name == WEB_GATHER_DB else target

    def _wg_coll(name: str) -> MagicMock:
        if name == DOCUMENTS_COLLECTION:
            return docs_coll
        return MagicMock()

    mock_mongo.__getitem__.side_effect = _get_db
    target.__getitem__.return_value = target_coll
    wg.__getitem__.side_effect = _wg_coll
    return {"target": target_coll, "documents": docs_coll}


def test_crawl_firecrawl_fallback_receives_clamped_depth(clients) -> None:
    """Depth is capped at 10 before it is forwarded as max_discovery_depth."""
    clients["firecrawl"].crawl.return_value = [
        {"url": "https://example.com/start", "title": "T", "markdown": "Body"}
    ]
    with patch("core.playwright_crawl", return_value=[]) as playwright, patch(
        "core.puppeteer_crawl", return_value=[]
    ) as puppeteer, patch("core.selenium_crawl", return_value=[]) as selenium:
        response = clients["client"].post(
            "/crawl",
            json={"url": "https://example.com/start", "depth": 15, "breadth": 150},
        )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["crawl_method"] == "firecrawl"
    assert payload["depth"] == 10
    assert payload["breadth"] == 100
    playwright.assert_called_once_with("https://example.com/start", 10, 100)
    puppeteer.assert_called_once_with("https://example.com/start", 10, 100)
    selenium.assert_called_once_with("https://example.com/start", 10, 100)
    clients["firecrawl"].crawl.assert_called_once_with(
        url="https://example.com/start",
        limit=100,
        max_discovery_depth=10,
    )


def test_crawl_in_range_depth_is_forwarded_unchanged(clients) -> None:
    clients["firecrawl"].crawl.return_value = [
        {"url": "https://example.com/start", "markdown": "Body"}
    ]
    with patch("core.playwright_crawl", return_value=[]), patch(
        "core.puppeteer_crawl", return_value=[]
    ), patch("core.selenium_crawl", return_value=[]):
        response = clients["client"].post(
            "/crawl",
            json={"url": "https://example.com/start", "depth": 7, "breadth": 4},
        )

    assert response.status_code == 200
    assert response.get_json()["depth"] == 7
    assert response.get_json()["breadth"] == 4
    clients["firecrawl"].crawl.assert_called_once_with(
        url="https://example.com/start",
        limit=4,
        max_discovery_depth=7,
    )


def test_ingest_does_not_forward_depth_even_above_crawl_cap(clients) -> None:
    """Ingest depth has no upper cap and never becomes max_discovery_depth."""
    _wire_mongo(clients["mongo"])
    clients["firecrawl"].crawl.return_value = [
        {"url": "https://example.com/page", "title": "T", "markdown": "Hello"}
    ]
    with patch("core.detect_content_type", return_value="text/html") as head_type:
        response = clients["client"].post(
            "/ingest",
            json={
                "url": "https://example.com/page",
                "database": "test_db",
                "collection": "docs",
                "depth": 999,
                "breadth": 4,
            },
        )

    assert response.status_code == 200
    assert response.get_json()["document_type"] == "web"
    assert response.get_json()["pages"] == 1
    assert "depth" not in response.get_json()
    head_type.assert_called_once_with("https://example.com/page")
    clients["firecrawl"].crawl.assert_called_once_with(
        url="https://example.com/page",
        limit=4,
    )


def test_ingest_omitted_bounds_use_breadth_default_only(clients) -> None:
    _wire_mongo(clients["mongo"])
    clients["firecrawl"].crawl.return_value = [
        {"url": "https://example.com/page", "markdown": "Hello"}
    ]
    with patch("core.detect_content_type", return_value=""):
        response = clients["client"].post(
            "/ingest",
            json={
                "url": "https://example.com/page",
                "database": "test_db",
                "collection": "docs",
            },
        )

    assert response.status_code == 200
    clients["firecrawl"].crawl.assert_called_once_with(
        url="https://example.com/page",
        limit=5,
    )


def test_ingest_non_numeric_depth_is_400_before_crawl(clients) -> None:
    _wire_mongo(clients["mongo"])
    response = clients["client"].post(
        "/ingest",
        json={
            "url": "https://example.com/page",
            "database": "test_db",
            "collection": "docs",
            "depth": "nope",
        },
    )
    assert response.status_code == 400
    assert response.get_json()["error"] == "depth must be a positive integer"
    clients["firecrawl"].crawl.assert_not_called()


def test_ingest_pdf_extension_skips_head_and_stores_truncated_description(clients) -> None:
    colls = _wire_mongo(clients["mongo"])
    url = "https://cdn.example/policies/Notice.PDF"
    page_text = "B" * 350
    response_body = MagicMock()
    response_body.content = b"%PDF-1.4"
    response_body.raise_for_status.return_value = None

    with patch("core.requests.head") as head, patch(
        "core.requests.get", return_value=response_body
    ) as get, patch(
        "core.extract_text_from_pdf",
        return_value=[{"page_number": 2, "text": page_text}],
    ):
        response = clients["client"].post(
            "/ingest",
            json={"url": url, "database": "test_db", "collection": "docs"},
        )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["document_type"] == "pdf"
    assert payload["pages"] == 1
    head.assert_not_called()
    get.assert_called_once_with(url, timeout=60)
    clients["firecrawl"].crawl.assert_not_called()

    stored = colls["target"].insert_one.call_args.args[0]
    meta = colls["documents"].insert_one.call_args.args[0]
    expected_text = f"Page 2 | Source: {url}\n{page_text}"
    assert stored["_id"] == payload["document_id"]
    assert stored["source_url"] == url
    assert stored["document_type"] == "pdf"
    assert stored["text"] == expected_text
    assert meta["document_id"] == payload["document_id"]
    assert meta["description"] == expected_text[:300]
    assert len(meta["description"]) == 300
    assert len(stored["text"]) > 300


def test_ingest_path_containing_pdf_is_not_a_pdf_download(clients) -> None:
    """Path suffix is exact. 'pdf' inside the filename does not skip the crawl."""
    _wire_mongo(clients["mongo"])
    url = "https://cdn.example/policies/notice.pdf.bak"
    clients["firecrawl"].crawl.return_value = [
        {"url": url, "markdown": "page"}
    ]
    with patch("core.requests.head") as head, patch("core.requests.get") as get:
        head.return_value = MagicMock(headers={"Content-Type": "text/html"})
        response = clients["client"].post(
            "/ingest",
            json={"url": url, "database": "test_db", "collection": "docs"},
        )

    assert response.status_code == 200
    assert response.get_json()["document_type"] == "web"
    head.assert_called_once_with(url, allow_redirects=True, timeout=10)
    get.assert_not_called()
    clients["firecrawl"].crawl.assert_called_once_with(url=url, limit=5)


def test_ingest_pdf_http_error_is_download_failure(clients) -> None:
    colls = _wire_mongo(clients["mongo"])
    with patch("core.download_pdf", side_effect=requests.HTTPError("404")):
        response = clients["client"].post(
            "/ingest",
            json={
                "url": "https://cdn.example/file.pdf",
                "database": "test_db",
                "collection": "docs",
            },
        )

    assert response.status_code == 500
    assert response.get_json()["error"] == (
        "Failed to download PDF. The URL may be unreachable."
    )
    colls["target"].insert_one.assert_not_called()
    clients["firecrawl"].crawl.assert_not_called()


def test_ingest_pdf_parse_error_is_distinct_from_download_failure(clients) -> None:
    colls = _wire_mongo(clients["mongo"])
    with patch("core.download_pdf", return_value=b"%PDF"), patch(
        "core.extract_text_from_pdf", side_effect=ValueError("corrupt")
    ):
        response = clients["client"].post(
            "/ingest",
            json={
                "url": "https://cdn.example/file.pdf",
                "database": "test_db",
                "collection": "docs",
            },
        )

    assert response.status_code == 500
    assert response.get_json()["error"] == (
        "Failed to parse PDF. The file may be corrupted or unsupported."
    )
    colls["target"].insert_one.assert_not_called()
