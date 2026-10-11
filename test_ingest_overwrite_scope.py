"""POST /ingest overwrite scope and PDF content-type routing.

Overwrite must delete only rows for the requested source URL. A HEAD
content type decides whether the URL is parsed as a PDF.
"""
from __future__ import annotations

import sys
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

sys.modules["sentence_transformers"] = MagicMock()

from core import DOCUMENTS_COLLECTION, WEB_GATHER_DB, core_bp, detect_content_type, init_core


@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app.test_client()


@pytest.fixture
def mongo() -> Dict[str, Any]:
    mock_mongo = MagicMock()
    mock_firecrawl = MagicMock()
    init_core(mock_mongo, mock_firecrawl, MagicMock())
    user_db = MagicMock()
    wg_db = MagicMock()
    user_coll = MagicMock()
    wg_coll = MagicMock()

    def _get_db(name: str) -> MagicMock:
        return wg_db if name == WEB_GATHER_DB else user_db

    mock_mongo.__getitem__.side_effect = _get_db
    user_db.__getitem__.return_value = user_coll
    wg_db.__getitem__.return_value = wg_coll
    return {
        "firecrawl": mock_firecrawl,
        "user_db": user_db,
        "wg_db": wg_db,
        "user_coll": user_coll,
        "wg_coll": wg_coll,
    }


def _web_page() -> dict:
    return {
        "url": "https://example.com/policy",
        "title": "Privacy",
        "markdown": "We retain data.",
    }


def _post_web(client, mongo, **extra: Any):
    mongo["firecrawl"].crawl.return_value = [_web_page()]
    body = {
        "url": "https://example.com/policy",
        "database": "test_db",
        "collection": "docs",
    }
    body.update(extra)
    with patch("core.detect_content_type", return_value="text/html"):
        return client.post("/ingest", json=body)


def test_overwrite_deletes_only_the_requested_source_url(client, mongo) -> None:
    mongo["user_coll"].count_documents.return_value = 2

    response = _post_web(client, mongo, mode="overwrite")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["mode"] == "overwrite"
    assert payload["document_type"] == "web"
    assert payload["overwritten"] is True
    assert payload["previous_document_count"] == 2

    source_filter = {"source_url": "https://example.com/policy"}
    mongo["user_coll"].count_documents.assert_called_once_with(source_filter)
    mongo["user_coll"].delete_many.assert_called_once_with(source_filter)
    mongo["wg_coll"].delete_many.assert_called_once_with(
        {
            "database_name": "test_db",
            "collection_name": "docs",
            "source_url": "https://example.com/policy",
        }
    )
    mongo["user_db"].__getitem__.assert_called_with("docs")
    mongo["wg_db"].__getitem__.assert_called_with(DOCUMENTS_COLLECTION)

    stored = mongo["user_coll"].insert_one.call_args.args[0]
    assert stored["source_url"] == "https://example.com/policy"
    assert stored["document_type"] == "web"
    assert stored["text"] == "Privacy | Source: https://example.com/policy\nWe retain data."
    metadata = mongo["wg_coll"].insert_one.call_args.args[0]
    assert metadata["document_id"] == stored["_id"]
    assert metadata["source_url"] == "https://example.com/policy"


def test_overwrite_with_no_existing_rows_does_not_delete(client, mongo) -> None:
    mongo["user_coll"].count_documents.return_value = 0

    response = _post_web(client, mongo, mode="OVERWRITE")

    payload = response.get_json()
    assert response.status_code == 200
    assert payload["mode"] == "overwrite"
    assert payload["overwritten"] is False
    assert payload["previous_document_count"] == 0
    mongo["user_coll"].delete_many.assert_not_called()
    mongo["wg_coll"].delete_many.assert_not_called()
    mongo["user_coll"].insert_one.assert_called_once()


def test_append_does_not_count_or_delete(client, mongo) -> None:
    response = _post_web(client, mongo)

    payload = response.get_json()
    assert response.status_code == 200
    assert payload["mode"] == "append"
    assert "overwritten" not in payload
    assert "previous_document_count" not in payload
    mongo["user_coll"].count_documents.assert_not_called()
    mongo["user_coll"].delete_many.assert_not_called()
    mongo["wg_coll"].delete_many.assert_not_called()


def test_padded_overwrite_mode_is_rejected_before_crawl(client, mongo) -> None:
    response = client.post(
        "/ingest",
        json={
            "url": "https://example.com/policy",
            "database": "test_db",
            "collection": "docs",
            "mode": " overwrite ",
        },
    )

    assert response.status_code == 400
    assert "mode" in response.get_json()["error"]
    mongo["firecrawl"].crawl.assert_not_called()
    mongo["user_coll"].delete_many.assert_not_called()


def test_pdf_content_type_downloads_and_overwrite_stays_scoped_to_source_url(client, mongo) -> None:
    mongo["user_coll"].count_documents.return_value = 1
    url = "https://cdn.example.com/download"
    with patch("core.detect_content_type", return_value="application/pdf; charset=binary"), patch(
        "core.download_pdf", return_value=b"%PDF"
    ) as download, patch(
        "core.extract_text_from_pdf",
        return_value=[{"page_number": 1, "text": "Clause"}],
    ), patch(
        "core.combine_pdf_pages", return_value="Clause"
    ):
        response = client.post(
            "/ingest",
            json={
                "url": url,
                "database": "test_db",
                "collection": "docs",
                "mode": "overwrite",
            },
        )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["document_type"] == "pdf"
    assert payload["overwritten"] is True
    assert payload["previous_document_count"] == 1
    download.assert_called_once_with(url)
    mongo["firecrawl"].crawl.assert_not_called()
    mongo["user_coll"].delete_many.assert_called_once_with({"source_url": url})
    stored = mongo["user_coll"].insert_one.call_args.args[0]
    assert stored["document_type"] == "pdf"
    assert stored["text"] == "Clause"


def test_x_pdf_content_type_is_crawled_as_a_web_page(client, mongo) -> None:
    mongo["firecrawl"].crawl.return_value = [_web_page()]
    with patch("core.detect_content_type", return_value="application/x-pdf"), patch(
        "core.download_pdf"
    ) as download:
        response = client.post(
            "/ingest",
            json={
                "url": "https://cdn.example.com/download",
                "database": "test_db",
                "collection": "docs",
            },
        )

    assert response.status_code == 200
    assert response.get_json()["document_type"] == "web"
    download.assert_not_called()
    mongo["firecrawl"].crawl.assert_called_once()


def test_detect_content_type_lowercases_header_and_blank_header_is_empty() -> None:
    pdf = MagicMock()
    pdf.headers = {"Content-Type": "Application/PDF"}
    blank = MagicMock()
    blank.headers = {}
    with patch("core.requests.head", side_effect=[pdf, blank]) as head:
        assert detect_content_type("https://cdn.example.com/a") == "application/pdf"
        assert detect_content_type("https://cdn.example.com/b") == ""
    assert head.call_args_list[0].args == ("https://cdn.example.com/a",)
    assert head.call_args_list[0].kwargs == {"allow_redirects": True, "timeout": 10}
