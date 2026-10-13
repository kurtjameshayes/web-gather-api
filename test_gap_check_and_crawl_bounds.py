"""Regression tests for POST /gap-check resolution and crawl content selection.

Covers document lookup, policy-window edge slices, gap-result coercion, and the
browser-versus-Firecrawl split over empty markdown. Integer clamps, float
truncation, null bounds, retry, and basic crawl text joining are covered elsewhere.
"""
from __future__ import annotations

import sys
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from bson import ObjectId
from flask import Flask, jsonify

sys.modules["sentence_transformers"] = MagicMock()

from core import _fetch_document_text, core_bp, init_core


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


def _collection(mock_clients: Dict[str, Any]) -> MagicMock:
    coll = MagicMock()
    mock_clients["mongo"].__getitem__.return_value.__getitem__.return_value = coll
    return coll


def _gap_body(**overrides: Any) -> Dict[str, Any]:
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


def _llm_response(*blocks: tuple[str, str]) -> MagicMock:
    response = MagicMock()
    content = []
    for block_type, text in blocks:
        block = MagicMock()
        block.type = block_type
        block.text = text
        content.append(block)
    response.content = content
    return response


def _prompt(mock_clients: Dict[str, Any], call_index: int = 0) -> str:
    call = mock_clients["anthropic"].messages.create.call_args_list[call_index]
    return call.kwargs["messages"][0]["content"]


class TestFetchDocumentText:
    def test_document_id_hit_strips_text_and_skips_id_lookup(self, app, mock_clients) -> None:
        coll = _collection(mock_clients)
        coll.find_one.return_value = {
            "text": "  policy body  ",
            "chunk_text": "should not win",
            "policy_chunks": [{"chunk_text": "also no"}],
        }
        with app.app_context():
            text, err = _fetch_document_text("pdb", "policies", "pol-1")
        assert err is None
        assert text == "policy body"
        coll.find_one.assert_called_once_with({"document_id": "pol-1"})

    def test_object_id_miss_does_not_query_raw_string_id(self, app, mock_clients) -> None:
        coll = _collection(mock_clients)
        oid = "656e69643030303030303031"
        coll.find_one.side_effect = [None, {"text": "from object id"}]
        with app.app_context():
            text, err = _fetch_document_text("pdb", "policies", oid)
        assert err is None
        assert text == "from object id"
        queries = [call.args[0] for call in coll.find_one.call_args_list]
        assert queries == [{"document_id": oid}, {"_id": ObjectId(oid)}]

    def test_invalid_object_id_falls_through_to_string_id(self, app, mock_clients) -> None:
        coll = _collection(mock_clients)
        coll.find_one.side_effect = [None, {"section_text": "  section body  ", "chunk_text": ""}]
        with app.app_context():
            text, err = _fetch_document_text("pdb", "policies", "not-an-object-id")
        assert err is None
        assert text == "section body"
        queries = [call.args[0] for call in coll.find_one.call_args_list]
        assert queries == [
            {"document_id": "not-an-object-id"},
            {"_id": "not-an-object-id"},
        ]

    def test_blank_text_uses_chunk_then_policy_chunks(self, app, mock_clients) -> None:
        coll = _collection(mock_clients)
        coll.find_one.return_value = {
            "text": "   ",
            "chunk_text": "  chunk wins  ",
            "section_text": "section loses",
            "policy_chunks": [{"chunk_text": "ignored"}],
        }
        with app.app_context():
            text, err = _fetch_document_text("pdb", "policies", "pol-1")
        assert err is None
        assert text == "chunk wins"

        coll.find_one.return_value = {
            "text": "",
            "chunk_text": "  ",
            "section_text": None,
            "policy_chunks": [
                "skip-me",
                {"chunk_text": "   "},
                {"chunk_header_text": " Header only "},
                {"chunk_text": " Body ", "chunk_header_text": "ignored header"},
                {"other": "no"},
            ],
        }
        with app.app_context():
            text, err = _fetch_document_text("pdb", "policies", "pol-1")
        assert err is None
        assert text == "Header only\n\nBody"

    def test_missing_or_empty_document_is_404(self, app, mock_clients) -> None:
        coll = _collection(mock_clients)
        coll.find_one.return_value = None
        with app.app_context():
            text, err = _fetch_document_text("pdb", "policies", "missing")
        assert text is None
        resp, code = err
        assert code == 404
        assert resp.get_json()["error"] == "Document not found"

        coll.find_one.return_value = {"text": "  ", "policy_chunks": "not-a-list"}
        with app.app_context():
            text, err = _fetch_document_text("pdb", "policies", "empty")
        assert text is None
        resp, code = err
        assert code == 404
        assert resp.get_json()["error"] == "Document has no extractable text"


class TestGapCheck:
    def test_default_policy_window_is_last_8000_chars(self, client, mock_clients) -> None:
        policy = "X" + ("Y" * 8000)
        with patch("core._fetch_document_text", side_effect=[("short statute", None), (policy, None)]):
            mock_clients["anthropic"].messages.create.return_value = _llm_response(
                ("text", '{"addressed": false, "policy_quote": null, "missing": true, "conflict": false, "conflict_description": null}')
            )
            response = client.post("/gap-check", json=_gap_body())
        assert response.status_code == 200
        prompt = _prompt(mock_clients)
        assert "short statute" in prompt
        assert "Y" * 8000 in prompt
        assert "XY" not in prompt
        call = mock_clients["anthropic"].messages.create.call_args
        assert call.kwargs["model"] == "claude-sonnet-4-6"
        assert call.kwargs["max_tokens"] == 500

    @pytest.mark.parametrize(
        ("max_policy_chars", "expected", "absent"),
        [
            (0, "ABCDEFGHIJ", None),
            (-4, "EFGHIJ", "ABCD"),
        ],
    )
    def test_non_positive_policy_window_uses_python_slice(
        self, client, mock_clients, max_policy_chars: int, expected: str, absent: str | None
    ) -> None:
        policy = "ABCDEFGHIJ"
        with patch("core._fetch_document_text", side_effect=[("STAT", None), (policy, None)]):
            mock_clients["anthropic"].messages.create.return_value = _llm_response(
                ("text", '{"addressed": false, "policy_quote": null, "missing": true, "conflict": false, "conflict_description": null}')
            )
            response = client.post("/gap-check", json=_gap_body(max_policy_chars=max_policy_chars))
        assert response.status_code == 200
        prompt = _prompt(mock_clients)
        assert expected in prompt
        if absent is not None:
            assert absent not in prompt

    def test_invalid_max_policy_chars_fails_before_fetch_or_llm(self, client, mock_clients) -> None:
        with patch("core._fetch_document_text") as fetch:
            response = client.post("/gap-check", json=_gap_body(max_policy_chars="nope"))
        assert response.status_code == 500
        fetch.assert_not_called()
        mock_clients["anthropic"].messages.create.assert_not_called()

    def test_statute_404_is_remapped_and_skips_policy_and_llm(self, client, mock_clients) -> None:
        seen = []

        def fetch(_db, _coll, doc_id):
            seen.append(doc_id)
            return None, (jsonify({"error": "Document not found"}), 404)

        with patch("core._fetch_document_text", side_effect=fetch):
            response = client.post("/gap-check", json=_gap_body())
        assert response.status_code == 404
        assert response.get_json()["error"] == "Statute document not found"
        assert seen == ["stat-1"]
        mock_clients["anthropic"].messages.create.assert_not_called()

    def test_policy_404_is_remapped_after_statute_fetch(self, client, mock_clients) -> None:
        def fetch(_db, _coll, doc_id):
            if doc_id == "stat-1":
                return "statute", None
            return None, (jsonify({"error": "Document has no extractable text"}), 404)

        with patch("core._fetch_document_text", side_effect=fetch):
            response = client.post("/gap-check", json=_gap_body())
        assert response.status_code == 404
        assert response.get_json()["error"] == "Policy document not found"
        mock_clients["anthropic"].messages.create.assert_not_called()

    def test_non_404_fetch_error_is_passed_through(self, client, mock_clients) -> None:
        def fetch(_db, _coll, _doc_id):
            return None, (jsonify({"error": "bad collection"}), 400)

        with patch("core._fetch_document_text", side_effect=fetch):
            response = client.post("/gap-check", json=_gap_body())
        assert response.status_code == 400
        assert response.get_json()["error"] == "bad collection"
        mock_clients["anthropic"].messages.create.assert_not_called()

    def test_string_bools_and_blank_quotes(self, client, mock_clients) -> None:
        raw = (
            'Note: {"addressed": "false", "policy_quote": "", "conflict": "false", '
            '"conflict_description": 0} thanks'
        )
        with patch("core._fetch_document_text", side_effect=[("statute", None), ("policy", None)]):
            mock_clients["anthropic"].messages.create.return_value = _llm_response(("text", raw))
            response = client.post("/gap-check", json=_gap_body())
        assert response.status_code == 200
        gap = response.get_json()["gap_check"]
        assert gap["addressed"] is True
        assert gap["missing"] is True
        assert gap["conflict"] is True
        assert gap["policy_quote"] is None
        assert gap["conflict_description"] is None

    def test_json_false_and_zero_quote_are_preserved(self, client, mock_clients) -> None:
        raw = (
            '{"addressed": false, "policy_quote": "0", "missing": null, '
            '"conflict": false, "conflict_description": "none"}'
        )
        with patch("core._fetch_document_text", side_effect=[("statute", None), ("policy", None)]):
            mock_clients["anthropic"].messages.create.return_value = _llm_response(("text", raw))
            response = client.post("/gap-check", json=_gap_body())
        assert response.status_code == 200
        gap = response.get_json()["gap_check"]
        assert gap == {
            "addressed": False,
            "policy_quote": "0",
            "missing": False,
            "conflict": False,
            "conflict_description": "none",
        }

    def test_ignores_non_text_blocks(self, client, mock_clients) -> None:
        wrong = '{"addressed": true, "policy_quote": "WRONG", "missing": false, "conflict": false, "conflict_description": null}'
        right = '{"addressed": false, "policy_quote": "RIGHT", "missing": true, "conflict": true, "conflict_description": "x"}'
        with patch("core._fetch_document_text", side_effect=[("statute", None), ("policy", None)]):
            mock_clients["anthropic"].messages.create.return_value = _llm_response(
                ("tool_use", wrong),
                ("text", right),
            )
            response = client.post("/gap-check", json=_gap_body())
        assert response.status_code == 200
        gap = response.get_json()["gap_check"]
        assert gap["policy_quote"] == "RIGHT"
        assert gap["addressed"] is False
        assert gap["conflict"] is True
        assert mock_clients["anthropic"].messages.create.call_count == 1

    def test_llm_exception_is_500(self, client, mock_clients) -> None:
        with patch("core._fetch_document_text", side_effect=[("statute", None), ("policy", None)]):
            mock_clients["anthropic"].messages.create.side_effect = RuntimeError("boom")
            response = client.post("/gap-check", json=_gap_body())
        assert response.status_code == 500
        assert response.get_json()["error"] == "LLM parsing failed: boom"


class _Page:
    def __init__(self, url: str, title: str, markdown: str, html: str = "") -> None:
        self.metadata = None
        self.url = url
        self.title = title
        self.markdown = markdown
        self.html = html
        self.raw_html = "<html>raw</html>"
        self.summary = "summary"


class TestCrawlContentSelection:
    def test_bool_and_digit_strings_pass_int_then_clamp(self, client, mock_clients) -> None:
        """JSON true and digit strings are ints, unlike non-numeric strings that 400."""
        calls = []

        def playwright(_url, depth, breadth):
            calls.append((depth, breadth))
            return [{"url": "https://example.com", "title": "T", "markdown": "Body"}]

        with patch("core.playwright_crawl", side_effect=playwright):
            as_bool = client.post(
                "/crawl",
                json={"url": "https://example.com", "depth": True, "breadth": True},
            )
            as_text = client.post(
                "/crawl",
                json={"url": "https://example.com", "depth": "4", "breadth": "101"},
            )
        assert as_bool.status_code == 200
        assert as_bool.get_json()["depth"] == 1
        assert as_bool.get_json()["breadth"] == 1
        assert as_text.status_code == 200
        assert as_text.get_json()["depth"] == 4
        assert as_text.get_json()["breadth"] == 100
        assert calls == [(1, 1), (4, 100)]
        mock_clients["firecrawl"].crawl.assert_not_called()

    def test_browser_crawl_ignores_html_and_blank_urls(self, client, mock_clients) -> None:
        pages = [
            {"url": "https://a.example/p", "title": "Privacy", "markdown": "Keep", "html": "<p>no</p>"},
            {"url": "", "title": "No URL", "markdown": "Still"},
            {"url": "https://c.example/r", "title": "Blank", "markdown": "", "html": "<p>Hi</p>"},
        ]
        with patch("core.playwright_crawl", return_value=pages):
            response = client.post("/crawl", json={"url": "https://example.com"})
        assert response.status_code == 200
        payload = response.get_json()
        assert payload["urls_crawled"] == ["https://a.example/p", "https://c.example/r"]
        assert payload["combined_text"] == (
            "Privacy | Source: https://a.example/p\nKeep\n\n"
            "No URL | Source: \nStill"
        )
        assert "<p>Hi</p>" not in payload["combined_text"]
        assert "<p>no</p>" not in payload["combined_text"]
        mock_clients["firecrawl"].crawl.assert_not_called()

    def test_firecrawl_empty_markdown_uses_html_not_raw_or_summary(self, client, mock_clients) -> None:
        page = _Page("https://html.example/policy", "HTML", "", html="<p>Hi</p>")
        mock_clients["firecrawl"].crawl.return_value = type("Job", (), {"data": [page]})()
        with patch.multiple(
            "core",
            playwright_crawl=MagicMock(return_value=[]),
            puppeteer_crawl=MagicMock(return_value=[]),
            selenium_crawl=MagicMock(return_value=[]),
        ):
            response = client.post("/crawl", json={"url": "https://example.com"})
        assert response.status_code == 200
        payload = response.get_json()
        assert payload["crawl_method"] == "firecrawl"
        assert payload["urls_crawled"] == ["https://html.example/policy"]
        assert payload["combined_text"] == "HTML | Source: https://html.example/policy\n<p>Hi</p>"
        assert "raw" not in payload["combined_text"]
        assert "summary" not in payload["combined_text"]
