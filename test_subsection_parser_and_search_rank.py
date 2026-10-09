"""Subsection parsing and in-process search ranking.

parse-policy-subsections delegates line slicing to the LLM parser, and
GET /search ranks chunks with an unnormalized dot product. Existing route
tests either mock the parser or stop at a two-chunk happy path.
"""
from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import requests
from flask import Flask

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("pyppeteer", MagicMock())
sys.modules.setdefault("pypdf", MagicMock())

import core
from core import DOCUMENTS_COLLECTION, PRIVACY_COMPLIANCE_DB, WEB_GATHER_DB, core_bp, init_core


def _llm(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    other = MagicMock()
    other.type = "tool_use"
    response = MagicMock()
    response.content = [other, block]
    client = MagicMock()
    client.messages.create.return_value = response
    core.anthropic_client = client
    return client


def _message(client: MagicMock) -> str:
    return client.messages.create.call_args.kwargs["messages"][0]["content"]


def test_parse_policy_section_skips_blank_text_without_calling_llm() -> None:
    core.anthropic_client = MagicMock()
    assert core._parse_policy_section_to_subsections("   ") == ([], None)
    assert core._parse_policy_section_to_subsections(None) == ([], None)
    core.anthropic_client.messages.create.assert_not_called()


def test_parse_policy_section_uses_line_range_category_and_extra_prompt() -> None:
    client = _llm(
        '{"sections": [{"identifier": "1", "start_line": 1, "end_line": 3, '
        '"text": "IGNORED", "heading": " Scope ", "category": "  privacy  "}]}'
    )
    rows, err = core._parse_policy_section_to_subsections(
        "alpha   beta\n\nGamma",
        parse_prompt="  focus on sale  ",
        doc={"category": "fallback"},
    )
    assert err is None
    assert rows == [
        {
            "subsection_identifier": "1",
            "subsection_text": "alpha beta Gamma",
            "heading": "Scope",
            "category": "privacy",
            "start_line": 1,
            "end_line": 3,
        }
    ]
    message = _message(client)
    assert "You are a privacy policy parser." in message
    assert "Additional parsing instructions:\nfocus on sale" in message
    assert "1: alpha   beta" in message
    assert "3: Gamma" in message
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-sonnet-4-6"
    assert kwargs["max_tokens"] == 8192


def test_parse_policy_section_whitespace_prompt_is_not_appended() -> None:
    client = _llm('{"sections": [{"identifier": "1", "text": "body"}]}')
    rows, err = core._parse_policy_section_to_subsections("body", parse_prompt="   ")
    assert err is None
    assert rows[0]["subsection_text"] == "body"
    assert "Additional parsing instructions" not in _message(client)


def test_parse_policy_section_invalid_or_empty_ranges_keep_model_text() -> None:
    _llm(
        '{"sections": ['
        '{"identifier": "bad", "start_line": "nope", "end_line": 1, "text": "  keep   me  "},'
        '{"identifier": "late", "start_line": 5, "end_line": 6, "text": "fallback"},'
        '{"identifier": "one-side", "start_line": 1, "text": "partial"},'
        '{"identifier": "  ", "text": ""},'
        '{"section_index": 2, "identifier": 0, "text": "indexed"},'
        '{"identifier": "trunc", "start_line": 2.9, "end_line": 2.9, "text": "nope"}'
        "]}"
    )
    rows, err = core._parse_policy_section_to_subsections("a\nb\nc")
    assert err is None
    by_id = {row["subsection_identifier"]: row["subsection_text"] for row in rows}
    assert by_id == {
        "bad": "keep me",
        "late": "fallback",
        "one-side": "partial",
        "2": "indexed",
        "trunc": "b",
    }


def test_parse_policy_section_category_falls_back_to_document() -> None:
    _llm(
        '{"sections": ['
        '{"identifier": "1", "text": "t", "category": 0},'
        '{"identifier": "2", "text": "u", "heading": " Intro "}'
        "]}"
    )
    rows, err = core._parse_policy_section_to_subsections(
        "t",
        doc={"category": 0},
    )
    assert err is None
    assert rows[0]["category"] == "0"
    assert rows[0]["heading"] is None
    assert rows[1]["subsection_identifier"] == "2"
    assert rows[1]["heading"] == "Intro"
    assert rows[1]["category"] == "0"


def test_parse_policy_section_empty_sections_falls_through_to_subsections() -> None:
    _llm(
        '{"sections": [], "subsections": [{"identifier": "s", "text": "from-sub"}]}'
    )
    rows, err = core._parse_policy_section_to_subsections("from-sub")
    assert err is None
    assert rows[0]["subsection_identifier"] == "s"
    assert rows[0]["subsection_text"] == "from-sub"


def test_parse_policy_section_non_list_sections_are_empty_success() -> None:
    _llm('{"sections": {"identifier": "s"}}')
    assert core._parse_policy_section_to_subsections("body") == ([], None)


def test_parse_policy_section_reports_llm_and_json_failures() -> None:
    client = MagicMock()
    client.messages.create.side_effect = RuntimeError("boom")
    core.anthropic_client = client
    assert core._parse_policy_section_to_subsections("body") == ([], "boom")

    _llm("no braces here")
    assert core._parse_policy_section_to_subsections("body") == (
        [],
        "No JSON block in LLM response",
    )

    tool_only = MagicMock()
    tool_only.type = "tool_use"
    response = MagicMock()
    response.content = [tool_only]
    client = MagicMock()
    client.messages.create.return_value = response
    core.anthropic_client = client
    assert core._parse_policy_section_to_subsections("body") == (
        [],
        "No JSON block in LLM response",
    )


def test_parse_policy_section_json5_fallback_and_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = 'prefix {"sections": [{"identifier": "9", "text": "ok",}]} tail'
    _llm(payload)
    fake = MagicMock()
    fake.loads.return_value = {"sections": [{"identifier": "9", "text": "ok"}]}
    monkeypatch.setitem(sys.modules, "json5", fake)
    rows, err = core._parse_policy_section_to_subsections("ok")
    assert err is None
    assert rows[0]["subsection_identifier"] == "9"
    raw = fake.loads.call_args.args[0]
    assert raw.startswith("{") and raw.endswith("}")
    assert "identifier" in raw

    _llm(payload)
    fake.loads.side_effect = ValueError("bad json5")
    rows, err = core._parse_policy_section_to_subsections("ok")
    assert rows == []
    assert err == "Invalid JSON from LLM: bad json5"


def test_parse_policy_section_truncates_to_50000_characters() -> None:
    client = _llm('{"sections": []}')
    core._parse_policy_section_to_subsections("A" * 50000 + "TAIL")
    message = _message(client)
    assert "TAIL" not in message
    assert "A" * 50000 in message


def test_parse_policy_section_clamps_end_line_and_treats_zero_as_first_line() -> None:
    _llm(
        '{"sections": [{"identifier": "z", "start_line": 0, "end_line": 99, "text": "nope"}]}'
    )
    rows, err = core._parse_policy_section_to_subsections("only")
    assert err is None
    assert rows[0]["subsection_text"] == "only"
    assert rows[0]["start_line"] == 0
    assert rows[0]["end_line"] == 99


def test_split_into_paragraph_chunks_drops_blanks_and_keeps_single_newlines() -> None:
    assert core._split_into_paragraph_chunks("") == []
    assert core._split_into_paragraph_chunks(None) == []
    assert core._split_into_paragraph_chunks("  \n\n  ") == []
    assert core._split_into_paragraph_chunks("a\nb\n\n  c  \n\n") == ["a\nb", "c"]


def test_relevance_score_edges() -> None:
    assert core.calculate_relevance_score("", "Data", "Data") == 0.0
    assert core.calculate_relevance_score(None, "Data", "Data") == 0.0
    assert core.calculate_relevance_score("!!!", "Data", "Data") == 0.0
    assert core.calculate_relevance_score("data", None, None) == 0.0
    # Phrase lives only in the description, so the boost is 0.1 rather than 0.2.
    assert core.calculate_relevance_score(
        "data retention",
        "other",
        "data retention policy",
    ) == 0.433
    # Term match is a substring, and the same substring also takes the phrase boost.
    assert core.calculate_relevance_score("data", "database", None) == 0.867


class _Meta:
    def __init__(self, url: str | None, title: str | None) -> None:
        self.url = url
        self.title = title


class _Page:
    def __init__(self, metadata: _Meta) -> None:
        self.url = "https://page.example"
        self.title = "Page title"
        self.description = "A data page"
        self.markdown = "md"
        self.metadata = metadata


def test_serialize_search_result_metadata_override_and_dict_copy() -> None:
    kept = core.serialize_search_result(_Page(_Meta(None, "")), query="data")
    assert kept["url"] == "https://page.example"
    assert kept["title"] == "Page title"
    assert kept["markdown"] == "md"
    assert kept["score"] > 0

    overridden = core.serialize_search_result(
        _Page(_Meta("https://meta.example", "Meta title")),
        query=None,
    )
    assert overridden["url"] == "https://meta.example"
    assert overridden["title"] == "Meta title"
    assert "score" not in overridden

    original = {"url": "u", "title": "t", "description": "d"}
    copied = core.serialize_search_result(original, query="t")
    copied["url"] = "changed"
    assert original["url"] == "u"

    nested = core.serialize_search_result(
        {"metadata": {"url": "https://nested", "title": "Nested"}},
        query="nested",
    )
    assert "url" not in nested
    assert nested["score"] == 0.0


def test_cosine_similarity_is_unnormalized_dot_product() -> None:
    empty = core.cosine_similarity(np.array([1.0, 0.0]), [])
    assert empty.size == 0
    scores = core.cosine_similarity(
        np.array([2.0, 0.0]),
        [np.array([2.0, 0.0]), np.array([0.0, 3.0])],
    )
    np.testing.assert_allclose(scores, [4.0, 0.0])


def test_detect_content_type_lowercases_header_and_swallows_errors() -> None:
    response = MagicMock()
    response.headers = {"Content-Type": "Application/PDF; charset=UTF-8"}
    with patch("core.requests.head", return_value=response) as head:
        assert core.detect_content_type("https://example.com/a") == (
            "application/pdf; charset=utf-8"
        )
    head.assert_called_once_with(
        "https://example.com/a",
        allow_redirects=True,
        timeout=10,
    )

    response.headers = {}
    with patch("core.requests.head", return_value=response):
        assert core.detect_content_type("https://example.com/a") == ""

    with patch("core.requests.head", side_effect=requests.Timeout("slow")):
        assert core.detect_content_type("https://example.com/a") == ""


@pytest.fixture
def client() -> Any:
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    mock_mongo = MagicMock()
    init_core(mock_mongo, MagicMock(), MagicMock())
    test_client = app.test_client()
    test_client.mongo = mock_mongo  # type: ignore[attr-defined]
    return test_client


class _DummyModel:
    def encode(self, inputs: list[str], **kwargs: object) -> np.ndarray:
        return np.array([[1.0, 0.0]])


def _wire(mongo: MagicMock, *names: str) -> dict[str, MagicMock]:
    dbs = {name: MagicMock() for name in names}

    def get_db(name: str) -> MagicMock:
        return dbs[name]

    mongo.__getitem__.side_effect = get_db
    for db in dbs.values():
        db.__getitem__.return_value = MagicMock()
    return dbs


def test_search_returns_top_five_and_clamps_percent_match(client: Any) -> None:
    dbs = _wire(client.mongo, WEB_GATHER_DB, "index_db")
    dbs[WEB_GATHER_DB].__getitem__.return_value.find_one.return_value = {
        "document_id": "doc-1",
        "database_name": "index_db",
        "collection_name": "docs",
        "chunk_collection": "chunks",
        "index_database_name": "index_db",
    }
    dbs["index_db"].__getitem__.return_value.find.return_value = [
        {"text": "a", "embedding": [1.0, 0.0]},
        {"text": "b", "embedding": [0.8, 0.0]},
        {"text": "c", "embedding": [0.5, 0.0]},
        {"text": "d", "embedding": [0.2, 0.0]},
        {"text": "e", "embedding": [0.1, 0.0]},
        {"text": "f", "embedding": [0.0, 1.0]},
        {"text": "g", "embedding": [-1.0, 0.0]},
    ]
    with patch("core.get_embedding_model_name", return_value="model"), patch(
        "core.get_application_embedding_model"
    ) as app_model, patch("core.get_model", return_value=_DummyModel()):
        response = client.get("/search?document_id=doc-1&query=test")
    app_model.assert_not_called()
    assert response.status_code == 200
    results = response.get_json()["results"]
    assert [row["text"] for row in results] == ["a", "b", "c", "d", "e"]
    assert results[0]["score"] == 1.0
    assert results[0]["percent_match"] == 100.0
    assert results[-1]["score"] == 0.1
    dbs["index_db"].__getitem__.return_value.find.assert_called_once_with(
        {"document_id": "doc-1"},
        {"_id": 0},
    )

    dbs["index_db"].__getitem__.return_value.find.return_value = [
        {"text": "neg", "embedding": [-1.0, 0.0]},
        {"text": "big", "embedding": [2.0, 0.0]},
    ]
    with patch("core.get_embedding_model_name", return_value="model"), patch(
        "core.get_model", return_value=_DummyModel()
    ):
        response = client.get("/search?document_id=doc-1&query=test")
    by_text = {row["text"]: row for row in response.get_json()["results"]}
    assert by_text["neg"]["score"] == -1.0
    assert by_text["neg"]["percent_match"] == 0.0
    assert by_text["big"]["score"] == 2.0
    assert by_text["big"]["percent_match"] == 100.0


def test_search_privacy_database_prefers_application_embedding_model(client: Any) -> None:
    dbs = _wire(client.mongo, WEB_GATHER_DB, PRIVACY_COMPLIANCE_DB)
    dbs[WEB_GATHER_DB].__getitem__.return_value.find_one.return_value = {
        "document_id": "doc-1",
        "database_name": "src",
        "collection_name": "docs",
        "index_database_name": PRIVACY_COMPLIANCE_DB,
        "chunk_collection": "policy_chunks",
    }
    dbs[PRIVACY_COMPLIANCE_DB].__getitem__.return_value.find.return_value = [
        {"text": "hit", "embedding": [1.0, 0.0]},
    ]
    with patch("core.get_application_embedding_model", return_value="app-model"), patch(
        "core.get_embedding_model_name"
    ) as db_model, patch("core.get_model", return_value=_DummyModel()) as get_model:
        response = client.get("/search?document_id=doc-1&query=test")
    assert response.status_code == 200
    db_model.assert_not_called()
    get_model.assert_called_once_with("app-model")

    for blank in (None, ""):
        with patch("core.get_application_embedding_model", return_value=blank), patch(
            "core.get_embedding_model_name", return_value="db-model"
        ) as db_model, patch("core.get_model", return_value=_DummyModel()) as get_model:
            response = client.get("/search?document_id=doc-1&query=test")
        assert response.status_code == 200
        db_model.assert_called_once_with(PRIVACY_COMPLIANCE_DB)
        get_model.assert_called_once_with("db-model")

    with patch("core.get_application_embedding_model", return_value="  mini  "), patch(
        "core.get_embedding_model_name"
    ) as db_model, patch("core.get_model", return_value=_DummyModel()) as get_model:
        response = client.get("/search?document_id=doc-1&query=test")
    assert response.status_code == 200
    db_model.assert_not_called()
    get_model.assert_called_once_with("  mini  ")


def test_search_missing_index_metadata_uses_source_location(client: Any) -> None:
    dbs = _wire(client.mongo, WEB_GATHER_DB, "srcdb")
    dbs[WEB_GATHER_DB][DOCUMENTS_COLLECTION].find_one.return_value = {
        "document_id": "doc-1",
        "database_name": "srcdb",
        "collection_name": "srccoll",
    }
    # The wire helper returns one collection mock per database via __getitem__.
    dbs["srcdb"].__getitem__.return_value.find.return_value = [
        {"text": "hit", "embedding": [1.0, 0.0]},
    ]
    with patch("core.get_embedding_model_name", return_value="model") as db_model, patch(
        "core.get_model", return_value=_DummyModel()
    ):
        response = client.get("/search?document_id=doc-1&query=test")
    assert response.status_code == 200
    assert response.get_json()["results"][0]["text"] == "hit"
    db_model.assert_called_once_with("srcdb")
    dbs["srcdb"].__getitem__.assert_called_with("srccoll")
    dbs["srcdb"].__getitem__.return_value.find.assert_called_once_with(
        {"document_id": "doc-1"},
        {"_id": 0},
    )
