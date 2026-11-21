"""Regression tests for gather relevance scores.

POST /gather does not sort. It attaches a term-overlap score, and a query term
matches as a substring. Title hits count double and an exact phrase adds a boost.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest
from flask import Flask
from unittest.mock import MagicMock

sys.modules.setdefault("sentence_transformers", MagicMock())

import core
from core import core_bp, init_core


@pytest.fixture
def client() -> Any:
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    init_core(MagicMock(), MagicMock(), MagicMock())
    return app.test_client()


@pytest.mark.parametrize(
    ("query", "title", "description", "expected"),
    [
        (None, "privacy", "policy", 0.0),
        ("", "privacy", "policy", 0.0),
        ("!!!", "privacy", "policy", 0.0),
        ("   ", "privacy", "policy", 0.0),
        ("privacy", None, None, 0.0),
        ("data data", "data", "", 0.667),
        ("privacy sale", "privacy", "", 0.333),
        ("privacy sale", "", "privacy", 0.167),
        ("privacy policy", "Notice", "This privacy policy applies.", 0.433),
        ("data retention", "Our data retention", "unrelated", 0.867),
        ("cat", "category", "", 0.867),
    ],
)
def test_calculate_relevance_score_weights_title_and_phrase(
    query: str | None,
    title: str | None,
    description: str | None,
    expected: float,
) -> None:
    assert core.calculate_relevance_score(query, title, description) == expected


def test_serialize_search_result_scores_a_dict_without_mutating_it() -> None:
    original = {
        "title": "privacy policy",
        "description": "unused",
        "url": "https://example.com/policy",
    }

    scored = core.serialize_search_result(original, query="privacy policy")

    assert "score" not in original
    assert scored["url"] == "https://example.com/policy"
    assert scored["score"] == 0.867
    assert scored["percent_match"] == 86.7


def test_serialize_search_result_omits_score_without_a_query() -> None:
    scored = core.serialize_search_result(
        {"title": "privacy policy", "description": ""},
    )

    assert "score" not in scored
    assert "percent_match" not in scored


def test_blank_query_still_records_a_zero_score() -> None:
    scored = core.serialize_search_result(
        {"title": "privacy policy", "description": "privacy policy"},
        query=" ",
    )

    assert scored["score"] == 0.0
    assert scored["percent_match"] == 0.0


def test_metadata_title_replaces_the_direct_title_before_scoring() -> None:
    hit = SimpleNamespace(
        url="https://direct.example/policy",
        title="privacy policy",
        description="unused",
        metadata=SimpleNamespace(url="https://meta.example/policy", title="other topic"),
    )

    scored = core.serialize_search_result(hit, query="privacy policy")

    assert scored["url"] == "https://meta.example/policy"
    assert scored["title"] == "other topic"
    assert scored["score"] == 0.0
    assert scored["percent_match"] == 0.0


def test_empty_metadata_title_keeps_the_direct_title() -> None:
    hit = SimpleNamespace(
        url="https://direct.example/policy",
        title="privacy policy",
        description="unused",
        metadata=SimpleNamespace(url=None, title=None),
    )

    scored = core.serialize_search_result(hit, query="privacy policy")

    assert scored["url"] == "https://direct.example/policy"
    assert scored["title"] == "privacy policy"
    assert scored["score"] == 0.867
    assert scored["percent_match"] == 86.7


def test_gather_attaches_title_phrase_and_substring_scores(client: Any) -> None:
    firecrawl = core.firecrawl_client
    firecrawl.search.return_value = SimpleNamespace(
        web=[
            {
                "url": "https://example.com/phrase",
                "title": "Our data retention",
                "description": "unrelated",
            },
            {
                "url": "https://example.com/miss",
                "title": "Unrelated notice",
                "description": "Nothing here.",
            },
            {
                "url": "https://example.com/substring",
                "title": "category",
                "description": "",
            },
        ]
    )

    response = client.post("/gather", json={"query": "data retention"})
    phrase_response = response.get_json()
    assert response.status_code == 200
    by_url = {row["url"]: row for row in phrase_response["results"]}
    assert by_url["https://example.com/phrase"]["score"] == 0.867
    assert by_url["https://example.com/phrase"]["percent_match"] == 86.7
    assert by_url["https://example.com/miss"]["score"] == 0.0
    assert by_url["https://example.com/substring"]["score"] == 0.0
    assert [row["url"] for row in phrase_response["results"]] == [
        "https://example.com/phrase",
        "https://example.com/miss",
        "https://example.com/substring",
    ]

    response = client.post("/gather", json={"query": "cat"})
    substring_response = response.get_json()
    assert response.status_code == 200
    by_url = {row["url"]: row for row in substring_response["results"]}
    assert by_url["https://example.com/substring"]["score"] == 0.867
    assert by_url["https://example.com/substring"]["percent_match"] == 86.7
    assert by_url["https://example.com/phrase"]["score"] == 0.0
    firecrawl.search.assert_called_with(query="cat", limit=10)
