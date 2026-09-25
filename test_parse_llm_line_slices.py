"""Regression tests for POST /parse-llm line slicing.

Happy-path extraction and sort order live in test_core.py. These cases pin
whitespace collapsing, the omitted-start_line sort/extract mismatch, clamped
and out-of-range line numbers, duplicate boundaries, and the blank line
inserted when multiple documents are joined.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest
from flask import Flask

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("pyppeteer", MagicMock())

from core import core_bp, init_core


@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    mock_mongo = MagicMock()
    mock_anthropic = MagicMock()
    init_core(mock_mongo, MagicMock(), mock_anthropic)
    test_client = app.test_client()
    test_client.mock_mongo = mock_mongo
    test_client.mock_anthropic = mock_anthropic
    return test_client


def _section(**overrides):
    section = {
        "section": "§ 1",
        "code_name": "Civil Code",
        "jurisdiction": "California",
        "parsed_header_text": "Header",
        "start_line": 1,
    }
    section.update(overrides)
    return section


def _parse(client, documents, sections):
    collection = MagicMock()
    collection.find.return_value = documents
    client.mock_mongo.__getitem__.return_value.__getitem__.return_value = collection

    tool_use = MagicMock()
    tool_use.type = "tool_use"
    tool_use.name = "identify_sections"
    tool_use.input = {"sections": sections}
    message = MagicMock()
    message.content = [tool_use]
    stream = MagicMock()
    stream.get_final_message.return_value = message
    client.mock_anthropic.messages.stream.return_value.__enter__.return_value = stream

    response = client.post(
        "/parse-llm",
        json={
            "database": "statutes",
            "collection": "sections",
            "parse_prompt": "Extract sections",
        },
    )
    user_message = client.mock_anthropic.messages.stream.call_args.kwargs["messages"][0]["content"]
    return response, user_message


def test_parse_llm_collapses_whitespace_in_section_text_and_header(client) -> None:
    """Newlines and repeated spaces in a section become single spaces.

    A missing or blank header is stored as an empty string, not null.
    """
    response, _user = _parse(
        client,
        [{"text": "Alpha   beta\nGamma\n\n  Delta"}],
        [
            _section(
                parsed_header_text="  Right   to\nDelete  ",
                start_line=1,
            )
        ],
    )
    assert response.status_code == 200
    parsed = response.get_json()["parsed_doc"]

    assert parsed[0]["parsed_text"] == "Alpha beta Gamma Delta"
    assert parsed[0]["parsed_header_text"] == "Right to Delete"

    response_blank, _user = _parse(
        client,
        [{"text": "Alpha"}],
        [_section(parsed_header_text=None, start_line=1)],
    )
    assert response_blank.status_code == 200
    parsed_blank = response_blank.get_json()["parsed_doc"]
    assert parsed_blank[0]["parsed_header_text"] == ""
    assert parsed_blank[0]["parsed_text"] == "Alpha"


def test_parse_llm_omitted_start_line_sorts_before_line_one_and_yields_empty_text(client) -> None:
    """A missing start_line sorts as 0 but is extracted as line 1.

    The next section's start_line of 1 then makes the omitted section's end
    line 0, so its parsed_text is empty and the explicit section keeps the
    whole document.
    """
    omitted = _section(section="§ omitted")
    del omitted["start_line"]
    response, _user = _parse(
        client,
        [{"text": "Alpha\nBeta"}],
        [
            _section(section="§ explicit", start_line=1),
            omitted,
        ],
    )
    assert response.status_code == 200
    parsed = response.get_json()["parsed_doc"]

    assert [row["section"] for row in parsed] == ["§ omitted", "§ explicit"]
    assert parsed[0]["parsed_text"] == ""
    assert parsed[1]["parsed_text"] == "Alpha Beta"


def test_parse_llm_start_line_clamps_low_and_out_of_range_is_empty(client) -> None:
    """start_line <= 0 reads from line 1. A start past the last line is empty."""
    clamped_response, _user = _parse(
        client,
        [{"text": "Alpha\nBeta"}],
        [_section(start_line=0)],
    )
    assert clamped_response.status_code == 200
    assert clamped_response.get_json()["parsed_doc"][0]["parsed_text"] == "Alpha Beta"

    past_end_response, _user = _parse(
        client,
        [{"text": "Alpha\nBeta"}],
        [_section(section="§ late", start_line=9)],
    )
    assert past_end_response.status_code == 200
    past_end = past_end_response.get_json()["parsed_doc"]
    assert past_end[0]["section"] == "§ late"
    assert past_end[0]["parsed_text"] == ""


def test_parse_llm_duplicate_start_line_gives_the_earlier_section_empty_text(client) -> None:
    """Two sections with the same start_line do not share the span.

    The earlier section ends on the line before that shared start, so its
    text is empty. The later section keeps the text from that line onward.
    """
    response, _user = _parse(
        client,
        [{"text": "L1\nL2\nL3"}],
        [
            _section(section="§ first", start_line=2),
            _section(section="§ second", start_line=2),
        ],
    )
    assert response.status_code == 200
    parsed = response.get_json()["parsed_doc"]

    assert [row["section"] for row in parsed] == ["§ first", "§ second"]
    assert parsed[0]["parsed_text"] == ""
    assert parsed[1]["parsed_text"] == "L2 L3"


def test_parse_llm_joined_documents_insert_a_blank_numbered_line(client) -> None:
    """Documents are joined with a blank line, so the second doc starts at line 3."""
    response, user_message = _parse(
        client,
        [{"text": "First."}, {"text": "Second."}],
        [
            _section(section="§ 1", start_line=1),
            _section(section="§ 2", start_line=3),
        ],
    )
    assert response.status_code == 200
    parsed = response.get_json()["parsed_doc"]

    assert "1: First.\n2: \n3: Second." in user_message
    assert parsed[0]["parsed_text"] == "First."
    assert parsed[1]["parsed_text"] == "Second."
    assert "Second." not in parsed[0]["parsed_text"]


def test_parse_llm_null_start_line_fails_the_whole_parse(client) -> None:
    """JSON null start_line is present, so sort compares None with ints and 500s.

    Distinct from an omitted start_line, which sorts as 0 and still returns 200.
    """
    response, _user = _parse(
        client,
        [{"text": "Alpha\nBeta"}],
        [
            _section(section="§ null", start_line=None),
            _section(section="§ explicit", start_line=1),
        ],
    )

    assert response.status_code == 500
    assert "LLM parsing failed" in response.get_json()["error"]
