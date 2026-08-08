"""Regression tests for policy subsection LLM parsing helpers.

Covers `_parse_policy_section_to_subsections` directly. Existing endpoint tests
mock this helper, so line-range extraction, JSON fallbacks, and category
inheritance previously had no coverage.
"""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# Avoid loading sentence_transformers / optional crawl deps in unit tests.
sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("pyppeteer", MagicMock())

import core


def _text_response(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.content = [block]
    return response


@pytest.fixture
def anthropic() -> MagicMock:
    client = MagicMock()
    core.init_core(MagicMock(), MagicMock(), client)
    return client


def test_parse_policy_section_empty_text_returns_no_subsections(anthropic: MagicMock) -> None:
    subsections, err = core._parse_policy_section_to_subsections("   ")
    assert subsections == []
    assert err is None
    anthropic.messages.create.assert_not_called()


def test_parse_policy_section_extracts_line_ranges_and_inherits_category(
    anthropic: MagicMock,
) -> None:
    text = "Header line\nYou may request deletion of personal data.\nContact us anytime."
    anthropic.messages.create.return_value = _text_response(
        json.dumps(
            {
                "sections": [
                    {
                        "identifier": "1.1",
                        "heading": "Deletion",
                        "start_line": 2,
                        "end_line": 2,
                    },
                    {
                        "identifier": "",
                        "text": "",
                        "start_line": 9,
                        "end_line": 9,
                    },
                ]
            }
        )
    )

    subsections, err = core._parse_policy_section_to_subsections(
        text,
        parse_prompt="Prefer compliance-relevant chunks",
        doc={"category": "consumer_rights"},
    )

    assert err is None
    assert len(subsections) == 1
    assert subsections[0]["subsection_identifier"] == "1.1"
    assert subsections[0]["heading"] == "Deletion"
    assert subsections[0]["category"] == "consumer_rights"
    assert subsections[0]["subsection_text"] == "You may request deletion of personal data."
    assert subsections[0]["start_line"] == 2
    assert subsections[0]["end_line"] == 2

    user_message = anthropic.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "Prefer compliance-relevant chunks" in user_message
    assert "1: Header line" in user_message
    assert "2: You may request deletion of personal data." in user_message


def test_parse_policy_section_falls_back_to_json5_and_uses_explicit_category(
    anthropic: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Force json.loads to fail so the json5 path is exercised.
    monkeypatch.setattr(core.json, "loads", MagicMock(side_effect=json.JSONDecodeError("bad", "", 0)))
    fake_json5 = SimpleNamespace(
        loads=MagicMock(
            return_value={
                "subsections": [
                    {
                        "heading": "Notice",
                        "text": "  We   collect   email.  ",
                        "category": "notice",
                    }
                ]
            }
        )
    )
    monkeypatch.setitem(sys.modules, "json5", fake_json5)

    anthropic.messages.create.return_value = _text_response(
        'Here is the result:\n{"subsections":[{"heading":"Notice","text":"We collect email.","category":"notice"}]}'
    )

    subsections, err = core._parse_policy_section_to_subsections(
        "We collect email.",
        doc={"category": "ignored_when_explicit"},
    )

    assert err is None
    assert len(subsections) == 1
    assert subsections[0]["subsection_identifier"] == "Notice"
    assert subsections[0]["subsection_text"] == "We collect email."
    assert subsections[0]["category"] == "notice"
    fake_json5.loads.assert_called_once()


def test_parse_policy_section_returns_llm_error_and_missing_json(
    anthropic: MagicMock,
) -> None:
    anthropic.messages.create.side_effect = RuntimeError("anthropic down")
    subsections, err = core._parse_policy_section_to_subsections("Policy body")
    assert subsections == []
    assert err == "anthropic down"

    anthropic.messages.create.side_effect = None
    anthropic.messages.create.return_value = _text_response("no json here")
    subsections, err = core._parse_policy_section_to_subsections("Policy body")
    assert subsections == []
    assert err == "No JSON block in LLM response"
