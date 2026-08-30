"""Regression tests for policy-subsection identifier fallbacks.

When the LLM omits identifier, the helper must use section_index, then heading,
before inserting an empty identifier.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("pyppeteer", MagicMock())

from core import _parse_policy_section_to_subsections, init_core


def _text_response(text: str) -> MagicMock:
    mock_response = MagicMock()
    block = MagicMock()
    block.type = "text"
    block.text = text
    mock_response.content = [block]
    return mock_response


@pytest.fixture
def anthropic() -> MagicMock:
    mock_anthropic = MagicMock()
    init_core(MagicMock(), MagicMock(), mock_anthropic)
    return mock_anthropic


def test_parse_policy_section_uses_section_index_when_identifier_missing(
    anthropic: MagicMock,
) -> None:
    anthropic.messages.create.return_value = _text_response(
        '{"sections":['
        '{"section_index":"3.2","heading":"Retention","text":"We keep data 12 months."}'
        "]}"
    )

    subsections, err = _parse_policy_section_to_subsections(
        "Retention\nWe keep data 12 months.",
        doc={"category": "notice"},
    )

    assert err is None
    assert len(subsections) == 1
    assert subsections[0]["subsection_identifier"] == "3.2"
    assert subsections[0]["heading"] == "Retention"
    assert subsections[0]["category"] == "notice"
    assert "We keep data 12 months." in subsections[0]["subsection_text"]
