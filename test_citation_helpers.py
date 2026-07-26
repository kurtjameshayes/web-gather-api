"""Regression tests for citation-binding helpers used by gap analysis."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

# llm_client imports anthropic at module load; stub for lightweight unit tests.
sys.modules.setdefault("anthropic", MagicMock())

from compliance_suite_service import (
    _citation_binding,
    _looks_like_section_citation,
    _section_value,
)


def test_citation_binding_normalizes_whitespace_and_requires_substring() -> None:
    """Quotes must appear in policy text after whitespace normalization."""
    policy = "Users  may  request deletion\nof personal data."
    assert _citation_binding("Users may request deletion of personal data.", policy) is True
    assert _citation_binding("  Users may request deletion  ", policy) is True
    assert _citation_binding("Users may sell personal data.", policy) is False
    assert _citation_binding("", policy) is False
    assert _citation_binding("quote", "") is False
    assert _citation_binding(None, policy) is False


def test_looks_like_section_citation_accepts_common_statute_forms() -> None:
    """Section citation detection must recognize numbered statute references."""
    assert _looks_like_section_citation("1798.105") is True
    assert _looks_like_section_citation("§ 1798.105(a)") is True
    assert _looks_like_section_citation("See 1798.140(b) for definitions") is True
    assert _looks_like_section_citation("Right to delete") is False
    assert _looks_like_section_citation("   ") is False
    assert _looks_like_section_citation(None) is False


def test_section_value_prefers_header_when_it_looks_like_citation() -> None:
    """Prefer chunk headers that look like citations over opaque section ids."""
    assert _section_value("sec-1", "1798.105(a)") == "1798.105(a)"
    assert _section_value("1798.110", "Right to know") == "1798.110"
    assert _section_value("", "Introductory text") == "Introductory text"
    assert _section_value("", "   ") is None
