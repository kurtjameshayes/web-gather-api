"""Regression tests for _v4_statute_text statute document shaping."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("anthropic", MagicMock())

from compliance_suite_service import ComplianceSuiteService


def test_v4_statute_text_combines_header_and_subtopic() -> None:
    text = ComplianceSuiteService._v4_statute_text(
        {
            "header_text": "Cal. Civ. Code Sec. 1798.100",
            "subtopic_text": "Businesses must disclose consumer access rights.",
            "requirement_summary": "Right to know",
        }
    )
    assert text == (
        "Cal. Civ. Code Sec. 1798.100\n\nBusinesses must disclose consumer access rights."
    )


def test_v4_statute_text_falls_back_to_requirement_summary() -> None:
    text = ComplianceSuiteService._v4_statute_text(
        {
            "header_text": "Header Only Context",
            "subtopic_text": "   ",
            "requirement_summary": "Consumers may request deletion.",
        }
    )
    assert text == "Header Only Context\n\nConsumers may request deletion."


def test_v4_statute_text_header_or_empty() -> None:
    assert (
        ComplianceSuiteService._v4_statute_text({"header_text": "Standalone header"})
        == "Standalone header"
    )
    assert ComplianceSuiteService._v4_statute_text({}) == ""
    assert ComplianceSuiteService._v4_statute_text({"subtopic_text": "Body only"}) == "Body only"
