"""Regression tests for PolicySegmenter edges used by statute-policy comparison.

Happy-path heading splits live in test_policy_compliance.py. These cases pin
existing-section reuse, empty input, numbered headings, heading-only sections,
wrapped-line paragraphs, and long-section splitting — all of which change which
text is sent to retrieval and the LLM.
"""
from __future__ import annotations

from segmenter import PolicySegmenter


def test_empty_or_whitespace_policy_returns_no_sections() -> None:
    segmenter = PolicySegmenter()
    assert segmenter.segment("") == []
    assert segmenter.segment("   \n\n  ") == []
    assert segmenter.segment(None) == []  # type: ignore[arg-type]


def test_existing_sections_use_section_text_or_text_and_skip_blanks() -> None:
    """Pre-parsed sections prefer section_text, then text, and drop empty rows."""
    segmenter = PolicySegmenter(max_section_chars=2000)
    sections = segmenter.segment(
        "ignored when existing_sections is non-empty",
        existing_sections=[
            {"section_id": "keep-id", "section_text": "Keep this body."},
            {"title": "Fallback Title", "text": "From text key."},
            {"section_id": "blank", "section_text": "   "},
            {"title": "No body"},
        ],
    )
    assert [s.section_id for s in sections] == ["keep-id", "fallback_title"]
    assert sections[0].section_text == "Keep this body."
    assert sections[1].section_text == "From text key."


def test_empty_existing_sections_list_falls_back_to_policy_text() -> None:
    """An empty list is falsy, so callers still get heading-based segmentation."""
    segmenter = PolicySegmenter()
    sections = segmenter.segment(
        "DATA RETENTION\nWe keep data for analytics.",
        existing_sections=[],
    )
    assert len(sections) == 1
    assert sections[0].section_id.startswith("data_retention")
    assert "We keep data" in sections[0].section_text


def test_numbered_heading_and_heading_only_section() -> None:
    """Numbered statute-style headings split sections; a heading with no body is kept."""
    segmenter = PolicySegmenter()
    text = "1. Introduction\n\nWe describe the policy.\n\n2. Definitions\n"
    sections = segmenter.segment(text)
    assert len(sections) == 2
    assert sections[0].section_id.startswith("1_introduction") or "introduction" in sections[0].section_id
    assert "We describe the policy." in sections[0].section_text
    assert "2. Definitions" in sections[1].section_text


def test_wrapped_lines_join_into_a_single_paragraph() -> None:
    segmenter = PolicySegmenter()
    text = "INTRODUCTION\nWe keep\ndata for analytics.\n"
    sections = segmenter.segment(text)
    assert len(sections) == 1
    assert "We keep data for analytics." in sections[0].section_text


def test_long_section_split_suffixes_ids_and_preserves_text() -> None:
    """Sections over max_section_chars are split into indexed chunks without dropping text."""
    segmenter = PolicySegmenter(max_section_chars=20)
    sections = segmenter.segment(
        "ignored",
        existing_sections=[{"section_id": "sec", "section_text": "abcdefghijklmnopqrstuvwxyz"}],
    )
    assert [s.section_id for s in sections] == ["sec_1", "sec_2"]
    assert "".join(s.section_text for s in sections) == "abcdefghijklmnopqrstuvwxyz"
    assert all(len(s.section_text) <= 20 for s in sections)
