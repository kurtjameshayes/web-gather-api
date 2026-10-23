"""Heading classification and section identity for PolicySegmenter.

The base test covers two short uppercase headings. These cases pin the
120-character heading cutoff, numbered headings, preamble sections, and
caller-supplied section ids. Oversized text is split on a boundary with no
spaces so this does not retest whitespace trimming at the cut.
"""
from __future__ import annotations

from segmenter import PolicySegmenter


def test_uppercase_heading_cutoff_is_120_characters() -> None:
    segmenter = PolicySegmenter(max_section_chars=5000)
    short = "A" * 120
    long = "B" * 121

    headed = segmenter.segment(f"{short}\n\nWe keep records.")
    assert len(headed) == 1
    assert headed[0].section_id == "a" * 120
    assert headed[0].section_text == f"{short}\n\nWe keep records."

    body = segmenter.segment(f"{long}\n\nWe keep records.")
    assert len(body) == 1
    assert body[0].section_id == "section_1"
    assert body[0].section_text == f"{long}\n\nWe keep records."


def test_numbered_short_uppercase_and_preamble_sections() -> None:
    segmenter = PolicySegmenter()
    text = (
        "We collect data.\n\n"
        "1. Intro\n\n"
        "First body.\n\n"
        "IV. Scope\n\n"
        "Second body.\n\n"
        "OK\n\n"
        "Ok this stays in the previous section."
    )
    sections = segmenter.segment(text)

    assert [(s.section_id, s.section_text) for s in sections] == [
        ("section_1", "We collect data."),
        ("1_intro", "1. Intro\n\nFirst body."),
        ("iv_scope", "IV. Scope\n\nSecond body."),
        ("ok", "OK\n\nOk this stays in the previous section."),
    ]


def test_wrapped_lines_join_with_spaces_and_title_only_heading() -> None:
    segmenter = PolicySegmenter()

    wrapped = segmenter.segment("We keep\ndata for\nanalytics.")
    assert len(wrapped) == 1
    assert wrapped[0].section_id == "section_1"
    assert wrapped[0].section_text == "We keep data for analytics."

    title_only = segmenter.segment("SECURITY")
    assert len(title_only) == 1
    assert title_only[0].section_id == "security"
    assert title_only[0].section_text == "SECURITY"


def test_existing_sections_keep_ids_and_skip_blank_text() -> None:
    segmenter = PolicySegmenter(max_section_chars=5000)
    sections = segmenter.segment(
        "ignored when sections are supplied",
        existing_sections=[
            {"section_text": "   ", "title": "Ignored"},
            {"section_id": "keep-me", "text": "  Body  "},
            {"section_id": "  ", "section_text": "Spaced id"},
            {"title": "Data Retention", "section_text": "Keep records."},
            {"title": "!!!", "section_text": "Fallback"},
            {"section_text": "No title"},
        ],
    )

    assert [(s.section_id, s.section_text) for s in sections] == [
        ("keep-me", "Body"),
        ("  ", "Spaced id"),
        ("data_retention", "Keep records."),
        ("section_5", "Fallback"),
        ("section_6", "No title"),
    ]


def test_oversize_section_uses_numbered_suffixes() -> None:
    segmenter = PolicySegmenter(max_section_chars=5)
    sections = segmenter.segment("abcdefghij")

    assert [(s.section_id, s.section_text) for s in sections] == [
        ("section_1_1", "abcde"),
        ("section_1_2", "fghij"),
    ]
