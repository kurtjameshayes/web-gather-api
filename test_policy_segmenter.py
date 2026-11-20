"""Tests for policy segmentation edges that change section boundaries."""
from __future__ import annotations

from segmenter import PolicySegmenter


def test_existing_sections_skip_blank_text_and_keep_split_content() -> None:
    segmenter = PolicySegmenter(max_section_chars=10)
    sections = segmenter.segment(
        "UNUSED POLICY TEXT",
        existing_sections=[
            {"section_id": "blank", "section_text": "   "},
            {"text": ""},
            {"title": "Ignored Title", "text": "ABCDEFGHIJKLMNO"},
        ],
    )
    assert [section.section_id for section in sections] == ["ignored_title_1", "ignored_title_2"]
    assert [section.section_text for section in sections] == ["ABCDEFGHIJ", "KLMNO"]
    assert "".join(section.section_text for section in sections) == "ABCDEFGHIJKLMNO"


def test_heading_heuristic_boundaries() -> None:
    segmenter = PolicySegmenter()
    assert segmenter._is_heading("OK") is True
    assert segmenter._is_heading("Ok") is False
    assert segmenter._is_heading("ABC is collected.") is True
    assert segmenter._is_heading("AB is collected.") is False
    assert segmenter._is_heading("DATA is collected for ads.") is True
    assert segmenter._is_heading("The company collects data.") is False
    assert segmenter._is_heading("IV. Security") is True
    assert segmenter._is_heading("W. Security") is False
    assert segmenter._is_heading("A" * 120) is True
    assert segmenter._is_heading("A" * 121) is False
    assert segmenter._is_heading("1. " + ("x" * 120)) is False


def test_blank_line_separates_heading_from_body_and_long_text_stays_attached() -> None:
    segmenter = PolicySegmenter()
    assert segmenter.segment("   \n\n  ") == []

    separated = segmenter.segment("DATA USE\n\nWe collect data.\n\nIV. Security\n\nEncrypt data.")
    assert [section.section_id for section in separated] == ["data_use", "iv_security"]
    assert separated[0].section_text == "DATA USE\n\nWe collect data."
    assert separated[1].section_text == "IV. Security\n\nEncrypt data."

    joined = segmenter.segment("DATA USE\nWe collect data.")
    assert len(joined) == 1
    assert joined[0].section_id == "data_use_we_collect_data"
    assert joined[0].section_text == "DATA USE We collect data."

    long_body = "A" * 121
    attached = segmenter.segment(f"DATA USE\n\n{long_body}\nMore detail.")
    assert len(attached) == 1
    assert attached[0].section_id == "data_use"
    assert long_body in attached[0].section_text
    assert "More detail." in attached[0].section_text

    title_only = segmenter.segment("SECURITY")
    assert len(title_only) == 1
    assert title_only[0].section_id == "security"
    assert title_only[0].section_text == "SECURITY"
