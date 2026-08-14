"""Edge-case regression tests for shared compliance_utils helpers.

Happy-path coverage lives in test_security_and_utils.py. These cases pin
LLM JSON extraction and filter/slug/truncate edges that affect retrieval
and prompt construction across the compliance suite.
"""
from __future__ import annotations

from compliance_utils import (
    extract_json_block,
    jurisdiction_filter_values,
    safe_truncate,
    slugify,
    validate_collection_name,
)


def test_extract_json_block_empty_and_missing() -> None:
    assert extract_json_block("") is None
    assert extract_json_block(None) is None  # type: ignore[arg-type]
    assert extract_json_block("no object here") is None


def test_extract_json_block_nested_object() -> None:
    raw = 'prefix {"outer": {"inner": 1}, "ok": true} suffix'
    assert extract_json_block(raw) == '{"outer": {"inner": 1}, "ok": true}'


def test_extract_json_block_greedy_span_across_multiple_objects() -> None:
    """The helper uses greedy DOTALL matching; callers must tolerate a span
    from the first '{' to the last '}' when the model emits extra objects."""
    raw = '{"a": 1} commentary {"b": 2}'
    assert extract_json_block(raw) == '{"a": 1} commentary {"b": 2}'


def test_extract_json_block_from_fenced_markdown() -> None:
    raw = "Here is the result:\n```json\n{\"compliance\": \"neither\"}\n```\n"
    assert extract_json_block(raw) == '{"compliance": "neither"}'


def test_jurisdiction_filter_values_empty_and_unknown() -> None:
    assert jurisdiction_filter_values("") == []
    values = jurisdiction_filter_values("NY")
    assert values[0] == "NY"
    assert "NY" in values


def test_slugify_falls_back_for_empty_or_punctuation() -> None:
    assert slugify("   ") == "section"
    assert slugify("***") == "section"
    assert slugify("", fallback="intro") == "intro"


def test_safe_truncate_non_positive_max_returns_original() -> None:
    assert safe_truncate("abcdef", 0) == "abcdef"
    assert safe_truncate("abcdef", -1) == "abcdef"
    assert safe_truncate("abcdef", 10) == "abcdef"


def test_validate_collection_name_rejects_empty() -> None:
    assert validate_collection_name("") is False
    assert validate_collection_name("ok_name") is True
