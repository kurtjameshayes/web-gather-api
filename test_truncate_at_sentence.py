"""Statute-quote truncation used by gap analysis v3, v4, and the compliance suite."""
from __future__ import annotations

from compliance_utils import truncate_at_sentence


def test_short_text_is_unchanged_including_trailing_space() -> None:
    assert truncate_at_sentence("Keep me. ", 350) == "Keep me. "
    assert truncate_at_sentence("exactlyten", 10) == "exactlyten"


def test_empty_and_non_positive_limit() -> None:
    assert truncate_at_sentence("", 20) == ""
    assert truncate_at_sentence(None, 20) == ""
    # A non-positive limit returns the original text instead of slicing it.
    long_text = "Sentence one. Sentence two."
    assert truncate_at_sentence(long_text, 0) == long_text
    assert truncate_at_sentence(long_text, -5) == long_text
    assert truncate_at_sentence("", 0) == ""
    assert truncate_at_sentence(None, 0) == ""


def test_cuts_at_last_sentence_boundary_inside_the_window() -> None:
    text = "Hello world. More text follows here and should be dropped."
    assert truncate_at_sentence(text, 20) == "Hello world."

    mixed = "Wait! Really? No more after this question."
    assert truncate_at_sentence(mixed, 16) == "Wait! Really?"


def test_no_boundary_returns_the_raw_window_stripped() -> None:
    assert truncate_at_sentence("abcdefghij", 4) == "abcd"
    assert truncate_at_sentence("abcd   efg", 7) == "abcd"


def test_boundary_whitespace_is_stripped_and_period_at_start_is_kept() -> None:
    assert truncate_at_sentence("Hi.   There is more text", 6) == "Hi."
    assert truncate_at_sentence(".abcdef", 4) == "."


def test_default_limit_is_350_and_does_not_add_an_ellipsis() -> None:
    body = ("A" * 349) + ". trailing words that must not appear"
    assert truncate_at_sentence(body) == ("A" * 349) + "."
    assert truncate_at_sentence("A" * 350) == "A" * 350
    assert "..." not in truncate_at_sentence("No punctuation " + ("x" * 400), 20)


def test_newlines_are_not_sentence_boundaries() -> None:
    text = "First line\nSecond line continues past the cut"
    assert truncate_at_sentence(text, 16) == "First line\nSecon"
    # A window that ends on the newline is rstrip'd because there is no . ! or ?.
    assert truncate_at_sentence(text, 11) == "First line"
