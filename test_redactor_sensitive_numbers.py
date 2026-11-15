"""Regression tests for SSN, card, and phone boundaries in the default redactor."""
from __future__ import annotations

from redactor import Redactor


def test_ssn_and_card_patterns_do_not_swallow_nearby_numbers() -> None:
    result = Redactor().redact(
        "ssn 123-45-6789 spaced 123 45 6789 dotted 123.45.6789 "
        "plain 123456789 card 4111111111111111 spacedcard 4111 1111 1111 1111 "
        "phone 555-123-4567 paren (555) 123-4567 "
        "twelve 123456789012 thirteen 1234567890123 "
        "tail 123-45-678901234"
    )
    assert result.redacted_text == (
        "ssn [REDACTED_SSN] spaced [REDACTED_SSN] dotted [REDACTED_SSN] "
        "plain [REDACTED_SSN] card [REDACTED_CARD] spacedcard [REDACTED_CARD] "
        "phone [REDACTED_PHONE] paren ([REDACTED_PHONE] "
        "twelve 123456789012 thirteen [REDACTED_CARD] "
        "tail 123-45-[REDACTED_SSN]"
    )
    assert result.redaction_count == 10


def test_empty_text_redacts_nothing() -> None:
    result = Redactor().redact("")
    assert result.redacted_text == ""
    assert result.redaction_count == 0
