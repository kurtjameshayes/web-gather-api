"""PII redaction helper with pluggable patterns."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple


DEFAULT_PATTERNS: List[Tuple[str, str]] = [
    (r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[REDACTED_EMAIL]"),
    (r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b", "[REDACTED_SSN]"),
    (r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[REDACTED_PHONE]"),
    (r"\b(?:\d[ -]*?){13,16}\b", "[REDACTED_CARD]"),
]


@dataclass
class RedactionResult:
    redacted_text: str
    redaction_count: int


class Redactor:
    def __init__(self, patterns: List[Tuple[str, str]] | None = None) -> None:
        # Default patterns target common PII like emails and phone numbers.
        self._patterns = patterns or DEFAULT_PATTERNS

    def redact(self, text: str) -> RedactionResult:
        if not text:
            return RedactionResult(redacted_text="", redaction_count=0)
        redacted = text
        total = 0
        for pattern, token in self._patterns:
            redacted, count = re.subn(pattern, token, redacted)
            total += count
        return RedactionResult(redacted_text=redacted, redaction_count=total)
