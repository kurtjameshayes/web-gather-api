"""Leftover helper contracts not covered on base or open coverage PRs.

Base tests cover clamp(1.5)==1.0 and US/CA jurisdiction aliases. PR #208
covers extract_json_block / slugify / empty collection names, not these
bounds, unknown-jurisdiction, unicode hashing, custom redactor patterns,
or gather ranking when title is missing.
"""
from __future__ import annotations

import hashlib
import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("pyppeteer", MagicMock())
sys.modules.setdefault("pypdf", MagicMock())

import core
from compliance_storage import _iso
from compliance_utils import clamp, hash_text, normalize_jurisdiction, utc_now
from redactor import Redactor


def test_clamp_below_min_and_custom_bounds() -> None:
    assert clamp(-0.2) == 0.0
    assert clamp(0.4) == 0.4
    assert clamp(5, min_value=1.0, max_value=2.0) == 2.0
    assert clamp(0.2, min_value=1.0, max_value=2.0) == 1.0


def test_normalize_jurisdiction_unknown_empty_and_none() -> None:
    """Mapped aliases strip/case-fold; unknown values are uppercased; None raises."""
    assert normalize_jurisdiction("  uk  ") == "UK"
    assert normalize_jurisdiction("  germany  ") == "GERMANY"
    assert normalize_jurisdiction("") == ""
    with pytest.raises(AttributeError):
        normalize_jurisdiction(None)  # type: ignore[arg-type]


def test_hash_text_encodes_unicode_as_utf8() -> None:
    text = "café § 1798"
    assert hash_text(text) == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert hash_text("") == hashlib.sha256(b"").hexdigest()
    assert len(hash_text(text)) == 64


def test_redactor_custom_patterns_replace_defaults() -> None:
    """Passing patterns disables DEFAULT_PATTERNS (emails would otherwise redact)."""
    redactor = Redactor(patterns=[(r"SECRET", "[X]")])
    result = redactor.redact("Email a@example.com and SECRET token")
    assert "a@example.com" in result.redacted_text
    assert "[X]" in result.redacted_text
    assert "SECRET" not in result.redacted_text
    assert result.redaction_count == 1


def test_calculate_relevance_score_none_title_and_title_weight() -> None:
    """None title is treated as empty; title matches are worth 2x description matches."""
    desc_only = core.calculate_relevance_score("privacy", None, "A privacy notice.")
    assert desc_only > 0.0
    title_only = core.calculate_relevance_score("privacy", "Privacy Act", None)
    assert title_only > desc_only

    title_match = core.calculate_relevance_score("alpha", "alpha doc", "")
    desc_match = core.calculate_relevance_score("alpha", "", "alpha doc")
    assert title_match > desc_match


def test_iso_uses_timezone_aware_isoformat() -> None:
    dt = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    assert _iso(dt) == dt.isoformat()
    assert utc_now().tzinfo is timezone.utc
