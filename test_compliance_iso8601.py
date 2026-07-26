"""Regression tests for compliance route ISO8601 query parsing."""
from __future__ import annotations

from compliance_routes import _parse_iso8601


def test_parse_iso8601_accepts_valid_forms_and_rejects_invalid() -> None:
    """Runs/alerts date filters must accept ISO8601 (incl. Z) and reject junk."""
    assert _parse_iso8601("2026-03-21T12:00:00+00:00") == "2026-03-21T12:00:00+00:00"
    assert _parse_iso8601(" 2026-03-21T12:00:00Z ") == "2026-03-21T12:00:00Z"
    assert _parse_iso8601("2026-03-21") == "2026-03-21"
    assert _parse_iso8601(None) is None
    assert _parse_iso8601("") is None
    assert _parse_iso8601("   ") is None
    assert _parse_iso8601("yesterday") is None
    assert _parse_iso8601("2026-13-40T99:99:99Z") is None
