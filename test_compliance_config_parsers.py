"""Regression tests for compliance config env-var parsers."""
from __future__ import annotations

import pytest

from compliance_config import _to_bool, _to_float, _to_int


@pytest.mark.parametrize(
    ("value", "default", "expected"),
    [
        (None, True, True),
        (None, False, False),
        ("true", False, True),
        (" YES ", False, True),
        ("on", False, True),
        ("1", False, True),
        ("y", False, True),
        ("false", True, False),
        ("0", True, False),
        ("nope", True, False),
        ("", True, False),
    ],
)
def test_to_bool_truthy_and_default_paths(value: str | None, default: bool, expected: bool) -> None:
    assert _to_bool(value, default) is expected


@pytest.mark.parametrize(
    ("value", "default", "expected"),
    [
        (None, 7, 7),
        ("42", 0, 42),
        ("-3", 0, -3),
        ("not-int", 9, 9),
        ("", 5, 5),
    ],
)
def test_to_int_parses_or_falls_back(value: str | None, default: int, expected: int) -> None:
    assert _to_int(value, default) == expected


@pytest.mark.parametrize(
    ("value", "default", "expected"),
    [
        (None, 0.75, 0.75),
        ("1.25", 0.0, 1.25),
        ("0", 1.0, 0.0),
        ("bad", 0.5, 0.5),
        ("", 0.1, 0.1),
    ],
)
def test_to_float_parses_or_falls_back(value: str | None, default: float, expected: float) -> None:
    assert _to_float(value, default) == expected
