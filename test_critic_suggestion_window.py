"""Regression tests for critic suggestion sanitizing and prompt injection.

The first ten raw suggestions are the only ones considered, and only stripped
instructions are sent back into the next gap-analysis prompt.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from adaptive_feedback_service import CriticService
from compliance_config import load_config


def test_suggestion_cap_counts_invalid_rows_before_validation() -> None:
    """A valid suggestion past the first ten raw items is discarded."""
    raw = [{"category": "not-a-category", "instruction": "ignore"} for _ in range(10)]
    raw.append(
        {
            "category": "false_positive",
            "instruction": "this valid item is past the window",
            "severity": "high",
        }
    )

    assert CriticService._validate_suggestions(raw) == []


def test_suggestion_window_drops_blank_instructions_and_keeps_whitespace_reference() -> None:
    raw = [
        "not-a-dict",
        {
            "category": "false_positive",
            "instruction": "   ",
            "severity": "high",
        },
        {
            "category": " citation_quality ",
            "instruction": " Require verbatim quotes ",
            "severity": "low",
            "description": None,
            "statute_reference": "   ",
        },
    ]

    assert CriticService._validate_suggestions(raw) == [
        {
            "category": "citation_quality",
            "description": "",
            "instruction": "Require verbatim quotes",
            "statute_reference": "   ",
            "severity": "low",
        }
    ]


def test_prompt_feedback_contains_only_stripped_instructions() -> None:
    config = load_config()
    critic = CriticService(MagicMock(), config, MagicMock())
    text = critic.format_feedback_for_prompt(
        [
            {
                "suggestions": [
                    {
                        "instruction": "  Keep this lesson  ",
                        "description": "SECRET description",
                        "category": "false_positive",
                        "severity": "high",
                    },
                    {"instruction": "   "},
                    {"instruction": None},
                ]
            },
            {},
        ]
    )

    assert text == (
        "PRIOR ANALYSIS FEEDBACK (incorporate these lessons into your analysis):\n\n"
        "- Keep this lesson"
    )
    assert "SECRET" not in text
    assert "false_positive" not in text

    assert critic.format_feedback_for_prompt(
        [{"suggestions": [{"instruction": "  "}]}]
    ) == ""
