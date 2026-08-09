"""Regression tests for llm_client.build_prompt retrieval prompt assembly."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

# anthropic is imported at llm_client module load.
sys.modules.setdefault("anthropic", MagicMock())

from llm_client import RETRIEVAL_PROMPT_TEMPLATE, build_prompt
from vector_retriever import StatuteCandidate


def test_build_prompt_injects_section_and_candidates() -> None:
    candidates = [
        StatuteCandidate(
            statute_id="stat-1",
            jurisdiction="CA",
            title="Right to Delete",
            section_id="1798.105",
            chunk_text="A consumer has the right to delete.",
            score=0.91,
            chunk_id="c1",
        ),
        StatuteCandidate(
            statute_id="stat-2",
            jurisdiction="VA",
            title="Access Rights",
            section_id="59.1",
            chunk_text="A consumer may access personal data.",
            score=0.42,
            chunk_id="c2",
        ),
    ]

    prompt = build_prompt("We delete data on request.", candidates)

    assert "We delete data on request." in prompt
    assert "<<<SECTION_TEXT>>>" not in prompt
    assert "1) [stat-1] CA Right to Delete" in prompt
    assert "A consumer has the right to delete." in prompt
    assert "score: 0.9100" in prompt
    assert "2) [stat-2] VA Access Rights" in prompt
    assert "score: 0.4200" in prompt
    # Placeholder candidate block from the template must be replaced.
    assert "[STATUTE_ID]" not in prompt
    assert "Output only valid JSON" in prompt
    assert prompt.count("score:") == 2


def test_build_prompt_empty_candidates_keeps_placeholder_shape() -> None:
    prompt = build_prompt("Section text only.", [])

    assert "Section text only." in prompt
    assert "<<<SECTION_TEXT>>>" not in prompt
    # With no candidates, the default placeholder example remains so the LLM
    # still sees the expected numbered-candidate shape.
    assert "[STATUTE_ID]" in prompt
    assert RETRIEVAL_PROMPT_TEMPLATE.split("Policy section:")[0] in prompt or "privacy law analyst" in prompt.lower()
