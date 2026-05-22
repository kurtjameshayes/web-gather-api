"""Regression tests for LLM prompt plumbing."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from llm_client import AnthropicLLMClient


def test_gap_check_v4_passes_adaptive_feedback_to_prompt() -> None:
    """Adaptive feedback must reach the rendered V4 prompt before the LLM call."""
    client = AnthropicLLMClient.__new__(AnthropicLLMClient)
    client._config = SimpleNamespace(gap_analysis_v4_prompt_path="prompts/gap_analysis_v4.yaml")
    client._call_json = AsyncMock(
        return_value={
            "status": "addressed",
            "policy_quote": "We disclose categories.",
            "statute_quote": "Disclose categories.",
            "requirement_summary": "Disclosure requirement",
            "confidence": "high",
        }
    )

    with patch("prompt_loader.load_prompt_yaml", return_value={"prompt": "template"}), patch(
        "prompt_loader.render_prompt", return_value="rendered prompt"
    ) as render_prompt:
        result = asyncio.run(
            client.gap_check_v4(
                reference_context="definitions",
                statutory_requirement="Disclose categories.",
                policy_text="We disclose categories.",
                adaptive_feedback="Require exact citations.",
            )
        )

    assert result["status"] == "addressed"
    render_prompt.assert_called_once_with(
        "template",
        ADAPTIVE_FEEDBACK="Require exact citations.",
        REFERENCE_CONTEXT="definitions",
        STATUTORY_REQUIREMENT="Disclose categories.",
        POLICY_TEXT="We disclose categories.",
    )
    client._call_json.assert_awaited_once_with("rendered prompt")
