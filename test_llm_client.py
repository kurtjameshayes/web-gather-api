"""Focused tests for LLM prompt plumbing."""
from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

sys.modules.setdefault("anthropic", MagicMock())

from compliance_config import load_config
from llm_client import AnthropicLLMClient


def test_gap_check_v4_renders_adaptive_feedback_into_prompt():
    config = load_config()
    config.gap_analysis_v4_prompt_path = "prompts/gap_analysis_v4.yaml"
    client = AnthropicLLMClient.__new__(AnthropicLLMClient)
    client._config = config
    client._call_json = AsyncMock(
        return_value={
            "status": "addressed",
            "policy_quote": "We honor deletion requests.",
            "statute_quote": "Consumers may request deletion.",
            "requirement_summary": "Right to delete",
            "conflict_description": None,
            "confidence": "high",
        }
    )

    with (
        patch(
            "prompt_loader.load_prompt_yaml",
            return_value={"prompt": "Feedback: <<<ADAPTIVE_FEEDBACK>>>"},
        ) as load_prompt_yaml,
        patch("prompt_loader.render_prompt", return_value="rendered prompt") as render_prompt,
    ):
        result = asyncio.run(
            client.gap_check_v4(
                reference_context="Definitions context",
                statutory_requirement="Deletion requirement",
                policy_text="We honor deletion requests.",
                adaptive_feedback="- Require verbatim deletion citations",
            )
        )

    load_prompt_yaml.assert_called_once_with("prompts/gap_analysis_v4.yaml")
    render_prompt.assert_called_once_with(
        "Feedback: <<<ADAPTIVE_FEEDBACK>>>",
        ADAPTIVE_FEEDBACK="- Require verbatim deletion citations",
        REFERENCE_CONTEXT="Definitions context",
        STATUTORY_REQUIREMENT="Deletion requirement",
        POLICY_TEXT="We honor deletion requests.",
    )
    client._call_json.assert_awaited_once_with("rendered prompt")
    assert result == {
        "status": "addressed",
        "policy_quote": "We honor deletion requests.",
        "statute_quote": "Consumers may request deletion.",
        "requirement_summary": "Right to delete",
        "conflict_description": None,
        "confidence": "high",
        "_analysis_failed": False,
    }
