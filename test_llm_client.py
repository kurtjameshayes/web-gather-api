"""Regression tests for LLM client JSON parsing and V4 prompt wiring."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from compliance_config import load_config
from llm_client import AnthropicLLMClient


class _TextBlock:
    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


def _response(text: str) -> SimpleNamespace:
    return SimpleNamespace(content=[_TextBlock(text)])


def _client_without_init() -> AnthropicLLMClient:
    client = AnthropicLLMClient.__new__(AnthropicLLMClient)
    client._config = load_config()
    client._model = "test-model"
    client._max_tokens = 128
    client._client = MagicMock()
    return client


def test_call_json_retries_malformed_response_with_reminder_and_token_override() -> None:
    client = _client_without_init()
    client._client.messages.create.side_effect = [
        _response("not json"),
        _response('{"ok": true, "count": 2}'),
    ]

    result = asyncio.run(client._call_json("Return JSON", max_tokens=2048))

    assert result == {"ok": True, "count": 2}
    assert client._client.messages.create.call_count == 2
    first_call, second_call = client._client.messages.create.call_args_list
    assert first_call.kwargs["max_tokens"] == 2048
    assert second_call.kwargs["max_tokens"] == 2048
    assert second_call.kwargs["messages"][0]["content"].startswith("Return JSON")
    assert "Output only valid JSON" in second_call.kwargs["messages"][0]["content"]


def test_gap_check_v4_injects_adaptive_feedback_into_prompt() -> None:
    client = _client_without_init()
    captured_prompts: list[str] = []

    async def fake_call_json(prompt: str):
        captured_prompts.append(prompt)
        return {
            "status": "partial",
            "policy_quote": "We honor deletion requests.",
            "statute_quote": "A consumer may request deletion.",
            "requirement_summary": "Deletion requests must be supported.",
            "gap_description": "The response window is missing.",
            "confidence": "medium",
        }

    client._call_json = AsyncMock(side_effect=fake_call_json)

    result = asyncio.run(
        client.gap_check_v4(
            reference_context="Definitions context",
            statutory_requirement="Deletion requirement",
            policy_text="We honor deletion requests.",
            adaptive_feedback="- Require exact response-window analysis",
        )
    )

    assert result["status"] == "partial"
    assert result["confidence"] == "medium"
    assert result["conflict_description"] == "The response window is missing."
    assert len(captured_prompts) == 1
    assert "- Require exact response-window analysis" in captured_prompts[0]
    assert "Definitions context" in captured_prompts[0]
    assert "Deletion requirement" in captured_prompts[0]
