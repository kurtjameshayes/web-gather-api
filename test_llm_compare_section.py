"""Regression tests for AnthropicLLMClient.compare_section JSON retry.

Service-layer suites mock this method. Open LLM PRs cover gap_check fallbacks
and StubLLMClient, but not the compare_section parse-then-reminder path used by
statute-policy section evaluation.
"""
from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.modules["sentence_transformers"] = MagicMock()
sys.modules.setdefault("anthropic", MagicMock())

import llm_client
from compliance_config import load_config
from llm_client import AnthropicLLMClient
from vector_retriever import StatuteCandidate


async def _run_inline(func, *args, **kwargs):
    return func(*args, **kwargs)


def _text_response(text: str) -> SimpleNamespace:
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


def _build_client() -> AnthropicLLMClient:
    cfg = load_config()
    with patch.object(llm_client.anthropic, "Anthropic", return_value=MagicMock()):
        return AnthropicLLMClient(api_key="test-key", config=cfg)


def _candidate() -> StatuteCandidate:
    return StatuteCandidate(
        statute_id="ccpa-105",
        jurisdiction="CA",
        title="Deletion",
        section_id="1798.105",
        chunk_text="Consumers may request deletion of personal information.",
        score=0.91,
        chunk_id="chunk-1",
    )


def test_compare_section_returns_first_valid_json_without_retry() -> None:
    client = _build_client()
    valid = '{"compliance":"compliant","confidence":0.9}'
    client._client.messages.create.return_value = _text_response(valid)

    with patch.object(llm_client, "run_in_thread", side_effect=_run_inline):
        result = asyncio.run(
            client.compare_section("sec-1", "Users may request deletion.", [_candidate()])
        )

    assert result == valid
    assert client._client.messages.create.call_count == 1
    prompt = client._client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "Users may request deletion." in prompt
    assert "Consumers may request deletion of personal information." in prompt
    # Template already includes the instruction once; a successful parse must not retry.
    assert prompt.count("Output only valid JSON with no surrounding text.") == 1


def test_compare_section_retries_with_reminder_when_json_is_invalid() -> None:
    client = _build_client()
    client._client.messages.create.side_effect = [
        _text_response("not json at all"),
        _text_response('{"compliance":"neither","confidence":0.2}'),
    ]

    with patch.object(llm_client, "run_in_thread", side_effect=_run_inline):
        result = asyncio.run(
            client.compare_section("sec-1", "Ambiguous retention language.", [_candidate()])
        )

    assert result == '{"compliance":"neither","confidence":0.2}'
    assert client._client.messages.create.call_count == 2
    retry_prompt = client._client.messages.create.call_args_list[1].kwargs["messages"][0]["content"]
    assert retry_prompt.endswith("Output only valid JSON with no surrounding text.")
    assert retry_prompt.count("Output only valid JSON with no surrounding text.") == 2


def test_compare_section_returns_second_response_even_if_still_invalid() -> None:
    """Current contract: a failed retry still returns the raw second payload."""
    client = _build_client()
    client._client.messages.create.side_effect = [
        _text_response("first malformed"),
        _text_response("still malformed"),
    ]

    with patch.object(llm_client, "run_in_thread", side_effect=_run_inline):
        result = asyncio.run(
            client.compare_section("sec-1", "Policy text.", [_candidate()])
        )

    assert result == "still malformed"
    assert client._client.messages.create.call_count == 2
