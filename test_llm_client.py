"""Regression tests for AnthropicLLMClient behavior."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import llm_client
from compliance_config import load_config


def _text_response(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
    )


def test_anthropic_client_sets_timeout_on_construction(monkeypatch) -> None:
    """Constructor should always apply explicit API timeout."""
    created: dict[str, object] = {}
    fake_client = MagicMock()

    def _fake_anthropic(*, api_key: str, timeout: int) -> MagicMock:
        created["api_key"] = api_key
        created["timeout"] = timeout
        return fake_client

    monkeypatch.setattr(llm_client.anthropic, "Anthropic", _fake_anthropic)

    config = load_config()
    client = llm_client.AnthropicLLMClient(api_key="test-key", config=config)

    assert client._client is fake_client
    assert created == {
        "api_key": "test-key",
        "timeout": llm_client.LLM_TIMEOUT_SECONDS,
    }


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_call_json_retries_with_reminder_after_invalid_json(
    monkeypatch,
    anyio_backend: str,
) -> None:
    """A malformed first response should trigger one reminder retry."""
    fake_messages = MagicMock()
    fake_messages.create.side_effect = [
        _text_response('prefix {"bad": } suffix'),
        _text_response('{"ok": true}'),
    ]
    fake_client = MagicMock()
    fake_client.messages = fake_messages

    client = llm_client.AnthropicLLMClient.__new__(llm_client.AnthropicLLMClient)
    client._client = fake_client
    client._model = "claude-test"
    client._max_tokens = 777

    async def _run_inline(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(llm_client, "run_in_thread", _run_inline)

    result = await client._call_json("Assess this policy")

    assert result == {"ok": True}
    assert fake_messages.create.call_count == 2
    first_call = fake_messages.create.call_args_list[0].kwargs
    second_call = fake_messages.create.call_args_list[1].kwargs
    assert first_call["max_tokens"] == 777
    assert second_call["max_tokens"] == 777
    assert first_call["messages"][0]["content"] == "Assess this policy"
    assert second_call["messages"][0]["content"].startswith("Assess this policy")
    assert "Output only valid JSON with no surrounding text." in second_call["messages"][0]["content"]


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_risk_assessment_uses_4096_token_budget(
    anyio_backend: str,
) -> None:
    """Risk assessment calls should use a larger token budget."""
    client = llm_client.AnthropicLLMClient.__new__(llm_client.AnthropicLLMClient)
    client._call_json = AsyncMock(return_value=None)

    result = await client.risk_assessment(
        policy_text="Policy body",
        statute_summary="Statute summary",
    )

    client._call_json.assert_awaited_once()
    assert client._call_json.await_args.kwargs["max_tokens"] == 4096
    assert result == {
        "processing_purposes": [],
        "data_categories": [],
        "risks": [],
        "mitigations": [],
        "gaps_from_statute": [],
    }
