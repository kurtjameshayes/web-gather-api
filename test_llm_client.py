"""Regression tests for llm_client JSON handling and token budgets."""
from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

# Avoid heavy optional imports in unit tests.
sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("anthropic", MagicMock())

import llm_client
from compliance_config import load_config


def _text_response(text: str) -> SimpleNamespace:
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


def _build_client(monkeypatch, responses):
    messages = MagicMock()
    messages.create.side_effect = responses
    transport = MagicMock()
    transport.messages = messages
    anthropic_factory = MagicMock(return_value=transport)
    monkeypatch.setattr(llm_client.anthropic, "Anthropic", anthropic_factory)

    config = load_config()
    config.llm_max_tokens = 321
    client = llm_client.AnthropicLLMClient(api_key="test-key", config=config)
    return client, messages


def test_call_json_retries_with_json_reminder(monkeypatch):
    client, messages = _build_client(
        monkeypatch,
        responses=[
            _text_response("not valid json"),
            _text_response('{"ok": true}'),
        ],
    )

    result = asyncio.run(client._call_json("base prompt"))

    assert result == {"ok": True}
    assert messages.create.call_count == 2
    assert messages.create.call_args_list[0].kwargs["max_tokens"] == 321
    first_prompt = messages.create.call_args_list[0].kwargs["messages"][0]["content"]
    second_prompt = messages.create.call_args_list[1].kwargs["messages"][0]["content"]
    assert first_prompt == "base prompt"
    assert second_prompt.startswith("base prompt")
    assert "Output only valid JSON with no surrounding text." in second_prompt


def test_call_json_can_skip_retry(monkeypatch):
    client, messages = _build_client(
        monkeypatch,
        responses=[_text_response("still not json")],
    )

    result = asyncio.run(client._call_json("base prompt", retry_with_reminder=False))

    assert result is None
    assert messages.create.call_count == 1


def test_risk_assessment_uses_large_token_budget(monkeypatch):
    client, _messages = _build_client(
        monkeypatch,
        responses=[_text_response('{"unused": true}')],
    )
    client._call_json = AsyncMock(
        return_value={
            "processing_purposes": ["analytics"],
            "data_categories": [],
            "risks": [],
            "mitigations": [],
            "gaps_from_statute": [],
        }
    )

    result = asyncio.run(client.risk_assessment("policy text", "statute summary"))

    assert result["processing_purposes"] == ["analytics"]
    client._call_json.assert_awaited_once()
    await_args = client._call_json.await_args
    assert await_args.kwargs["max_tokens"] == 4096
    assert "policy text" in await_args.args[0]
    assert "statute summary" in await_args.args[0]
