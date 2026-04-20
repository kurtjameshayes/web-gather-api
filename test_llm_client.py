"""Focused regression tests for AnthropicLLMClient parsing and normalization."""
from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

# Keep imports deterministic even when anthropic SDK is unavailable.
sys.modules.setdefault("anthropic", MagicMock())

import llm_client
from compliance_config import load_config
from llm_client import AnthropicLLMClient


def _text_response(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
    )


async def _sync_run_in_thread(func, *args, **kwargs):
    return func(*args, **kwargs)


def _build_client() -> AnthropicLLMClient:
    cfg = load_config()
    with patch.object(llm_client.anthropic, "Anthropic", return_value=MagicMock()):
        return AnthropicLLMClient(api_key="test-key", config=cfg)


def test_call_json_retries_with_reminder_and_parses_second_response() -> None:
    cfg = load_config()
    fake_transport = MagicMock()
    fake_transport.messages.create.side_effect = [
        _text_response("not json"),
        _text_response('prefix {"decision":"ok","score":1} suffix'),
    ]

    with patch.object(llm_client.anthropic, "Anthropic", return_value=fake_transport):
        client = AnthropicLLMClient(api_key="test-key", config=cfg)

    with patch.object(llm_client, "run_in_thread", side_effect=_sync_run_in_thread):
        result = asyncio.run(client._call_json("Evaluate this", max_tokens=321))

    assert result == {"decision": "ok", "score": 1}
    assert fake_transport.messages.create.call_count == 2

    first_call = fake_transport.messages.create.call_args_list[0].kwargs
    second_call = fake_transport.messages.create.call_args_list[1].kwargs
    assert first_call["max_tokens"] == 321
    assert second_call["max_tokens"] == 321
    assert first_call["messages"][0]["content"] == "Evaluate this"
    assert second_call["messages"][0]["content"].endswith(
        "\n\nOutput only valid JSON with no surrounding text."
    )


def test_call_json_returns_none_when_both_attempts_fail() -> None:
    cfg = load_config()
    fake_transport = MagicMock()
    fake_transport.messages.create.side_effect = [
        _text_response("{ malformed"),
        _text_response("still not valid"),
    ]

    with patch.object(llm_client.anthropic, "Anthropic", return_value=fake_transport):
        client = AnthropicLLMClient(api_key="test-key", config=cfg)

    with patch.object(llm_client, "run_in_thread", side_effect=_sync_run_in_thread):
        result = asyncio.run(client._call_json("Evaluate this"))

    assert result is None
    assert fake_transport.messages.create.call_count == 2


def test_gap_check_v4_normalizes_unexpected_status_and_confidence() -> None:
    client = _build_client()
    client._call_json = AsyncMock(
        return_value={
            "status": "INVALID_STATUS",
            "policy_quote": "",
            "statute_quote": None,
            "requirement_summary": "  Keep retention limits  ",
            "gap_description": "Policy omits deletion timing.",
            "confidence": "very_high",
        }
    )

    captured = {}

    def _render(_template: str, **kwargs: str) -> str:
        captured.update(kwargs)
        return "rendered-v4-prompt"

    with patch("prompt_loader.load_prompt_yaml", return_value={"prompt": "template"}), patch(
        "prompt_loader.render_prompt", side_effect=_render
    ):
        result = asyncio.run(
            client.gap_check_v4(
                reference_context="context",
                statutory_requirement="requirement",
                policy_text="policy",
                adaptive_feedback="prior feedback",
            )
        )

    assert captured["ADAPTIVE_FEEDBACK"] == "prior feedback"
    assert captured["REFERENCE_CONTEXT"] == "context"
    assert captured["STATUTORY_REQUIREMENT"] == "requirement"
    assert captured["POLICY_TEXT"] == "policy"
    client._call_json.assert_awaited_once_with("rendered-v4-prompt")

    assert result["status"] == "missing"
    assert result["confidence"] == "low"
    assert result["policy_quote"] is None
    assert result["statute_quote"] == ""
    assert result["requirement_summary"] == "Keep retention limits"
    assert result["conflict_description"] == "Policy omits deletion timing."
    assert result["_analysis_failed"] is False


def test_gap_check_v4_returns_fallback_when_llm_parse_fails() -> None:
    client = _build_client()
    client._call_json = AsyncMock(return_value=None)

    with patch("prompt_loader.load_prompt_yaml", return_value={"prompt": "template"}), patch(
        "prompt_loader.render_prompt", return_value="rendered-v4-prompt"
    ):
        result = asyncio.run(
            client.gap_check_v4(
                reference_context="ctx",
                statutory_requirement="req",
                policy_text="policy",
            )
        )

    assert result == {
        "status": "missing",
        "policy_quote": None,
        "statute_quote": None,
        "requirement_summary": "Requirement",
        "conflict_description": None,
        "confidence": "low",
        "_analysis_failed": True,
    }


def test_risk_assessment_uses_4096_tokens_and_falls_back_on_failure() -> None:
    client = _build_client()
    client._call_json = AsyncMock(return_value=None)

    result = asyncio.run(client.risk_assessment("policy text", "statute summary"))

    assert result == {
        "processing_purposes": [],
        "data_categories": [],
        "risks": [],
        "mitigations": [],
        "gaps_from_statute": [],
    }
    assert client._call_json.await_count == 1
    _, kwargs = client._call_json.call_args
    assert kwargs["max_tokens"] == 4096
