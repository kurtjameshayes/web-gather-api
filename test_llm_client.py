"""Focused regression tests for AnthropicLLMClient parsing and prompt wiring."""
from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

# Keep unit tests deterministic even when anthropic SDK is absent.
sys.modules.setdefault("anthropic", MagicMock())

from compliance_config import load_config
from llm_client import AnthropicLLMClient


def _text_response(text: str) -> SimpleNamespace:
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


async def _run_sync(func, *args, **kwargs):
    """Replacement for run_in_thread that executes inline for tests."""
    return func(*args, **kwargs)


def _config():
    cfg = load_config()
    cfg.llm_model_name = "test-model"
    cfg.llm_max_tokens = 777
    return cfg


def test_call_json_retries_with_reminder_and_parses_retry_payload():
    fake_client = MagicMock()
    fake_client.messages.create = MagicMock(
        side_effect=[
            _text_response("not-json"),
            _text_response('{"ok": true, "count": 2}'),
        ]
    )
    with patch("llm_client.anthropic.Anthropic", return_value=fake_client):
        client = AnthropicLLMClient(api_key="test-key", config=_config())
    with patch("llm_client.run_in_thread", side_effect=_run_sync):
        result = asyncio.run(client._call_json("PROMPT"))

    assert result == {"ok": True, "count": 2}
    assert fake_client.messages.create.call_count == 2
    first_prompt = fake_client.messages.create.call_args_list[0].kwargs["messages"][0]["content"]
    second_prompt = fake_client.messages.create.call_args_list[1].kwargs["messages"][0]["content"]
    assert first_prompt == "PROMPT"
    assert second_prompt.endswith("\n\nOutput only valid JSON with no surrounding text.")


def test_call_json_respects_max_tokens_override():
    fake_client = MagicMock()
    fake_client.messages.create.return_value = _text_response('{"status":"ok"}')
    with patch("llm_client.anthropic.Anthropic", return_value=fake_client):
        client = AnthropicLLMClient(api_key="test-key", config=_config())
    with patch("llm_client.run_in_thread", side_effect=_run_sync):
        result = asyncio.run(client._call_json("PROMPT", max_tokens=4096))

    assert result == {"status": "ok"}
    assert fake_client.messages.create.call_args.kwargs["max_tokens"] == 4096


def test_compare_section_retries_when_first_response_is_not_json():
    fake_client = MagicMock()
    fake_client.messages.create = MagicMock(
        side_effect=[
            _text_response("analysis without JSON"),
            _text_response('{"section_id":"s1","compliance":"compliant"}'),
        ]
    )
    with patch("llm_client.anthropic.Anthropic", return_value=fake_client):
        client = AnthropicLLMClient(api_key="test-key", config=_config())
    with patch("llm_client.run_in_thread", side_effect=_run_sync):
        text = asyncio.run(client.compare_section("s1", "Policy text", []))

    assert '"section_id":"s1"' in text
    assert fake_client.messages.create.call_count == 2
    second_prompt = fake_client.messages.create.call_args_list[1].kwargs["messages"][0]["content"]
    assert second_prompt.endswith("\n\nOutput only valid JSON with no surrounding text.")


def test_gap_check_v4_includes_adaptive_feedback_in_rendered_prompt():
    fake_client = MagicMock()
    with patch("llm_client.anthropic.Anthropic", return_value=fake_client):
        client = AnthropicLLMClient(api_key="test-key", config=_config())
    client._call_json = AsyncMock(
        return_value={
            "status": "addressed",
            "policy_quote": "quote",
            "statute_quote": "statute",
            "requirement_summary": "summary",
            "confidence": "high",
        }
    )

    with patch(
        "prompt_loader.load_prompt_yaml",
        return_value={
            "prompt": (
                "Feedback: <<<ADAPTIVE_FEEDBACK>>>\n"
                "Context: <<<REFERENCE_CONTEXT>>>\n"
                "Requirement: <<<STATUTORY_REQUIREMENT>>>\n"
                "Policy: <<<POLICY_TEXT>>>"
            )
        },
    ):
        result = asyncio.run(
            client.gap_check_v4(
                reference_context="ctx",
                statutory_requirement="must provide deletion right",
                policy_text="policy body",
                adaptive_feedback="Prefer strict citation matching.",
            )
        )

    prompt_sent = client._call_json.call_args.args[0]
    assert "Prefer strict citation matching." in prompt_sent
    assert "must provide deletion right" in prompt_sent
    assert result["status"] == "addressed"
    assert result["_analysis_failed"] is False


def test_risk_assessment_calls_call_json_with_4096_tokens():
    fake_client = MagicMock()
    with patch("llm_client.anthropic.Anthropic", return_value=fake_client):
        client = AnthropicLLMClient(api_key="test-key", config=_config())
    client._call_json = AsyncMock(return_value=None)

    result = asyncio.run(client.risk_assessment("policy", "statute summary"))

    assert result == {
        "processing_purposes": [],
        "data_categories": [],
        "risks": [],
        "mitigations": [],
        "gaps_from_statute": [],
    }
    assert client._call_json.call_args.kwargs["max_tokens"] == 4096
