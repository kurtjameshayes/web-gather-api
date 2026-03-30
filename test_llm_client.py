"""Targeted unit tests for AnthropicLLMClient parsing and prompt wiring."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import llm_client
from compliance_config import load_config


def _anthropic_text_response(text: str) -> MagicMock:
    """Build a minimal Anthropic-style response with one text block."""
    response = MagicMock()
    response.content = [SimpleNamespace(type="text", text=text)]
    return response


def test_anthropic_client_uses_request_timeout():
    """Regression guard: constructor must pass explicit timeout to SDK."""
    config = load_config()
    with patch("llm_client.anthropic.Anthropic") as mock_anthropic:
        llm_client.AnthropicLLMClient(api_key="test-key", config=config)

    mock_anthropic.assert_called_once_with(
        api_key="test-key",
        timeout=llm_client.LLM_TIMEOUT_SECONDS,
    )


def test_call_json_retries_once_when_first_response_not_json():
    config = load_config()
    with patch("llm_client.anthropic.Anthropic") as mock_anthropic:
        sdk_client = MagicMock()
        sdk_client.messages.create.side_effect = [
            _anthropic_text_response("Not JSON at all"),
            _anthropic_text_response('{"ok": true, "source": "retry"}'),
        ]
        mock_anthropic.return_value = sdk_client

        client = llm_client.AnthropicLLMClient(api_key="k", config=config)
        result = asyncio.run(client._call_json("prompt-body"))

    assert result == {"ok": True, "source": "retry"}
    assert sdk_client.messages.create.call_count == 2


def test_call_json_without_retry_returns_none_after_parse_failure():
    config = load_config()
    with patch("llm_client.anthropic.Anthropic") as mock_anthropic:
        sdk_client = MagicMock()
        sdk_client.messages.create.return_value = _anthropic_text_response('{"bad": }')
        mock_anthropic.return_value = sdk_client

        client = llm_client.AnthropicLLMClient(api_key="k", config=config)
        result = asyncio.run(client._call_json("prompt-body", retry_with_reminder=False))

    assert result is None
    assert sdk_client.messages.create.call_count == 1


def test_gap_check_v4_injects_adaptive_feedback_and_normalizes_output():
    config = load_config()
    with patch("llm_client.anthropic.Anthropic"):
        client = llm_client.AnthropicLLMClient(api_key="k", config=config)

    fake_out = {
        "status": "UNKNOWN_STATUS",
        "policy_quote": "",
        "statute_quote": None,
        "requirement_summary": "  Required disclosure  ",
        "gap_description": "Gap from critic lessons",
        "confidence": "definitely",
    }
    client._call_json = AsyncMock(return_value=fake_out)

    with patch("prompt_loader.load_prompt_yaml", return_value={"prompt": "TPL"}) as mock_load:
        with patch("prompt_loader.render_prompt", return_value="rendered-prompt") as mock_render:
            result = asyncio.run(
                client.gap_check_v4(
                    reference_context="ref-context",
                    statutory_requirement="statutory-requirement",
                    policy_text="policy-text",
                    adaptive_feedback="- tighten citations",
                )
            )

    mock_load.assert_called_once()
    mock_render.assert_called_once_with(
        "TPL",
        ADAPTIVE_FEEDBACK="- tighten citations",
        REFERENCE_CONTEXT="ref-context",
        STATUTORY_REQUIREMENT="statutory-requirement",
        POLICY_TEXT="policy-text",
    )
    assert result == {
        "status": "missing",
        "policy_quote": None,
        "statute_quote": "",
        "requirement_summary": "Required disclosure",
        "conflict_description": "Gap from critic lessons",
        "confidence": "low",
        "_analysis_failed": False,
    }
