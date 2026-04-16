"""Targeted tests for high-risk AnthropicLLMClient logic paths."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from llm_client import AnthropicLLMClient


def _build_client() -> AnthropicLLMClient:
    client = AnthropicLLMClient.__new__(AnthropicLLMClient)
    client._config = SimpleNamespace(gap_analysis_v4_prompt_path="prompts/custom_gap_v4.yaml")
    client._call_json = AsyncMock()
    return client


def test_gap_check_v4_injects_adaptive_feedback_and_normalizes_output():
    client = _build_client()
    client._call_json.return_value = {
        "status": "partial",
        "policy_quote": "Policy quote",
        "statute_quote": "Statute quote",
        "requirement_summary": " Requirement summary ",
        "gap_description": "Missing retention details",
        "confidence": "medium",
    }

    with patch("prompt_loader.load_prompt_yaml", return_value={"prompt": "prompt-template"}) as mock_load:
        with patch("prompt_loader.render_prompt", return_value="rendered-prompt") as mock_render:
            result = asyncio.run(
                client.gap_check_v4(
                    reference_context="reference context",
                    statutory_requirement="statutory requirement",
                    policy_text="policy text",
                    adaptive_feedback="prior feedback instructions",
                )
            )

    mock_load.assert_called_once_with("prompts/custom_gap_v4.yaml")
    mock_render.assert_called_once_with(
        "prompt-template",
        ADAPTIVE_FEEDBACK="prior feedback instructions",
        REFERENCE_CONTEXT="reference context",
        STATUTORY_REQUIREMENT="statutory requirement",
        POLICY_TEXT="policy text",
    )
    client._call_json.assert_awaited_once_with("rendered-prompt")
    assert result == {
        "status": "partial",
        "policy_quote": "Policy quote",
        "statute_quote": "Statute quote",
        "requirement_summary": "Requirement summary",
        "conflict_description": "Missing retention details",
        "confidence": "medium",
        "_analysis_failed": False,
    }


def test_gap_check_v4_returns_safe_missing_result_when_llm_parse_fails():
    client = _build_client()
    client._call_json.return_value = None

    with patch("prompt_loader.load_prompt_yaml", return_value={"prompt": "prompt-template"}):
        with patch("prompt_loader.render_prompt", return_value="rendered-prompt"):
            result = asyncio.run(
                client.gap_check_v4(
                    reference_context="reference context",
                    statutory_requirement="statutory requirement",
                    policy_text="policy text",
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


def test_risk_assessment_uses_extended_max_tokens_budget():
    client = _build_client()
    client._call_json.return_value = {
        "processing_purposes": ["analytics"],
        "data_categories": ["email"],
        "risks": [{"description": "re-identification", "severity": "medium", "mitigation": "masking"}],
        "mitigations": ["masking"],
        "gaps_from_statute": [{"requirement": "notice", "jurisdiction": "CA", "status": "missing"}],
    }

    result = asyncio.run(
        client.risk_assessment(
            policy_text="policy body",
            statute_summary="statute summary",
        )
    )

    await_call = client._call_json.await_args
    assert await_call.kwargs["max_tokens"] == 4096
    assert "policy body" in await_call.args[0]
    assert "statute summary" in await_call.args[0]
    assert result["processing_purposes"] == ["analytics"]
    assert result["data_categories"] == ["email"]
