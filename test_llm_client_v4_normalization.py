"""Regression tests for V4 LLM output normalization and client fallbacks.

Open coverage PRs cover V4 prompt injection (adaptive feedback) and V1/V3
parse-failure defaults, but not V4 status whitelisting for partial/ambiguous,
gap_description aliasing, or suggest_policy / consumer_rights_router None
fallbacks. Those paths silently change gap scores and generated policy text.
"""
from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

sys.modules.setdefault("anthropic", MagicMock())
sys.modules.setdefault("sentence_transformers", MagicMock())

from llm_client import AnthropicLLMClient


def _client_with_json(payload) -> AnthropicLLMClient:
    client = AnthropicLLMClient.__new__(AnthropicLLMClient)
    client._config = MagicMock()
    client._config.gap_analysis_v4_prompt_path = "prompts/gap_analysis_v4.yaml"
    client._config.consumer_rights_router_prompt_path = "prompts/consumer_rights_router.yaml"
    client._call_json = AsyncMock(return_value=payload)
    return client


def _run_gap_check_v4(client: AnthropicLLMClient):
    with (
        patch("prompt_loader.load_prompt_yaml", return_value={"prompt": "tmpl"}),
        patch("prompt_loader.render_prompt", return_value="rendered"),
    ):
        return asyncio.run(
            client.gap_check_v4(
                reference_context="defs",
                statutory_requirement="req",
                policy_text="policy",
            )
        )


def test_gap_check_v4_empty_llm_response_is_failed_missing() -> None:
    """A None LLM payload must not look like a successful 'missing' finding."""
    client = _client_with_json(None)
    result = _run_gap_check_v4(client)
    assert result["status"] == "missing"
    assert result["_analysis_failed"] is True
    assert result["policy_quote"] is None
    assert result["confidence"] == "low"
    assert result["requirement_summary"] == "Requirement"


def test_gap_check_v4_keeps_partial_and_ambiguous_statuses() -> None:
    """partial/ambiguous are first-class V4 statuses; mapping them to missing hides gaps."""
    for status in ("partial", "ambiguous", "addressed", "conflict", "missing"):
        client = _client_with_json(
            {
                "status": status.upper(),
                "policy_quote": "quote",
                "statute_quote": "stat",
                "requirement_summary": "summary",
                "conflict_description": "note",
                "confidence": "high",
            }
        )
        result = _run_gap_check_v4(client)
        assert result["status"] == status
        assert result["_analysis_failed"] is False


def test_gap_check_v4_unknown_status_falls_back_to_missing() -> None:
    client = _client_with_json({"status": "not-a-real-status", "confidence": "medium"})
    result = _run_gap_check_v4(client)
    assert result["status"] == "missing"
    assert result["_analysis_failed"] is False
    assert result["confidence"] == "medium"


def test_gap_check_v4_aliases_gap_description_and_empty_quote() -> None:
    """gap_description is accepted as conflict_description; blank quotes become None."""
    client = _client_with_json(
        {
            "status": "conflict",
            "policy_quote": "",
            "statute_quote": None,
            "requirement_summary": "  Keep me  ",
            "gap_description": "Policy contradicts the statute.",
            "confidence": "not-a-level",
        }
    )
    result = _run_gap_check_v4(client)
    assert result["status"] == "conflict"
    assert result["policy_quote"] is None
    assert result["statute_quote"] == ""
    assert result["requirement_summary"] == "Keep me"
    assert result["conflict_description"] == "Policy contradicts the statute."
    assert result["confidence"] == "low"


def test_suggest_policy_returns_none_without_suggested_text() -> None:
    """Missing suggested_policy_text must not become an empty rewrite payload."""
    client = _client_with_json({"modifications_description": "tweaks only"})
    result = asyncio.run(
        client.suggest_policy(
            policy_text="old",
            gap_analysis_text="gap",
            gap_analysis_match="missing",
            statute_text="statute",
        )
    )
    assert result is None


def test_suggest_policy_returns_normalized_payload() -> None:
    client = _client_with_json(
        {
            "suggested_policy_text": "Revised policy.",
            "modifications_description": None,
        }
    )
    result = asyncio.run(
        client.suggest_policy(
            policy_text="old",
            gap_analysis_text="gap",
            gap_analysis_match="missing",
            statute_text="statute",
        )
    )
    assert result == {
        "suggested_policy_text": "Revised policy.",
        "modifications_description": "",
    }


def test_consumer_rights_router_returns_none_without_trees() -> None:
    """A malformed router payload must fail closed rather than persist empty trees."""
    client = _client_with_json({"policy_gaps": {"ca": {"covered": False}}})
    with (
        patch("prompt_loader.load_prompt_yaml", return_value={"prompt": "tmpl"}),
        patch("prompt_loader.render_prompt", return_value="rendered"),
    ):
        result = asyncio.run(
            client.consumer_rights_router(
                request_type="deletion",
                request_type_label="Delete",
                policy_text="policy",
                statute_context="statute",
                jurisdictions="CA",
            )
        )
    assert result is None
    client._call_json.assert_awaited_once()
    assert client._call_json.await_args.kwargs.get("max_tokens") == 4096
