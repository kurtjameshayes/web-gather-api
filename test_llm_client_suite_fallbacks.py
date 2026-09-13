"""Regression tests for LLM client fallbacks used by risk/suggest/router flows."""
from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock

sys.modules.setdefault("anthropic", MagicMock())

from compliance_config import load_config
from llm_client import AnthropicLLMClient


def _client() -> AnthropicLLMClient:
    return AnthropicLLMClient("test-key", load_config())


def test_risk_assessment_parse_failure_returns_empty_lists() -> None:
    client = _client()
    client._call_json = AsyncMock(return_value=None)

    out = asyncio.run(client.risk_assessment("policy", "statute"))

    assert out == {
        "processing_purposes": [],
        "data_categories": [],
        "risks": [],
        "mitigations": [],
        "gaps_from_statute": [],
    }
    assert client._call_json.await_args.kwargs.get("max_tokens") == 4096


def test_risk_assessment_null_fields_become_empty_lists() -> None:
    client = _client()
    client._call_json = AsyncMock(
        return_value={
            "processing_purposes": None,
            "data_categories": None,
            "risks": ["r1"],
            "mitigations": None,
            "gaps_from_statute": None,
        }
    )

    out = asyncio.run(client.risk_assessment("policy", "statute"))

    assert out["processing_purposes"] == []
    assert out["data_categories"] == []
    assert out["risks"] == ["r1"]
    assert out["mitigations"] == []
    assert out["gaps_from_statute"] == []


def test_suggest_policy_rejects_non_string_suggested_text() -> None:
    client = _client()
    client._call_json = AsyncMock(return_value={"suggested_policy_text": 123})

    out = asyncio.run(
        client.suggest_policy("policy", "gap text", "missing", "statute")
    )

    assert out is None


def test_suggest_policy_defaults_blank_modifications_description() -> None:
    client = _client()
    client._call_json = AsyncMock(return_value={"suggested_policy_text": "rewritten"})

    out = asyncio.run(
        client.suggest_policy("policy", "gap text", "missing", "statute")
    )

    assert out == {
        "suggested_policy_text": "rewritten",
        "modifications_description": "",
    }


def test_consumer_rights_router_returns_none_when_trees_not_dict() -> None:
    client = _client()
    client._call_json = AsyncMock(return_value={"trees": []})

    out = asyncio.run(
        client.consumer_rights_router(
            "deletion",
            "Deletion",
            "policy",
            "statute",
            "CA",
        )
    )

    assert out is None


def test_consumer_rights_router_defaults_missing_policy_gaps() -> None:
    client = _client()
    client._call_json = AsyncMock(return_value={"trees": {"ca": {"label": "start"}}})

    out = asyncio.run(
        client.consumer_rights_router(
            "deletion",
            "Deletion",
            "policy",
            "statute",
            "CA",
        )
    )

    assert out == {"trees": {"ca": {"label": "start"}}, "policy_gaps": {}}
