"""Regression tests for recently added llm_client paths."""
from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

# Avoid loading sentence_transformers in unit tests.
sys.modules["sentence_transformers"] = MagicMock()
sys.modules["anthropic"] = MagicMock()

from compliance_config import load_config
from llm_client import AnthropicLLMClient


def _new_client(max_tokens: int = 128) -> AnthropicLLMClient:
    """Build a lightweight client instance without constructing real Anthropic SDK objects."""
    client = object.__new__(AnthropicLLMClient)
    client._config = load_config()
    client._model = "test-model"
    client._max_tokens = max_tokens
    client._client = MagicMock()
    return client


def _response(text: str) -> SimpleNamespace:
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


def test_call_json_retries_and_honors_max_tokens_override() -> None:
    client = _new_client(max_tokens=64)
    client._client.messages.create = MagicMock(
        side_effect=[
            _response('prefix {"broken": } suffix'),
            _response('prefix {"ok": true} suffix'),
        ]
    )

    parsed = asyncio.run(client._call_json("prompt-body", max_tokens=4096))

    assert parsed == {"ok": True}
    assert client._client.messages.create.call_count == 2

    first_call = client._client.messages.create.call_args_list[0].kwargs
    second_call = client._client.messages.create.call_args_list[1].kwargs
    assert first_call["max_tokens"] == 4096
    assert second_call["max_tokens"] == 4096
    assert "Output only valid JSON with no surrounding text." in second_call["messages"][0]["content"]


def test_risk_assessment_uses_large_token_budget_and_fallback_shape() -> None:
    client = _new_client()
    client._call_json = AsyncMock(return_value=None)

    result = asyncio.run(
        client.risk_assessment(
            policy_text="policy body",
            statute_summary="statute summary",
        )
    )

    assert result == {
        "processing_purposes": [],
        "data_categories": [],
        "risks": [],
        "mitigations": [],
        "gaps_from_statute": [],
    }
    client._call_json.assert_awaited_once()
    assert client._call_json.await_args.kwargs["max_tokens"] == 4096


def test_suggest_policy_requires_suggested_text_field() -> None:
    client = _new_client()
    client._call_json = AsyncMock(return_value={"modifications_description": "details only"})

    result = asyncio.run(
        client.suggest_policy(
            policy_text="old policy",
            gap_analysis_text="missing opt-out disclosure",
            gap_analysis_match="missing",
            statute_text="authoritative requirement",
        )
    )

    assert result is None


def test_suggest_policy_coerces_modification_description_to_string() -> None:
    client = _new_client()
    client._call_json = AsyncMock(
        return_value={
            "suggested_policy_text": "revised policy text",
            "modifications_description": 123,
        }
    )

    result = asyncio.run(
        client.suggest_policy(
            policy_text="old policy",
            gap_analysis_text="missing opt-out disclosure",
            gap_analysis_match="missing",
            statute_text="authoritative requirement",
        )
    )

    assert result == {
        "suggested_policy_text": "revised policy text",
        "modifications_description": "123",
    }
    assert client._call_json.await_args.kwargs["max_tokens"] == 4096


def test_consumer_rights_router_requires_trees_dict(monkeypatch: object) -> None:
    client = _new_client()
    client._call_json = AsyncMock(return_value={"policy_gaps": {"access": {"covered": True}}})

    import prompt_loader

    monkeypatch.setattr(prompt_loader, "load_prompt_yaml", lambda _: {"prompt": "template"})
    monkeypatch.setattr(prompt_loader, "render_prompt", lambda *_args, **_kwargs: "rendered-prompt")

    result = asyncio.run(
        client.consumer_rights_router(
            request_type="access",
            request_type_label="Access request",
            policy_text="policy text",
            statute_context="statute context",
            jurisdictions="CA,VA",
            prompt_path="unused.yaml",
        )
    )

    assert result is None
    assert client._call_json.await_args.kwargs["max_tokens"] == 4096


def test_consumer_rights_router_defaults_policy_gaps_to_empty_dict(monkeypatch: object) -> None:
    client = _new_client()
    client._call_json = AsyncMock(return_value={"trees": {"access": {"question": "q"}}})

    import prompt_loader

    monkeypatch.setattr(prompt_loader, "load_prompt_yaml", lambda _: {"prompt": "template"})
    monkeypatch.setattr(prompt_loader, "render_prompt", lambda *_args, **_kwargs: "rendered-prompt")

    result = asyncio.run(
        client.consumer_rights_router(
            request_type="access",
            request_type_label="Access request",
            policy_text="policy text",
            statute_context="statute context",
            jurisdictions="CA,VA",
            prompt_path="unused.yaml",
        )
    )

    assert result == {
        "trees": {"access": {"question": "q"}},
        "policy_gaps": {},
    }
