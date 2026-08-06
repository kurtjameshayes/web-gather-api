"""Regression tests for AnthropicLLMClient fallback and normalization paths.

These methods are widely mocked at the service layer in open coverage PRs, but the
client-side parse-failure defaults and field coercion were previously untested.
"""
from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

# Keep imports deterministic even when anthropic SDK is unavailable.
sys.modules.setdefault("anthropic", MagicMock())

import llm_client
from compliance_config import load_config
from llm_client import AnthropicLLMClient, StubLLMClient


def _build_client() -> AnthropicLLMClient:
    cfg = load_config()
    with patch.object(llm_client.anthropic, "Anthropic", return_value=MagicMock()):
        return AnthropicLLMClient(api_key="test-key", config=cfg)


def test_gap_check_returns_safe_fallback_when_parse_fails() -> None:
    client = _build_client()
    client._call_json = AsyncMock(return_value=None)

    result = asyncio.run(
        client.gap_check(
            statute_chunk_text="Consumers may request deletion.",
            statute_reference="§ 1798.105",
            policy_text="We retain data indefinitely.",
        )
    )

    assert result == {
        "addressed": False,
        "policy_quote": None,
        "missing": True,
        "conflict": False,
        "conflict_description": None,
    }
    prompt = client._call_json.await_args.args[0]
    assert "Consumers may request deletion." in prompt
    assert "We retain data indefinitely." in prompt


def test_gap_check_normalizes_empty_quotes_and_bools() -> None:
    client = _build_client()
    client._call_json = AsyncMock(
        return_value={
            "addressed": 1,
            "policy_quote": "",
            "missing": 0,
            "conflict": "yes",
            "conflict_description": "",
        }
    )

    result = asyncio.run(
        client.gap_check("statute", "ref", "policy")
    )

    assert result["addressed"] is True
    assert result["missing"] is False
    assert result["conflict"] is True
    assert result["policy_quote"] is None
    assert result["conflict_description"] is None


def test_gap_check_v3_normalizes_invalid_status_and_confidence() -> None:
    client = _build_client()
    client._call_json = AsyncMock(
        return_value={
            "status": "UNKNOWN",
            "policy_quote": "",
            "statute_quote": None,
            "requirement_summary": "  Disclose sale opt-out  ",
            "gap_description": "Opt-out link missing.",
            "confidence": "extreme",
        }
    )

    with patch("prompt_loader.load_prompt_yaml", return_value={"prompt": "template"}), patch(
        "prompt_loader.render_prompt", return_value="rendered-v3-prompt"
    ) as render:
        result = asyncio.run(
            client.gap_check_v3(
                statute_chunk_text="statute chunk",
                policy_chunk_text="policy chunk",
            )
        )

    render.assert_called_once()
    client._call_json.assert_awaited_once_with("rendered-v3-prompt")
    assert result["status"] == "missing"
    assert result["confidence"] == "low"
    assert result["policy_quote"] is None
    assert result["statute_quote"] == ""
    assert result["requirement_summary"] == "Disclose sale opt-out"
    assert result["conflict_description"] == "Opt-out link missing."
    assert result["_analysis_failed"] is False


def test_gap_check_v3_returns_analysis_failed_fallback() -> None:
    client = _build_client()
    client._call_json = AsyncMock(return_value=None)

    with patch("prompt_loader.load_prompt_yaml", return_value={"prompt": "template"}), patch(
        "prompt_loader.render_prompt", return_value="rendered-v3-prompt"
    ):
        result = asyncio.run(client.gap_check_v3("statute", "policy"))

    assert result == {
        "status": "missing",
        "policy_quote": None,
        "statute_quote": None,
        "requirement_summary": "Requirement",
        "conflict_description": None,
        "confidence": "low",
        "_analysis_failed": True,
    }


def test_gap_check_chunks_uses_yaml_prompt_and_fallback() -> None:
    client = _build_client()
    client._call_json = AsyncMock(return_value=None)
    captured = {}

    def _render(template: str, **kwargs: str) -> str:
        captured.update(kwargs)
        return "chunk-prompt"

    with patch("prompt_loader.load_prompt_yaml", return_value={"prompt": "tpl"}), patch(
        "prompt_loader.render_prompt", side_effect=_render
    ):
        result = asyncio.run(
            client.gap_check_chunks("statute chunk text", "policy chunk text")
        )

    assert captured["STATUTE_CHUNK"] == "statute chunk text"
    assert captured["POLICY_CHUNK"] == "policy chunk text"
    client._call_json.assert_awaited_once_with("chunk-prompt")
    assert result["addressed"] is False
    assert result["missing"] is True
    assert result["conflict"] is False


def test_gap_check_subchunks_returns_safe_fallback_on_parse_failure() -> None:
    client = _build_client()
    client._call_json = AsyncMock(return_value=None)

    result = asyncio.run(
        client.gap_check_subchunks(
            statute_subchunk_text="sub-statute",
            statute_chunk_text="parent statute",
            policy_subchunk_text="sub-policy",
            policy_chunk_text="parent policy",
        )
    )

    assert result == {
        "addressed": False,
        "policy_quote": None,
        "missing": True,
        "conflict": False,
        "conflict_description": None,
    }
    prompt = client._call_json.await_args.args[0]
    assert "sub-statute" in prompt
    assert "parent statute" in prompt
    assert "sub-policy" in prompt
    assert "parent policy" in prompt


def test_applicability_rejects_non_list_jurisdictions_and_non_dict_confidence() -> None:
    client = _build_client()
    client._call_json = AsyncMock(
        return_value={"applicable_jurisdictions": "CA", "confidence": ["bad"]}
    )
    empty = asyncio.run(client.applicability("policy"))
    assert empty == {"applicable_jurisdictions": [], "confidence": {}}

    client._call_json = AsyncMock(
        return_value={"applicable_jurisdictions": ["CA", "VA"], "confidence": "bad"}
    )
    normalized = asyncio.run(client.applicability("policy"))
    assert normalized == {
        "applicable_jurisdictions": ["CA", "VA"],
        "confidence": {},
    }


def test_requirement_extraction_coerces_rows_and_falls_back() -> None:
    client = _build_client()
    client._call_json = AsyncMock(return_value={"requirements": "not-a-list"})
    assert asyncio.run(client.requirement_extraction("statute")) == {"requirements": []}

    client._call_json = AsyncMock(
        return_value={
            "requirements": [
                {"label": "Access", "description": "Provide access"},
                {"label": None, "description": None},
            ]
        }
    )
    result = asyncio.run(client.requirement_extraction("statute"))
    assert result == {
        "requirements": [
            {"label": "Access", "description": "Provide access"},
            {"label": "None", "description": "None"},
        ]
    }


def test_strictness_comparison_normalizes_other_jurisdictions() -> None:
    client = _build_client()
    client._call_json = AsyncMock(return_value=None)
    assert asyncio.run(
        client.strictness_comparison("right_to_delete", [{"jurisdiction": "CA", "description": "30 days"}])
    ) == {
        "strictest_jurisdiction": "",
        "strictest_description": "",
        "other_jurisdictions": [],
    }

    client._call_json = AsyncMock(
        return_value={
            "strictest_jurisdiction": "CA",
            "strictest_description": "30 days",
            "other_jurisdictions": "bad",
        }
    )
    result = asyncio.run(
        client.strictness_comparison(
            "right_to_delete",
            [{"jurisdiction": "CA", "description": "30 days"}],
        )
    )
    assert result["strictest_jurisdiction"] == "CA"
    assert result["other_jurisdictions"] == []


def test_policy_alignment_maps_invalid_values_to_unclear() -> None:
    client = _build_client()
    client._call_json = AsyncMock(
        return_value={"policy_alignment": "maybe", "policy_note": ""}
    )
    result = asyncio.run(client.policy_alignment("req a", "req b", "policy"))
    assert result == {"policy_alignment": "unclear", "policy_note": None}

    client._call_json = AsyncMock(return_value=None)
    assert asyncio.run(client.policy_alignment("a", "b", "p")) == {
        "policy_alignment": "unclear",
        "policy_note": None,
    }


def test_citation_check_returns_false_alignment_on_parse_failure() -> None:
    client = _build_client()
    client._call_json = AsyncMock(return_value=None)
    result = asyncio.run(
        client.citation_check(
            policy_excerpt="We sell data",
            statute_chunk_text="Sale disclosure required",
            statute_reference="§ 1798.120",
            jurisdiction="CA",
        )
    )
    assert result == {
        "alignment": False,
        "policy_quote": None,
        "statute_excerpt": None,
    }


def test_stub_llm_client_returns_configured_response() -> None:
    stub = StubLLMClient('{"compliance":"neither"}')
    result = asyncio.run(stub.compare_section("s1", "section text", []))
    assert result == '{"compliance":"neither"}'
