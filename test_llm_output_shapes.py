"""Shape contracts for LLM parsers that service tests stub out.

Risk-assessment reports, policy rewrites, and consumer-rights trees all trust
these parsers. A string where a list is expected is iterated character by
character, and consumer-rights placeholders are cut to 3000 characters even
though the caller slices them much later.
"""
from __future__ import annotations

import asyncio
import json
import sys
from typing import Any, Callable, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("anthropic", MagicMock())

from compliance_config import load_config
from compliance_suite_schemas import (
    ConsumerRightsRouterRequest,
    RiskAssessmentRequest,
)
from compliance_suite_service import ComplianceSuiteService
from llm_client import AnthropicLLMClient


def _scripted_client(responses: List[str]) -> tuple[AnthropicLLMClient, List[str], List[int]]:
    """Return a client whose messages.create yields `responses` in order."""
    prompts: List[str] = []
    max_tokens: List[int] = []
    cursor = {"n": 0}

    def create(**kwargs: Any) -> MagicMock:
        prompts.append(kwargs["messages"][0]["content"])
        max_tokens.append(kwargs["max_tokens"])
        text = responses[cursor["n"]]
        cursor["n"] += 1
        text_block = MagicMock(type="text", text=text)
        return MagicMock(content=[text_block])

    mock_sdk = MagicMock()
    mock_sdk.messages.create.side_effect = create
    with patch("llm_client.anthropic.Anthropic", return_value=mock_sdk):
        client = AnthropicLLMClient("test-key", load_config())
    return client, prompts, max_tokens


def _run(coro_factory: Callable[[AnthropicLLMClient], Any], responses: List[str]):
    client, prompts, max_tokens = _scripted_client(responses)
    result = asyncio.run(coro_factory(client))
    return result, prompts, max_tokens


def test_risk_assessment_keeps_string_and_dict_fields_and_drops_falsy_ones() -> None:
    """A string or dict is returned unchanged. Only falsy values become [].

    The risk-assessment report iterates these fields. A purpose string therefore
    becomes one bullet per character, and a risks dict becomes one bullet per key.
    """
    payload = {
        "processing_purposes": "billing",
        "data_categories": "  ",
        "risks": {"description": "over-collection"},
        "mitigations": "",
        "gaps_from_statute": 0,
    }
    result, prompts, tokens = _run(
        lambda client: client.risk_assessment("policy", "statute"),
        [json.dumps(payload)],
    )
    assert result == {
        "processing_purposes": "billing",
        "data_categories": "  ",
        "risks": {"description": "over-collection"},
        "mitigations": [],
        "gaps_from_statute": [],
    }
    assert tokens == [4096]
    assert len(prompts) == 1

    missing, _prompts, _tokens = _run(
        lambda client: client.risk_assessment("policy", "statute"),
        [json.dumps({})],
    )
    assert missing == {
        "processing_purposes": [],
        "data_categories": [],
        "risks": [],
        "mitigations": [],
        "gaps_from_statute": [],
    }


def test_risk_assessment_truncates_policy_at_8000_and_statute_at_6000() -> None:
    policy = ("P" * 8000) + "POLICY_OVERFLOW"
    statute = ("S" * 6000) + "STATUTE_OVERFLOW"
    _result, prompts, tokens = _run(
        lambda client: client.risk_assessment(policy, statute),
        [json.dumps({"processing_purposes": ["billing"]})],
    )
    assert tokens == [4096]
    assert "POLICY_OVERFLOW" not in prompts[0]
    assert "STATUTE_OVERFLOW" not in prompts[0]
    assert "P" * 8000 in prompts[0]
    assert "S" * 6000 in prompts[0]


def test_risk_assessment_parse_failure_returns_empty_lists_after_one_retry() -> None:
    result, prompts, tokens = _run(
        lambda client: client.risk_assessment("policy", "statute"),
        ["not json", "still not json"],
    )
    assert result == {
        "processing_purposes": [],
        "data_categories": [],
        "risks": [],
        "mitigations": [],
        "gaps_from_statute": [],
    }
    assert len(prompts) == 2
    assert tokens == [4096, 4096]
    assert prompts[1].endswith("\n\nOutput only valid JSON with no surrounding text.")


def test_risk_report_splits_string_fields_into_characters() -> None:
    """The markdown renderer iterates whatever the parser returned."""
    config = load_config()
    llm = MagicMock()
    llm.risk_assessment = AsyncMock(
        return_value={
            "processing_purposes": "ab",
            "data_categories": [],
            "risks": "hi",
            "mitigations": [],
            "gaps_from_statute": "gap",
        }
    )
    service = ComplianceSuiteService(
        mongo_client=MagicMock(),
        config=config,
        retriever=MagicMock(),
        llm_client=llm,
        storage=MagicMock(),
        rate_limiter=MagicMock(allow=AsyncMock(return_value=True)),
    )
    service._load_policy_text = AsyncMock(return_value=("policy body", "Acme"))
    service._fetch_v4_statute_docs = AsyncMock(return_value={})

    response = asyncio.run(
        service.risk_assessment(
            RiskAssessmentRequest(
                policy_document_id="pol-1",
                applicable_jurisdictions=["CA"],
                include_report=True,
            )
        )
    )

    assert response.assessment["processing_purposes"] == "ab"
    assert response.assessment["risks"] == "hi"
    assert response.assessment["gaps_from_statute"] == "gap"
    assert response.report == (
        "# Risk Assessment\n"
        "**Policy:** pol-1\n"
        "\n"
        "## Processing purposes\n"
        "- a\n"
        "- b\n"
        "\n"
        "## Data categories\n"
        "\n"
        "## Risks\n"
        "- h\n"
        "- i\n"
        "\n"
        "## Mitigations"
    )
    assert "gap" not in response.report
    service._storage.write_compliance_result.assert_not_called()


def test_suggest_policy_truncates_each_input_and_keeps_blank_text() -> None:
    policy = ("P" * 8000) + "POLICY_OVERFLOW"
    gap = ("G" * 3000) + "GAP_OVERFLOW"
    match = ("M" * 500) + "MATCH_OVERFLOW"
    statute = ("S" * 6000) + "STATUTE_OVERFLOW"
    result, prompts, tokens = _run(
        lambda client: client.suggest_policy(policy, gap, match, statute),
        [json.dumps({"suggested_policy_text": "", "modifications_description": "  note  "})],
    )
    assert result == {
        "suggested_policy_text": "",
        "modifications_description": "  note  ",
    }
    assert tokens == [4096]
    assert "POLICY_OVERFLOW" not in prompts[0]
    assert "GAP_OVERFLOW" not in prompts[0]
    assert "MATCH_OVERFLOW" not in prompts[0]
    assert "STATUTE_OVERFLOW" not in prompts[0]
    assert "P" * 8000 in prompts[0]
    assert "G" * 3000 in prompts[0]
    assert "M" * 500 in prompts[0]
    assert "S" * 6000 in prompts[0]


def test_suggest_policy_stringifies_description_and_drops_falsy_values() -> None:
    empty_list, _prompts, _tokens = _run(
        lambda client: client.suggest_policy("policy", "gap", "missing", "statute"),
        [json.dumps({"suggested_policy_text": "  ", "modifications_description": []})],
    )
    assert empty_list == {
        "suggested_policy_text": "  ",
        "modifications_description": "",
    }

    numeric_zero, _prompts, _tokens = _run(
        lambda client: client.suggest_policy("policy", "gap", "missing", "statute"),
        [json.dumps({"suggested_policy_text": "revised", "modifications_description": 0})],
    )
    assert numeric_zero == {
        "suggested_policy_text": "revised",
        "modifications_description": "",
    }

    description_list = ["added a retention limit"]
    listed, _prompts, _tokens = _run(
        lambda client: client.suggest_policy("policy", "gap", "partial", "statute"),
        [json.dumps({"suggested_policy_text": "revised", "modifications_description": description_list})],
    )
    assert listed == {
        "suggested_policy_text": "revised",
        "modifications_description": str(description_list),
    }


def test_consumer_rights_router_caps_every_placeholder_at_3000() -> None:
    """Caller slices policy at 12000 and statute context at 8000, then render_prompt cuts again at 3000.

    A policy that contains the jurisdictions placeholder is rewritten when that
    later value is substituted, so statute text can land inside the policy section.
    """
    policy_prefix = "before <<<JURISDICTIONS>>> after "
    policy = policy_prefix + ("P" * 3000) + "MID_POLICY" + "POLICY_TAIL"
    statute = ("S" * 3000) + "MID_STATUTE" + ("T" * 5000) + "STATUTE_8000"
    jurisdictions = "CA-SLUG" + ("J" * 3000) + "JURIS_TAIL"
    visible_policy = policy[:3000]
    visible_jurisdictions = jurisdictions[:3000]

    _result, prompts, tokens = _run(
        lambda client: client.consumer_rights_router(
            request_type="deletion",
            request_type_label="Right to Delete",
            policy_text=policy,
            statute_context=statute,
            jurisdictions=jurisdictions,
        ),
        [json.dumps({"trees": {"california": {"action": "DELETE"}}, "policy_gaps": {}})],
    )

    prompt = prompts[0]
    assert tokens == [4096]
    assert visible_policy.replace("<<<JURISDICTIONS>>>", visible_jurisdictions) in prompt
    assert "<<<JURISDICTIONS>>>" not in prompt
    assert "<<<POLICY_TEXT>>>" not in prompt
    assert "<<<STATUTE_CONTEXT>>>" not in prompt
    assert "Right to Delete" in prompt
    assert "deletion" in prompt
    assert "S" * 3000 in prompt
    assert "MID_POLICY" not in prompt
    assert "POLICY_TAIL" not in prompt
    assert "MID_STATUTE" not in prompt
    assert "STATUTE_8000" not in prompt
    assert "JURIS_TAIL" not in prompt
    assert visible_jurisdictions in prompt


def test_consumer_rights_router_requires_a_dict_of_trees() -> None:
    kept, _prompts, _tokens = _run(
        lambda client: client.consumer_rights_router("access", "Right to Access / Know", "policy", "statute", "CA"),
        [json.dumps({"trees": {}, "policy_gaps": ""})],
    )
    assert kept == {"trees": {}, "policy_gaps": {}}

    listed_gaps, _prompts, _tokens = _run(
        lambda client: client.consumer_rights_router("access", "Right to Access / Know", "policy", "statute", "CA"),
        [json.dumps({"trees": {"california": {"action": "KNOW"}}, "policy_gaps": ["california"]})],
    )
    assert listed_gaps == {
        "trees": {"california": {"action": "KNOW"}},
        "policy_gaps": ["california"],
    }

    for bad_trees in ([], "california", None):
        result, _prompts, _tokens = _run(
            lambda client: client.consumer_rights_router("access", "label", "policy", "statute", "CA"),
            [json.dumps({"trees": bad_trees, "policy_gaps": {"california": {"covered": False}}})],
        )
        assert result is None


def test_consumer_rights_list_gaps_abort_before_save() -> None:
    """A list of policy gaps has no .items(), so the router raises before it can save.

    Dict entries that are not objects are skipped. The container itself is not
    coerced, so one request type fails the whole response.
    """
    config = load_config()
    llm = MagicMock()
    llm.consumer_rights_router = AsyncMock(
        return_value={"trees": {"california": {"action": "DELETE"}}, "policy_gaps": ["california"]}
    )
    storage = MagicMock()
    service = ComplianceSuiteService(
        mongo_client=MagicMock(),
        config=config,
        retriever=MagicMock(),
        llm_client=llm,
        storage=storage,
        rate_limiter=MagicMock(allow=AsyncMock(return_value=True)),
    )
    service._fetch_v4_statute_docs = AsyncMock(return_value={})

    with pytest.raises(AttributeError):
        asyncio.run(
            service.consumer_rights_router(
                ConsumerRightsRouterRequest(
                    text="We delete personal information on request.",
                    applicable_jurisdictions=["CA"],
                    request_types=["deletion"],
                    save_results=True,
                )
            )
        )
    storage.write_compliance_result.assert_not_called()
