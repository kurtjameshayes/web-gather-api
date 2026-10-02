"""Normalization contracts for AnthropicLLMClient suite parsers.

These methods sit between raw model text and gap-analysis status. A drift in
status, confidence, quote emptiness, or prompt truncation changes findings
without a service-level test noticing, because those tests stub the client.
"""
from __future__ import annotations

import asyncio
import json
import sys
from typing import Any, Callable, List
from unittest.mock import MagicMock, patch

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("anthropic", MagicMock())

from compliance_config import load_config
from llm_client import AnthropicLLMClient, build_prompt
from vector_retriever import StatuteCandidate


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
        tool_block = MagicMock(type="tool_use", text="not-json")
        text_block = MagicMock(type="text", text=text)
        return MagicMock(content=[tool_block, text_block])

    mock_sdk = MagicMock()
    mock_sdk.messages.create.side_effect = create
    with patch("llm_client.anthropic.Anthropic", return_value=mock_sdk):
        client = AnthropicLLMClient("test-key", load_config())
    return client, prompts, max_tokens


def _wrapped(payload: dict) -> str:
    return "analysis:\n" + json.dumps(payload) + "\nend"


def _run(coro_factory: Callable[[AnthropicLLMClient], Any], responses: List[str]):
    client, prompts, max_tokens = _scripted_client(responses)
    result = asyncio.run(coro_factory(client))
    return result, prompts, max_tokens


def test_build_prompt_empty_candidates_keeps_schema_placeholder() -> None:
    prompt = build_prompt("We retain data.", [])
    assert "We retain data." in prompt
    assert "[STATUTE_ID] [JURISDICTION] [TITLE]" in prompt
    assert "score: [SCORE]" in prompt


def test_compare_section_returns_first_text_and_retries_invalid_json_once() -> None:
    valid = 'prefix {"compliance": "neither"} suffix'
    client, prompts, _tokens = _scripted_client([valid])
    text = asyncio.run(client.compare_section("sec-1", "policy text", []))
    assert text == valid
    assert len(prompts) == 1
    assert "policy text" in prompts[0]

    recovered = '{"compliance": "compliant"}'
    client, prompts, _tokens = _scripted_client(
        ['{"compliance": "neither"} {"compliance": "extra"}', recovered]
    )
    candidate = StatuteCandidate(
        statute_id="stat-1",
        jurisdiction="CA",
        title="Access",
        section_id="1798.100",
        chunk_text="Consumers may request access.",
        score=0.5,
        chunk_id="c1",
    )
    text = asyncio.run(client.compare_section("sec-1", "policy text", [candidate]))
    assert text == recovered
    assert len(prompts) == 2
    assert prompts[1].endswith("\n\nOutput only valid JSON with no surrounding text.")
    assert "[stat-1] CA Access" in prompts[0]


def test_gap_check_bool_and_empty_quotes_and_parse_failure() -> None:
    payload = {
        "addressed": "false",
        "policy_quote": "",
        "conflict": 0,
        "conflict_description": "  ",
    }
    result, prompts, tokens = _run(
        lambda client: client.gap_check("statute", "ref", "policy"),
        [_wrapped(payload)],
    )
    assert result == {
        "addressed": True,
        "policy_quote": None,
        "missing": True,
        "conflict": False,
        "conflict_description": "  ",
    }
    assert len(prompts) == 1
    assert tokens == [load_config().llm_max_tokens]

    explicit, _prompts, _tokens = _run(
        lambda client: client.gap_check("statute", "ref", "policy"),
        [_wrapped({"addressed": False, "missing": False, "conflict": True, "policy_quote": "we sell data", "conflict_description": "sale"})],
    )
    assert explicit["addressed"] is False
    assert explicit["missing"] is False
    assert explicit["conflict"] is True
    assert explicit["policy_quote"] == "we sell data"
    assert explicit["conflict_description"] == "sale"

    failed, prompts, _tokens = _run(
        lambda client: client.gap_check("statute", "ref", "policy"),
        ["not json", "still not json"],
    )
    assert failed == {
        "addressed": False,
        "policy_quote": None,
        "missing": True,
        "conflict": False,
        "conflict_description": None,
    }
    assert len(prompts) == 2
    assert prompts[1].endswith("\n\nOutput only valid JSON with no surrounding text.")


def test_gap_check_truncates_statute_at_3000_and_policy_at_6000() -> None:
    statute = ("S" * 3000) + "STATUTE_OVERFLOW"
    policy = ("P" * 6000) + "POLICY_OVERFLOW"
    _result, prompts, _tokens = _run(
        lambda client: client.gap_check(statute, "ref", policy),
        [_wrapped({"addressed": False, "missing": True})],
    )
    assert "STATUTE_OVERFLOW" not in prompts[0]
    assert "POLICY_OVERFLOW" not in prompts[0]
    assert "S" * 3000 in prompts[0]
    assert "P" * 6000 in prompts[0]


def test_chunk_and_subchunk_gap_checks_share_bool_contract_and_3000_cap() -> None:
    payload = {"addressed": "false", "policy_quote": "", "conflict": 0}
    chunk_result, chunk_prompts, _tokens = _run(
        lambda client: client.gap_check_chunks("statute", "policy"),
        [_wrapped(payload)],
    )
    sub_result, sub_prompts, _tokens = _run(
        lambda client: client.gap_check_subchunks("sub-s", "stat-s", "sub-p", "pol-p"),
        [_wrapped(payload)],
    )
    expected = {
        "addressed": True,
        "policy_quote": None,
        "missing": True,
        "conflict": False,
        "conflict_description": None,
    }
    assert chunk_result == expected
    assert sub_result == expected

    overflow = ("X" * 3000) + "CHUNK_OVERFLOW"
    _result, prompts, _tokens = _run(
        lambda client: client.gap_check_chunks(overflow, overflow),
        [_wrapped({"addressed": False})],
    )
    assert prompts[0].count("CHUNK_OVERFLOW") == 0
    assert "X" * 3000 in prompts[0]

    _result, prompts, _tokens = _run(
        lambda client: client.gap_check_subchunks(overflow, overflow, overflow, overflow),
        [_wrapped({"addressed": False})],
    )
    assert prompts[0].count("CHUNK_OVERFLOW") == 0
    assert prompts[0].count("X" * 3000) >= 1


def test_v3_and_v4_status_confidence_and_quote_normalization() -> None:
    accepted = _wrapped(
        {
            "status": "PARTIAL",
            "policy_quote": "we share data",
            "statute_quote": "disclose sharing",
            "requirement_summary": "  disclose sharing  ",
            "gap_description": "only some sharing",
            "conflict_description": "ignored when gap_description is set",
            "confidence": "medium",
        }
    )
    client, prompts, tokens = _scripted_client([accepted, accepted])
    v3, v4 = asyncio.run(_drive(client))
    for result in (v3, v4):
        assert result["status"] == "partial"
        assert result["policy_quote"] == "we share data"
        assert result["statute_quote"] == "disclose sharing"
        assert result["requirement_summary"] == "disclose sharing"
        assert result["conflict_description"] == "only some sharing"
        assert result["confidence"] == "medium"
        assert result["_analysis_failed"] is False
    assert tokens == [load_config().llm_max_tokens, load_config().llm_max_tokens]
    assert "Require verbatim quotes." in prompts[1]
    assert "<<<ADAPTIVE_FEEDBACK>>>" not in prompts[1]
    assert "<<<POLICY_TEXT>>>" not in prompts[1]


async def _drive(client: AnthropicLLMClient):
    v3 = await client.gap_check_v3("statute", "policy")
    v4 = await client.gap_check_v4(
        "definitions",
        "must disclose sale",
        "policy",
        adaptive_feedback="Require verbatim quotes.",
    )
    return v3, v4


def _v3_v4(responses: List[str]):
    client, prompts, _tokens = _scripted_client(responses)

    async def both():
        return (
            await client.gap_check_v3("statute", "policy"),
            await client.gap_check_v4("ctx", "req", "policy", adaptive_feedback=""),
        )

    return asyncio.run(both()), prompts


def test_v3_and_v4_unknown_status_is_missing_without_analysis_failure() -> None:
    padded = _wrapped({"status": " addressed ", "confidence": "high", "policy_quote": "quote"})
    unknown = _wrapped({"status": "conflicted", "confidence": "HIGH", "requirement_summary": "   "})
    (v3_pad, v4_pad), _prompts = _v3_v4([padded, padded])
    (v3_unknown, v4_unknown), _prompts = _v3_v4([unknown, unknown])
    for result in (v3_pad, v4_pad):
        assert result["status"] == "missing"
        assert result["confidence"] == "high"
        assert result["policy_quote"] == "quote"
        assert result["_analysis_failed"] is False
    for result in (v3_unknown, v4_unknown):
        assert result["status"] == "missing"
        assert result["confidence"] == "low"
        assert result["requirement_summary"] == ""
        assert result["statute_quote"] == ""
        assert result["policy_quote"] is None
        assert result["conflict_description"] is None
        assert result["_analysis_failed"] is False


def test_v3_and_v4_empty_gap_description_falls_through_and_numeric_confidence_is_low() -> None:
    payload = _wrapped(
        {
            "status": "conflict",
            "gap_description": "",
            "conflict_description": "policy allows sale",
            "confidence": 0.95,
            "policy_quote": "we may sell",
            "statute_quote": 0,
            "requirement_summary": None,
        }
    )
    (v3, v4), _prompts = _v3_v4([payload, payload])
    for result in (v3, v4):
        assert result["status"] == "conflict"
        assert result["conflict_description"] == "policy allows sale"
        assert result["confidence"] == "low"
        assert result["statute_quote"] == ""
        assert result["requirement_summary"] == "Requirement"
        assert result["_analysis_failed"] is False


def test_v3_and_v4_parse_failure_sets_analysis_failed_after_one_retry() -> None:
    greedy = '{"status": "addressed", "confidence": "high"} {"status": "conflict"}'
    recovered = _wrapped(
        {
            "status": "ambiguous",
            "confidence": "low",
            "gap_description": "unclear opt-out",
            "policy_quote": "maybe",
        }
    )
    (v3, v4), prompts = _v3_v4([greedy, recovered, greedy, recovered])
    for result in (v3, v4):
        assert result["status"] == "ambiguous"
        assert result["conflict_description"] == "unclear opt-out"
        assert result["_analysis_failed"] is False
    assert len(prompts) == 4
    assert prompts[1].endswith("\n\nOutput only valid JSON with no surrounding text.")
    assert prompts[3].endswith("\n\nOutput only valid JSON with no surrounding text.")

    (failed_v3, failed_v4), prompts = _v3_v4(["no braces", "no braces", "no braces", "no braces"])
    for result in (failed_v3, failed_v4):
        assert result == {
            "status": "missing",
            "policy_quote": None,
            "statute_quote": None,
            "requirement_summary": "Requirement",
            "conflict_description": None,
            "confidence": "low",
            "_analysis_failed": True,
        }
    assert len(prompts) == 4


def test_v4_prompt_truncates_feedback_context_requirement_and_policy_at_3000() -> None:
    overflow = "OVERFLOW"
    feedback = ("F" * 3000) + overflow
    context = ("C" * 3000) + overflow
    requirement = ("R" * 3000) + overflow
    policy = ("P" * 3000) + overflow
    _result, prompts, _tokens = _run(
        lambda client: client.gap_check_v4(context, requirement, policy, adaptive_feedback=feedback),
        [_wrapped({"status": "missing", "confidence": "low"})],
    )
    assert overflow not in prompts[0]
    assert "F" * 3000 in prompts[0]
    assert "C" * 3000 in prompts[0]
    assert "R" * 3000 in prompts[0]
    assert "P" * 3000 in prompts[0]


def test_applicability_rejects_non_list_and_non_dict_confidence() -> None:
    dropped, prompts, _tokens = _run(
        lambda client: client.applicability("A" * 8000 + "POLICY_OVERFLOW"),
        [_wrapped({"applicable_jurisdictions": "CA", "confidence": {"CA": 0.9}})],
    )
    assert dropped == {"applicable_jurisdictions": [], "confidence": {}}
    assert "POLICY_OVERFLOW" not in prompts[0]
    assert "A" * 8000 in prompts[0]

    kept, _prompts, _tokens = _run(
        lambda client: client.applicability("short policy"),
        [_wrapped({"applicable_jurisdictions": ["CA", None], "confidence": "high"})],
    )
    assert kept == {"applicable_jurisdictions": ["CA", None], "confidence": {}}


def test_requirement_extraction_stringifies_missing_fields_and_truncates() -> None:
    result, prompts, _tokens = _run(
        lambda client: client.requirement_extraction(("S" * 3000) + "STATUTE_OVERFLOW"),
        [
            _wrapped(
                {
                    "requirements": [
                        {"label": "Access"},
                        {"description": None},
                        {},
                    ]
                }
            )
        ],
    )
    assert result == {
        "requirements": [
            {"label": "Access", "description": ""},
            {"label": "", "description": "None"},
            {"label": "", "description": ""},
        ]
    }
    assert "STATUTE_OVERFLOW" not in prompts[0]

    empty, _prompts, _tokens = _run(
        lambda client: client.requirement_extraction("statute"),
        [_wrapped({"requirements": "not-a-list"})],
    )
    assert empty == {"requirements": []}


def test_strictness_comparison_coerces_fields_and_keeps_full_canonical_id() -> None:
    canonical = "CANON-" + ("I" * 4000)
    result, prompts, _tokens = _run(
        lambda client: client.strictness_comparison(
            canonical,
            [{"jurisdiction": "CA", "description": "broad"}, {"description": "no code"}],
        ),
        [
            _wrapped(
                {
                    "strictest_jurisdiction": None,
                    "strictest_description": 3,
                    "other_jurisdictions": {"jurisdiction": "VA"},
                }
            )
        ],
    )
    assert result == {
        "strictest_jurisdiction": "None",
        "strictest_description": "3",
        "other_jurisdictions": [],
    }
    assert canonical in prompts[0]
    assert "- CA: broad" in prompts[0]
    assert "- : no code" in prompts[0]

    empty_block, prompts, _tokens = _run(
        lambda client: client.strictness_comparison("access", []),
        [_wrapped({})],
    )
    assert empty_block["other_jurisdictions"] == []
    assert "Jurisdiction formulations:\nNone" in prompts[0]


def test_policy_alignment_rejects_unknown_enum_and_truncates_inputs() -> None:
    req_a = ("A" * 1500) + "REQ_A_OVERFLOW"
    req_b = ("B" * 1500) + "REQ_B_OVERFLOW"
    excerpt = ("E" * 3000) + "EXCERPT_OVERFLOW"
    result, prompts, _tokens = _run(
        lambda client: client.policy_alignment(req_a, req_b, excerpt),
        [_wrapped({"policy_alignment": "Satisfies_all", "policy_note": ""})],
    )
    assert result == {"policy_alignment": "unclear", "policy_note": None}
    assert "REQ_A_OVERFLOW" not in prompts[0]
    assert "REQ_B_OVERFLOW" not in prompts[0]
    assert "EXCERPT_OVERFLOW" not in prompts[0]
    assert "A" * 1500 in prompts[0]
    assert "E" * 3000 in prompts[0]

    kept, _prompts, _tokens = _run(
        lambda client: client.policy_alignment("a", "b", "policy"),
        [_wrapped({"policy_alignment": "satisfies_strictest_only", "policy_note": "  note  "})],
    )
    assert kept == {"policy_alignment": "satisfies_strictest_only", "policy_note": "  note  "}


def test_citation_check_bool_alignment_and_truncation() -> None:
    excerpt = ("P" * 2000) + "POLICY_OVERFLOW"
    statute = ("S" * 3000) + "STATUTE_OVERFLOW"
    result, prompts, _tokens = _run(
        lambda client: client.citation_check(excerpt, statute, "1798.100", "CA"),
        [_wrapped({"alignment": "false", "policy_quote": "", "statute_excerpt": ""})],
    )
    assert result == {"alignment": True, "policy_quote": None, "statute_excerpt": None}
    assert "POLICY_OVERFLOW" not in prompts[0]
    assert "STATUTE_OVERFLOW" not in prompts[0]
    assert "(CA)" in prompts[0] or "CA" in prompts[0]
    assert "P" * 2000 in prompts[0]
    assert "S" * 3000 in prompts[0]

    failed, prompts, _tokens = _run(
        lambda client: client.citation_check("policy", "statute", "ref", ""),
        ["no json", "no json"],
    )
    assert failed == {"alignment": False, "policy_quote": None, "statute_excerpt": None}
    assert len(prompts) == 2
