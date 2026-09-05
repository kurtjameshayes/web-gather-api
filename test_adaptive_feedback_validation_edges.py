"""Leftover CriticService validation and resilience contracts.

Existing adaptive-feedback tests cover valid suggestion shaping, happy-path
evaluate/record, and active_only query flags. They do not pin mixed-type
LLM arrays, index-creation failures, evaluate exceptions, unfiltered log
queries, or blank-instruction skipping in prompt rendering.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from adaptive_feedback_service import CriticService
from compliance_config import load_config


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _critic(mongo=None, config=None, llm=None) -> CriticService:
    config = config or load_config()
    config.adaptive_feedback_enabled = True
    config.adaptive_feedback_collection = "adaptive_feedback"
    config.adaptive_feedback_log_collection = "adaptive_feedback_log"
    return CriticService(
        mongo_client=mongo or MagicMock(),
        config=config,
        llm_client=llm or MagicMock(),
    )


def test_validate_suggestions_skips_non_dict_items() -> None:
    raw = [
        "not-a-dict",
        None,
        3,
        {"category": "prompt_guidance", "instruction": "Keep this", "severity": "low"},
        {"category": "false_positive", "instruction": "", "severity": "high"},
    ]
    result = CriticService._validate_suggestions(raw)
    assert result == [
        {
            "category": "prompt_guidance",
            "description": "",
            "instruction": "Keep this",
            "statute_reference": None,
            "severity": "low",
        }
    ]


def test_ensure_indexes_swallows_create_index_errors() -> None:
    mongo = MagicMock()
    coll = MagicMock()
    coll.create_index.side_effect = RuntimeError("index build failed")
    mongo.__getitem__.return_value.__getitem__.return_value = coll
    critic = _critic(mongo=mongo)

    critic.ensure_indexes()


def test_evaluate_returns_none_when_llm_raises() -> None:
    from compliance_suite_schemas import GapAnalysisResponse, GapSummary, RetrievalMetadata

    critic = _critic()
    critic._llm._call_json = AsyncMock(side_effect=RuntimeError("anthropic down"))
    response = GapAnalysisResponse(
        policy_document_id="pol-1",
        applicable_jurisdictions=["CA"],
        analyzed_at="2026-03-18T00:00:00Z",
        gaps=[],
        summary=GapSummary(),
        retrieval_metadata=RetrievalMetadata(),
        run_types=["gap_v4"],
        run_type="gap_analysis_v4",
        version="v4",
    )

    assert _run(critic.evaluate(response, "run-err")) is None


def test_get_feedback_log_without_run_id_is_unfiltered() -> None:
    mongo = MagicMock()
    coll = MagicMock()
    cursor = MagicMock()
    cursor.sort.return_value = cursor
    cursor.skip.return_value = cursor
    cursor.limit.return_value = [{"_id": "log-1", "run_id": "run-1"}]
    coll.find.return_value = cursor
    mongo.__getitem__.return_value.__getitem__.return_value = coll

    critic = _critic(mongo=mongo)
    docs = critic.get_feedback_log()
    assert docs == [{"_id": "log-1", "run_id": "run-1"}]
    coll.find.assert_called_once_with({})


def test_format_feedback_skips_blank_instructions() -> None:
    critic = _critic()
    rendered = critic.format_feedback_for_prompt(
        [
            {
                "_id": "fb-1",
                "suggestions": [
                    {"instruction": "   "},
                    {"instruction": None},
                    {"instruction": "Require verbatim quotes"},
                ],
            }
        ]
    )
    assert "PRIOR ANALYSIS FEEDBACK" in rendered
    assert "- Require verbatim quotes" in rendered
    assert rendered.count("\n- ") == 1
