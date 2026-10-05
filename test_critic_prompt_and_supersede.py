"""Critic prompt payload and feedback-supersede contracts.

Happy-path evaluate tests store one valid suggestion. They do not prove that
bulky statute/policy text is kept out of the critic prompt, that a sanitized
suggestion is what gets persisted, or that an empty critic object skips the
write while a suggestion-less object still retires prior feedback.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from adaptive_feedback_service import CriticService
from compliance_config import load_config
from compliance_suite_schemas import (
    GapAnalysisResponse,
    GapItem,
    GapSummary,
    RetrievalMetadata,
)

_BULKY = {
    "policy_subchunk_text": "BULKY_POLICY_SUBCHUNK_SHOULD_NOT_LEAK",
    "policy_combined_sections": "BULKY_POLICY_SECTIONS_SHOULD_NOT_LEAK",
    "statute_subchunk_text": "BULKY_STATUTE_SUBCHUNK_SHOULD_NOT_LEAK",
    "statute_chunk_text": "BULKY_STATUTE_CHUNK_SHOULD_NOT_LEAK",
}


def _run(coro):
    return asyncio.run(coro)


async def _sync_run(func, *args, **kwargs):
    return func(*args, **kwargs)


def _critic(mongo=None) -> CriticService:
    config = load_config()
    config.adaptive_feedback_enabled = True
    config.adaptive_feedback_max_items = 5
    config.adaptive_feedback_collection = "adaptive_feedback"
    config.adaptive_feedback_log_collection = "adaptive_feedback_log"
    config.compliance_database = "privacy-compliance"
    return CriticService(
        mongo_client=mongo or MagicMock(),
        config=config,
        llm_client=MagicMock(),
    )


def _response() -> GapAnalysisResponse:
    return GapAnalysisResponse(
        policy_document_id="pol-1",
        policy_document_ids=["pol-1", "pol-2"],
        applicable_jurisdictions=["CA"],
        analyzed_at="2026-03-18T00:00:00Z",
        gaps=[
            GapItem(
                jurisdiction="CA",
                statute_reference="§ 1798.100",
                requirement_summary="Disclose sale of personal information",
                status="addressed",
                policy_quote="QUOTE_WE_DISCLOSE_SALE",
                confidence="high",
                **_BULKY,
            )
        ],
        summary=GapSummary(total_requirements=2, missing=0, addressed=1, conflicts=13),
        retrieval_metadata=RetrievalMetadata(statute_items_considered=2),
        run_types=["gap_v4"],
        run_type="gap_analysis_v4",
        version="v4",
    )


def _wire_collection(critic: CriticService) -> MagicMock:
    coll = MagicMock()
    critic._mongo.__getitem__.return_value.__getitem__.return_value = coll
    return coll


def test_evaluate_omits_bulky_gap_text_from_critic_prompt() -> None:
    critic = _critic()
    coll = _wire_collection(critic)
    captured: dict = {}

    async def _capture(prompt, max_tokens=2048):
        captured["prompt"] = prompt
        captured["max_tokens"] = max_tokens
        return {
            "summary_evaluation": "ok",
            "suggestions": [
                {
                    "category": "citation_quality",
                    "instruction": "Quote the sale disclosure verbatim",
                    "severity": "medium",
                }
            ],
        }

    critic._llm._call_json = _capture

    with patch("adaptive_feedback_service.run_in_thread", side_effect=_sync_run):
        feedback_id = _run(critic.evaluate(_response(), "run-prompt"))

    assert feedback_id is not None
    prompt = captured["prompt"]
    assert captured["max_tokens"] == 2048
    assert "Disclose sale of personal information" in prompt
    assert "QUOTE_WE_DISCLOSE_SALE" in prompt
    assert '"conflicts": 13' in prompt
    for marker in _BULKY.values():
        assert marker not in prompt
    coll.insert_one.assert_called_once()


def test_evaluate_persists_sanitized_suggestions_and_supersedes_prior() -> None:
    critic = _critic()
    coll = _wire_collection(critic)
    critic._llm._call_json = AsyncMock(
        return_value={
            "summary_evaluation": "  Needs tighter quotes.  ",
            "suggestions": [
                {
                    "category": " False_Positive ",
                    "description": "  vague quote  ",
                    "instruction": "  Require verbatim quotes  ",
                    "statute_reference": "",
                    "severity": " HIGH ",
                },
                {
                    "category": "made_up",
                    "instruction": "Ignore the statute",
                    "severity": "high",
                },
            ],
        }
    )

    with patch("adaptive_feedback_service.run_in_thread", side_effect=_sync_run):
        feedback_id = _run(critic.evaluate(_response(), "run-sanitize"))

    assert feedback_id is not None
    inserted = coll.insert_one.call_args[0][0]
    assert inserted["suggestions"] == [
        {
            "category": "false_positive",
            "description": "vague quote",
            "instruction": "Require verbatim quotes",
            "statute_reference": None,
            "severity": "high",
        }
    ]
    assert "Ignore the statute" not in str(inserted["suggestions"])
    assert inserted["summary_evaluation"] == "Needs tighter quotes."
    assert inserted["policy_document_id"] == "pol-1"
    assert inserted["policy_document_ids"] == ["pol-1", "pol-2"]
    assert inserted["superseded_by"] is None
    coll.update_many.assert_called_once_with(
        {
            "policy_document_id": "pol-1",
            "superseded_by": None,
            "_id": {"$ne": feedback_id},
        },
        {"$set": {"superseded_by": feedback_id}},
    )


def test_evaluate_empty_object_does_not_retire_prior_feedback() -> None:
    critic = _critic()
    coll = _wire_collection(critic)
    critic._llm._call_json = AsyncMock(return_value={})

    result = _run(critic.evaluate(_response(), "run-empty"))

    assert result is None
    coll.insert_one.assert_not_called()
    coll.update_many.assert_not_called()


def test_evaluate_suggestionless_object_still_supersedes_prior_feedback() -> None:
    critic = _critic()
    coll = _wire_collection(critic)
    critic._llm._call_json = AsyncMock(
        return_value={"summary_evaluation": "No actionable issues.", "suggestions": None}
    )

    with patch("adaptive_feedback_service.run_in_thread", side_effect=_sync_run):
        feedback_id = _run(critic.evaluate(_response(), "run-no-suggestions"))

    assert feedback_id is not None
    inserted = coll.insert_one.call_args[0][0]
    assert inserted["suggestions"] == []
    assert inserted["summary_evaluation"] == "No actionable issues."
    coll.update_many.assert_called_once()
    assert coll.update_many.call_args[0][1] == {"$set": {"superseded_by": feedback_id}}


def test_get_active_feedback_uses_config_cap_and_stringifies_ids() -> None:
    critic = _critic()
    coll = MagicMock()
    cursor = MagicMock()
    cursor.sort.return_value = cursor
    cursor.limit.return_value = [{"_id": 99, "suggestions": [{"instruction": "Keep quotes"}]}]
    coll.find.return_value = cursor
    critic._mongo.__getitem__.return_value.__getitem__.return_value = coll

    docs = critic.get_active_feedback("pol-1")

    assert docs[0]["_id"] == "99"
    coll.find.assert_called_once_with(
        {"policy_document_id": "pol-1", "superseded_by": None}
    )
    cursor.sort.assert_called_once_with("created_at", -1)
    cursor.limit.assert_called_once_with(5)

    cursor.limit.reset_mock()
    cursor.limit.return_value = []
    critic.get_active_feedback("pol-1", limit=2)
    cursor.limit.assert_called_once_with(2)

    cursor.limit.reset_mock()
    critic.get_active_feedback("pol-1", limit=0)
    cursor.limit.assert_called_once_with(5)
