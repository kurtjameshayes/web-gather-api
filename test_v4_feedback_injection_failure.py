"""V4 gap analysis when adaptive feedback renders but cannot be identified.

A successful lookup can still fail while the prompt text is being attached.
Formatting and id extraction share one try block, and they assign different
state. These tests pin that split so a later cleanup does not drop rendered
instructions, record ids that were never captured, or abort the analysis.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from compliance_config import load_config
from compliance_suite_schemas import GapAnalysisRequest
from gap_analysis_service_v4 import GapAnalysisServiceV4

POLICY_TEXT = "We collect personal information."
RENDERED_FEEDBACK = (
    "PRIOR ANALYSIS FEEDBACK (incorporate these lessons into your analysis):\n\n"
    "- Quote the policy verbatim"
)


class _AllowAll:
    async def allow(self) -> bool:
        return True


def _mongo():
    collections: dict[str, MagicMock] = {}

    def get_db(_name: str) -> MagicMock:
        database = MagicMock()

        def get_coll(name: str) -> MagicMock:
            coll = collections.get(name)
            if coll is None:
                coll = MagicMock()
                collections[name] = coll
            return coll

        database.__getitem__.side_effect = get_coll
        return database

    client = MagicMock()
    client.__getitem__.side_effect = get_db
    return client, collections


def _coll(collections: dict[str, MagicMock], name: str) -> MagicMock:
    coll = collections.get(name)
    if coll is None:
        coll = MagicMock()
        collections[name] = coll
    return coll


def _wire_corpus(collections: dict[str, MagicMock]) -> None:
    _coll(collections, "policies").find_one.return_value = {
        "document_id": "pol-1",
        "text": POLICY_TEXT,
        "company_name": "Acme",
    }
    legal = _coll(collections, "policy_legal_embeddings")
    legal.count_documents.return_value = 1
    legal.find.return_value = [{"chunk_text": POLICY_TEXT}]
    _coll(collections, "category_mapping").find.return_value = [
        {
            "statute_category": "consumer_rights",
            "policy_categories": ["collection"],
            "sub_topic": "disclosure",
        }
    ]

    statute = {
        "_id": "stat-1",
        "document_id": "statute-doc",
        "header_text": "Civil Code 1798.100",
        "subtopic_text": "Disclose collection.",
        "jurisdiction": "CA",
        "category": "consumer_rights",
    }

    def statute_find(query, *args, **kwargs):
        if query.get("category") == "consumer_rights":
            return [statute]
        return []

    _coll(collections, "statute_sub_topic_embeddings").find.side_effect = statute_find


def _service():
    config = load_config()
    config.adaptive_feedback_enabled = True
    mongo, collections = _mongo()
    _wire_corpus(collections)
    llm = MagicMock()

    async def gap_check_v4(**kwargs):
        llm.calls.append(kwargs)
        return {
            "status": "addressed",
            "policy_quote": POLICY_TEXT,
            "statute_quote": "Disclose collection.",
            "requirement_summary": "Disclose collection",
            "conflict_description": None,
            "confidence": "high",
        }

    llm.calls = []
    llm.gap_check_v4 = gap_check_v4
    critic = MagicMock()
    critic.record_feedback_usage = AsyncMock()
    critic.evaluate = AsyncMock(return_value="fb-new")
    service = GapAnalysisServiceV4(mongo, config, llm, _AllowAll(), critic=critic)
    return service, llm, critic, collections


def _request() -> GapAnalysisRequest:
    return GapAnalysisRequest(
        policy_document_id="pol-1",
        applicable_jurisdictions=["CA"],
        save_results=True,
        run_async=False,
    )


def test_missing_feedback_id_still_injects_rendered_text() -> None:
    """A doc without _id is rendered, then id extraction fails.

    The prompt keeps the rendered text. No id is logged, including an earlier
    doc that did have an id, because the list comprehension aborts. The run
    still persists and the critic still evaluates.
    """
    service, llm, critic, collections = _service()
    critic.get_active_feedback.return_value = [
        {"_id": "fb-1", "suggestions": [{"instruction": "Quote the policy verbatim"}]},
        {"suggestions": [{"instruction": "Do not paraphrase"}]},
    ]
    critic.format_feedback_for_prompt.return_value = RENDERED_FEEDBACK

    response = asyncio.run(service.run(_request()))

    assert llm.calls[0]["adaptive_feedback"] == RENDERED_FEEDBACK
    critic.get_active_feedback.assert_called_once_with("pol-1")
    critic.record_feedback_usage.assert_not_called()
    critic.evaluate.assert_awaited_once()
    run_id = critic.evaluate.await_args.args[1]
    assert isinstance(run_id, str) and run_id
    assert response.gaps[0].status == "addressed"
    assert response.summary.addressed == 1
    stored = collections["compliance_results"].insert_one.call_args[0][0]
    assert stored["_id"] == run_id
    assert stored["policy_document_id"] == "pol-1"


def test_feedback_format_error_omits_prompt_and_continues() -> None:
    """If rendering itself raises, the prompt stays empty and the run continues.

    Lookup already succeeded, so this is not a retrieval failure. Usage is not
    recorded, and the critic still evaluates the completed run.
    """
    service, llm, critic, collections = _service()
    critic.get_active_feedback.return_value = [
        {"_id": "fb-1", "suggestions": ["not-a-suggestion"]}
    ]
    critic.format_feedback_for_prompt.side_effect = AttributeError(
        "suggestion must be a mapping"
    )

    response = asyncio.run(service.run(_request()))

    assert llm.calls[0]["adaptive_feedback"] == ""
    critic.record_feedback_usage.assert_not_called()
    critic.evaluate.assert_awaited_once()
    assert response.gaps[0].status == "addressed"
    collections["compliance_results"].insert_one.assert_called_once()
    collections["compliance_run_log"].insert_one.assert_called_once()
