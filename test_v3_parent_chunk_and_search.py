"""Regression tests for v3 parent-chunk joins and vector-search bounds.

Integer parent id 0 is fetched for policy chunks and then dropped from the
LLM policy block because the join key treats 0 as missing. Statute parent id
0 skips the parent read entirely. The inline v3 search uses ``max(500, top_k
* 500)`` for both limit and numCandidates and ignores
``gap_analysis_v3_num_candidates``. A filtered search error falls back to an
unfiltered search plus ``$match``; a second failure is not swallowed.
"""
from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.modules.setdefault("anthropic", MagicMock())

from compliance_config import load_config
from compliance_suite_schemas import GapAnalysisRequest
from gap_analysis_service_v3 import GapAnalysisServiceV3


PARENT_BODY = "ENCLOSING POLICY SECTION"
POLICY_SUB = "We delete data on request."
STATUTE_SUB = "Consumers may request deletion."


class _Allow:
    async def allow(self) -> bool:
        return True


def _config():
    cfg = load_config()
    cfg.compliance_database = "privacy-compliance"
    cfg.statute_database = ""
    cfg.policies_collection = "policies"
    cfg.policy_sub_embeddings_collection = "policy_sub_embeddings"
    cfg.statute_sub_embeddings_collection = "statute_sub_embeddings"
    cfg.statute_embeddings_collection = "statute_embeddings"
    cfg.policy_embeddings_collection = "policy_embeddings"
    cfg.policy_document_id_field = "document_id"
    cfg.statute_jurisdiction_field = "jurisdiction"
    cfg.embedding_vector_field = "embedding"
    cfg.embedding_text_field = "chunk_text"
    cfg.statute_subchunk_text_field = "sub_chunk_text"
    cfg.policy_subchunk_text_field = "sub_chunk_text"
    cfg.policy_chunk_text_field = "chunk_text"
    cfg.statute_chunk_text_field = "chunk_text"
    cfg.vector_index_name = "vector_index"
    cfg.gap_analysis_v3_top_k = 1
    cfg.gap_analysis_v3_num_candidates = 50
    cfg.gap_analysis_v3_score_threshold = 0.99
    return cfg


def _mongo():
    collections: dict[str, MagicMock] = {
        name: MagicMock()
        for name in (
            "policies",
            "policy_sub_embeddings",
            "statute_sub_embeddings",
            "statute_embeddings",
            "policy_embeddings",
            "compliance_results",
            "compliance_run_log",
        )
    }

    def get_collection(name: str) -> MagicMock:
        collections.setdefault(name, MagicMock())
        return collections[name]

    database = MagicMock()
    database.__getitem__.side_effect = get_collection
    mongo = MagicMock()
    mongo.__getitem__.return_value = database
    collections["policies"].find_one.return_value = {
        "document_id": "pol-1",
        "text": "We retain personal information for one year.",
        "company_name": "Acme",
    }
    collections["policy_sub_embeddings"].count_documents.return_value = 1
    return mongo, collections


def _service(mongo, llm=None):
    if llm is None:
        llm = MagicMock()
        llm.gap_check_v3 = AsyncMock(
            return_value={
                "status": "missing",
                "policy_quote": None,
                "statute_quote": "quote",
                "requirement_summary": "Deletion",
                "conflict_description": "gap",
                "confidence": "low",
                "_analysis_failed": False,
            }
        )
    service = GapAnalysisServiceV3(
        mongo_client=mongo,
        config=_config(),
        retriever=MagicMock(),
        llm_client=llm,
        storage=MagicMock(),
        rate_limiter=_Allow(),
    )
    return service, llm


def _request(**overrides) -> GapAnalysisRequest:
    payload = {
        "policy_document_id": "pol-1",
        "applicable_jurisdictions": ["CA"],
        "save_results": False,
        "run_async": False,
    }
    payload.update(overrides)
    return GapAnalysisRequest(**payload)


@pytest.mark.parametrize(
    ("parent_id", "fetches_statute_parent", "policy_prompt_includes_parent"),
    [
        (0, False, False),
        ("0", True, True),
        (7, True, True),
    ],
)
def test_parent_chunk_id_zero_drops_policy_context(
    parent_id, fetches_statute_parent, policy_prompt_includes_parent
) -> None:
    mongo, collections = _mongo()
    collections["statute_sub_embeddings"].find.return_value = [
        {
            "_id": "stat-1",
            "embedding": [0.2, 0.4],
            "sub_chunk_text": STATUTE_SUB,
            "statute_reference": "1798.105",
            "jurisdiction": "CA",
            "parent_chunk_id": parent_id,
        }
    ]
    collections["policy_sub_embeddings"].aggregate.return_value = [
        {
            "sub_chunk_text": POLICY_SUB,
            "parent_chunk_id": parent_id,
            "score": 0.11,
        }
    ]
    collections["policy_embeddings"].find.return_value = [
        {"_id": parent_id, "chunk_text": PARENT_BODY}
    ]
    collections["statute_embeddings"].find_one.return_value = {
        "_id": parent_id,
        "chunk_text": "STATUTE PARENT",
    }
    service, llm = _service(mongo)

    asyncio.run(service.run(_request()))

    policy_find = collections["policy_embeddings"].find.call_args[0][0]
    assert policy_find == {"_id": {"$in": [parent_id]}}
    if fetches_statute_parent:
        collections["statute_embeddings"].find_one.assert_called_once()
        assert collections["statute_embeddings"].find_one.call_args[0][0] == {
            "_id": parent_id
        }
    else:
        collections["statute_embeddings"].find_one.assert_not_called()

    policy_prompt = llm.gap_check_v3.await_args.kwargs["policy_chunk_text"]
    if policy_prompt_includes_parent:
        assert policy_prompt == f"{PARENT_BODY}\n\n{POLICY_SUB}"
    else:
        assert policy_prompt == POLICY_SUB
        assert PARENT_BODY not in policy_prompt

    pipeline = collections["policy_sub_embeddings"].aggregate.call_args[0][0]
    stage = pipeline[0]["$vectorSearch"]
    assert stage["limit"] == 500
    assert stage["numCandidates"] == 500
    assert stage["numCandidates"] != service.cfg.gap_analysis_v3_num_candidates
    assert pipeline[2] == {"$limit": 1}
    assert stage["filter"] == {"document_id": "pol-1"}


def test_filtered_search_error_falls_back_to_match() -> None:
    mongo, collections = _mongo()
    collections["statute_sub_embeddings"].find.return_value = [
        {
            "_id": "stat-1",
            "embedding": [1, 0],
            "sub_chunk_text": STATUTE_SUB,
            "statute_reference": "1798.105",
            "jurisdiction": "CA",
        }
    ]
    collections["policy_sub_embeddings"].aggregate.side_effect = [
        RuntimeError("filter unsupported"),
        [{"sub_chunk_text": "fallback hit", "score": 0.4}],
    ]
    service, llm = _service(mongo)

    asyncio.run(service.run(_request()))

    calls = collections["policy_sub_embeddings"].aggregate.call_args_list
    assert len(calls) == 2
    assert "filter" in calls[0].args[0][0]["$vectorSearch"]
    fallback = calls[1].args[0]
    assert "filter" not in fallback[0]["$vectorSearch"]
    assert fallback[2] == {"$match": {"document_id": "pol-1"}}
    assert fallback[3] == {"$limit": 1}
    assert llm.gap_check_v3.await_args.kwargs["policy_chunk_text"] == "fallback hit"


def test_second_search_failure_skips_persist() -> None:
    mongo, collections = _mongo()
    collections["statute_sub_embeddings"].find.return_value = [
        {
            "_id": "stat-1",
            "embedding": [1, 0],
            "sub_chunk_text": STATUTE_SUB,
            "jurisdiction": "CA",
        }
    ]
    collections["policy_sub_embeddings"].aggregate.side_effect = RuntimeError("disk down")
    service, _llm = _service(mongo)

    with pytest.raises(RuntimeError, match="disk down"):
        asyncio.run(service.run(_request(save_results=True)))

    assert collections["policy_sub_embeddings"].aggregate.call_count == 2
    collections["compliance_results"].insert_one.assert_not_called()
    collections["compliance_run_log"].insert_one.assert_not_called()


@pytest.mark.parametrize(
    ("statute_document_id", "expected"),
    [
        (None, None),
        ("", None),
        ("   ", "   "),
        ("doc-9", "doc-9"),
    ],
)
def test_statute_document_id_keeps_whitespace(statute_document_id, expected) -> None:
    mongo, collections = _mongo()
    collections["statute_sub_embeddings"].find.return_value = []
    service, llm = _service(mongo)
    overrides = {}
    if statute_document_id is not None:
        overrides["statute_document_id"] = statute_document_id

    asyncio.run(service.run(_request(**overrides)))

    query = collections["statute_sub_embeddings"].find.call_args[0][0]
    if expected is None:
        assert "document_id" not in query
    else:
        assert query["document_id"] == expected
    llm.gap_check_v3.assert_not_called()
