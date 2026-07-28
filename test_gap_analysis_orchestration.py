"""Regression tests for ComplianceSuiteService.gap_analysis and gap_analysis_v2."""
from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Lightweight unit tests must not require the Anthropic SDK at import time.
sys.modules.setdefault("anthropic", MagicMock())

from compliance_config import load_config
from compliance_suite_schemas import GapAnalysisRequest
from compliance_suite_service import ComplianceSuiteService, ComplianceSuiteServiceError
from vector_retriever import (
    ChunkPair,
    RetrievePolicyChunksResult,
    RetrievePolicySubchunksResult,
    SubchunkPair,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _service(**overrides: object) -> ComplianceSuiteService:
    config = load_config()
    config.default_jurisdictions = ["CA"]
    config.max_statute_quote_chars = 300
    rate_limiter = MagicMock()
    rate_limiter.allow = AsyncMock(return_value=True)
    storage = MagicMock()
    storage.write_compliance_result = AsyncMock(return_value=None)
    storage.write_compliance_run_log = AsyncMock(return_value=None)
    llm_client = MagicMock()
    retriever = MagicMock()
    kwargs = {
        "mongo_client": MagicMock(),
        "config": config,
        "retriever": retriever,
        "llm_client": llm_client,
        "storage": storage,
        "rate_limiter": rate_limiter,
        "gap_analysis_v4_service": None,
    }
    kwargs.update(overrides)
    return ComplianceSuiteService(**kwargs)


def _subchunk_pair(
    *,
    statute_ref: str = "ccpa-100",
    subchunk_id: str = "sc-1",
    jurisdiction: str = "CA",
    statute_sub: str = "Consumers have the right to know. More statute context follows here.",
    statute_chunk: str = "Full statute chunk.",
    policy_sub: str = "We disclose categories of personal information we collect.",
    policy_chunk: str = "Full policy chunk with disclosure language.",
) -> SubchunkPair:
    return SubchunkPair(
        statute_doc={
            "document_id": statute_ref,
            "subchunk_id": subchunk_id,
            "jurisdiction": jurisdiction,
            "sub_chunk_text": statute_sub,
            "chunk_text": statute_chunk,
        },
        policy_doc={
            "sub_chunk_text": policy_sub,
            "chunk_text": policy_chunk,
        },
        score=0.9,
    )


def _chunk_pair(
    *,
    statute_ref: str = "ccpa-100",
    chunk_id: str = "0",
    jurisdiction: str = "CA",
    statute_text: str = "Consumers have the right to delete. Extra statute text.",
    policy_text: str = "Users may request deletion of personal data.",
) -> ChunkPair:
    return ChunkPair(
        statute_doc={
            "document_id": statute_ref,
            "chunk_index": chunk_id,
            "jurisdiction": jurisdiction,
            "chunk_text": statute_text,
            "text": statute_text,
        },
        policy_doc={"chunk_text": policy_text},
        score=0.85,
    )


# ----- gap_analysis (v1 subchunk) -----


@pytest.mark.anyio
async def test_gap_analysis_rate_limit_and_missing_policy_id() -> None:
    rate_limiter = MagicMock()
    rate_limiter.allow = AsyncMock(return_value=False)
    service = _service(rate_limiter=rate_limiter)

    with pytest.raises(ComplianceSuiteServiceError) as rate_exc:
        await service.gap_analysis(
            GapAnalysisRequest(policy_document_id="policy-1", save_results=False)
        )
    assert rate_exc.value.status_code == 429

    service = _service()
    with pytest.raises(ComplianceSuiteServiceError) as missing_exc:
        # Bypass Pydantic validator to exercise service-level guard.
        req = GapAnalysisRequest.model_construct(policy_document_id=None)
        await service.gap_analysis(req)
    assert missing_exc.value.status_code == 400
    assert "policy_document_id is required" in str(missing_exc.value)


@pytest.mark.anyio
async def test_gap_analysis_status_mapping_citation_binding_and_defaults() -> None:
    """Map LLM outcomes, downgrade unbound quotes, and default missing conflict text."""
    pairs = [
        _subchunk_pair(subchunk_id="addr", policy_sub="We disclose categories of personal information we collect."),
        _subchunk_pair(subchunk_id="bind", policy_sub="We disclose categories of personal information we collect."),
        _subchunk_pair(subchunk_id="conf", statute_sub="Sale of personal information requires opt-out."),
        _subchunk_pair(subchunk_id="miss", statute_sub="Provide a privacy notice at collection."),
        _subchunk_pair(subchunk_id="fail", statute_sub="Respond to access requests within 45 days."),
        # Duplicate key should be skipped after first occurrence.
        _subchunk_pair(subchunk_id="addr", policy_sub="We disclose categories of personal information we collect."),
    ]
    retriever = MagicMock()
    retriever.retrieve_policy_subchunks_for_statute_subchunks = AsyncMock(
        return_value=RetrievePolicySubchunksResult(pairs=pairs, statute_subchunks_considered=6)
    )
    llm = MagicMock()
    llm.gap_check_subchunks = AsyncMock(
        side_effect=[
            {
                "addressed": True,
                "missing": False,
                "conflict": False,
                "policy_quote": "We disclose categories of personal information we collect.",
            },
            {
                "addressed": True,
                "missing": False,
                "conflict": False,
                "policy_quote": "Quote not present in policy text",
            },
            {
                "addressed": False,
                "missing": False,
                "conflict": True,
                "conflict_description": "Policy conflicts with opt-out timing.",
                "policy_quote": None,
            },
            {
                "addressed": False,
                "missing": True,
                "conflict": False,
                "conflict_description": None,
                "policy_quote": None,
            },
            {},  # analysis_failed
        ]
    )
    service = _service(retriever=retriever, llm_client=llm)
    service._load_policy_text = AsyncMock(return_value=("policy body", "Acme"))

    result = await service.gap_analysis(
        GapAnalysisRequest(
            policy_document_id="policy-1",
            applicable_jurisdictions=["CA"],
            save_results=True,
        )
    )

    assert result.run_types == ["gap"]
    assert result.company_name == "Acme"
    assert result.summary.total_requirements == 5
    assert result.summary.addressed == 1
    assert result.summary.missing == 3
    assert result.summary.conflicts == 1
    assert result.retrieval_metadata is not None
    assert result.retrieval_metadata.statute_subchunks_considered == 6
    assert result.statute_chunk_ids_used == [
        "ccpa-100:addr",
        "ccpa-100:bind",
        "ccpa-100:conf",
        "ccpa-100:miss",
        "ccpa-100:fail",
    ]

    by_id = {g.statute_chunk_id: g for g in result.gaps}
    assert by_id["addr"].status == "addressed"
    assert by_id["addr"].policy_quote is not None

    assert by_id["bind"].status == "missing"
    assert by_id["bind"].policy_quote is None
    assert "does not contain provisions" in (by_id["bind"].conflict_description or "")

    assert by_id["conf"].status == "conflict"
    assert by_id["conf"].conflict_description == "Policy conflicts with opt-out timing."

    assert by_id["miss"].status == "missing"
    assert "does not contain provisions" in (by_id["miss"].conflict_description or "")

    assert by_id["fail"].status == "missing"
    assert by_id["fail"].analysis_failed is True

    # Requirement summaries truncate at sentence boundaries.
    assert by_id["addr"].requirement_summary.endswith(".")
    assert llm.gap_check_subchunks.await_count == 5
    service._storage.write_compliance_result.assert_awaited_once()
    persisted = service._storage.write_compliance_result.await_args.args[0]
    assert persisted["run_types"] == ["gap"]
    service._storage.write_compliance_run_log.assert_awaited_once()


@pytest.mark.anyio
async def test_gap_analysis_num_rows_unindexed_and_empty_indexed() -> None:
    """num_rows slices pairs; unindexed raises; empty indexed returns empty gaps."""
    many_pairs = [_subchunk_pair(subchunk_id=f"s{i}") for i in range(3)]
    retriever = MagicMock()
    retriever.retrieve_policy_subchunks_for_statute_subchunks = AsyncMock(
        return_value=RetrievePolicySubchunksResult(pairs=many_pairs, statute_subchunks_considered=3)
    )
    llm = MagicMock()
    llm.gap_check_subchunks = AsyncMock(
        return_value={"addressed": True, "missing": False, "conflict": False, "policy_quote": None}
    )
    service = _service(retriever=retriever, llm_client=llm)
    service._load_policy_text = AsyncMock(return_value=("text", "Co"))

    limited = await service.gap_analysis(
        GapAnalysisRequest(
            policy_document_id="policy-1",
            applicable_jurisdictions=["CA"],
            num_rows=1,
            save_results=False,
        )
    )
    assert len(limited.gaps) == 1
    assert llm.gap_check_subchunks.await_count == 1
    service._storage.write_compliance_result.assert_not_awaited()

    # Unindexed policy: no pairs and count_documents == 0
    retriever.retrieve_policy_subchunks_for_statute_subchunks = AsyncMock(
        return_value=RetrievePolicySubchunksResult(pairs=[], statute_subchunks_considered=0)
    )
    mock_coll = MagicMock()
    mock_coll.count_documents.return_value = 0
    service._mongo_client = MagicMock()
    service._mongo_client.__getitem__.return_value.__getitem__.return_value = mock_coll
    with pytest.raises(ComplianceSuiteServiceError) as exc:
        await service.gap_analysis(
            GapAnalysisRequest(
                policy_document_id="policy-1",
                applicable_jurisdictions=["CA"],
                save_results=False,
            )
        )
    assert exc.value.status_code == 400
    assert "Policy not indexed" in str(exc.value)
    assert service._config.policy_sub_embeddings_collection in str(exc.value)

    # Indexed but no matches: empty gaps, no error
    mock_coll.count_documents.return_value = 4
    empty = await service.gap_analysis(
        GapAnalysisRequest(
            policy_document_id="policy-1",
            applicable_jurisdictions=["CA"],
            save_results=False,
        )
    )
    assert empty.gaps == []
    assert empty.summary.total_requirements == 0
    assert empty.retrieval_metadata is not None
    assert empty.retrieval_metadata.statute_pairs_matched == 0


@pytest.mark.anyio
async def test_gap_analysis_applicability_fallback_and_persistence_failure() -> None:
    """Fall back to applicability when jurisdictions unset; persistence errors do not fail."""
    retriever = MagicMock()
    retriever.retrieve_policy_subchunks_for_statute_subchunks = AsyncMock(
        return_value=RetrievePolicySubchunksResult(
            pairs=[_subchunk_pair()],
            statute_subchunks_considered=1,
        )
    )
    llm = MagicMock()
    llm.gap_check_subchunks = AsyncMock(
        return_value={"addressed": False, "missing": True, "conflict": False}
    )
    llm.applicability = AsyncMock(return_value={"applicable_jurisdictions": ["VA"], "confidence": {}})
    service = _service(retriever=retriever, llm_client=llm)
    service._config.default_jurisdictions = []
    service._load_policy_text = AsyncMock(return_value=("text", None))
    service._storage.write_compliance_result = AsyncMock(side_effect=RuntimeError("db down"))
    service._storage.write_compliance_run_log = AsyncMock(return_value=None)

    with patch("logging.Logger.warning") as warn:
        result = await service.gap_analysis(
            GapAnalysisRequest(policy_document_id="policy-1", save_results=True)
        )

    assert result.applicable_jurisdictions == ["VA"]
    assert result.gaps[0].status == "missing"
    retriever.retrieve_policy_subchunks_for_statute_subchunks.assert_awaited_once()
    call_kwargs = retriever.retrieve_policy_subchunks_for_statute_subchunks.await_args.kwargs
    assert call_kwargs["applicable_jurisdictions"] == ["VA"]
    warn.assert_called()
    assert "Failed to write compliance result" in warn.call_args.args[0]


# ----- gap_analysis_v2 (chunk-level) -----


@pytest.mark.anyio
async def test_gap_analysis_v2_status_mapping_and_run_types() -> None:
    """v2 uses chunk retriever/LLM, citation binding, and run_types=['gap_v2']."""
    pairs = [
        _chunk_pair(chunk_id="a", policy_text="Users may request deletion of personal data."),
        _chunk_pair(chunk_id="b", policy_text="Users may request deletion of personal data."),
        _chunk_pair(chunk_id="c"),
    ]
    retriever = MagicMock()
    retriever.retrieve_policy_chunks_for_statute_chunks = AsyncMock(
        return_value=RetrievePolicyChunksResult(pairs=pairs, statute_chunks_considered=3)
    )
    llm = MagicMock()
    llm.gap_check_chunks = AsyncMock(
        side_effect=[
            {
                "addressed": True,
                "missing": False,
                "conflict": False,
                "policy_quote": "Users may request deletion of personal data.",
            },
            {
                "addressed": True,
                "missing": False,
                "conflict": False,
                "policy_quote": "Unbound citation",
            },
            {
                "addressed": False,
                "missing": False,
                "conflict": True,
                "conflict_description": "Timing conflict",
            },
        ]
    )
    service = _service(retriever=retriever, llm_client=llm)
    service._load_policy_text = AsyncMock(return_value=("body", "Beta"))

    result = await service.gap_analysis_v2(
        GapAnalysisRequest(
            policy_document_id="policy-2",
            applicable_jurisdictions=["CA"],
            save_results=True,
        )
    )

    assert result.run_types == ["gap_v2"]
    assert result.company_name == "Beta"
    assert result.summary.total_requirements == 3
    assert result.summary.addressed == 1
    assert result.summary.missing == 1
    assert result.summary.conflicts == 1
    assert result.retrieval_metadata is not None
    assert result.retrieval_metadata.statute_chunks_considered == 3

    by_id = {g.statute_chunk_id: g for g in result.gaps}
    assert by_id["a"].status == "addressed"
    assert by_id["b"].status == "missing"
    assert by_id["b"].policy_quote is None
    assert by_id["c"].status == "conflict"
    assert by_id["a"].policy_combined_sections == "Users may request deletion of personal data."
    assert by_id["a"].requirement_summary.startswith("Consumers have the right to delete.")

    persisted = service._storage.write_compliance_result.await_args.args[0]
    assert persisted["run_types"] == ["gap_v2"]
    llm.gap_check_subchunks.assert_not_called()


@pytest.mark.anyio
async def test_gap_analysis_v2_unindexed_and_rate_limit() -> None:
    """v2 unindexed message references policy_embeddings_collection; rate limit still applies."""
    rate_limiter = MagicMock()
    rate_limiter.allow = AsyncMock(return_value=False)
    service = _service(rate_limiter=rate_limiter)
    with pytest.raises(ComplianceSuiteServiceError) as rate_exc:
        await service.gap_analysis_v2(
            GapAnalysisRequest(policy_document_id="policy-1", save_results=False)
        )
    assert rate_exc.value.status_code == 429

    retriever = MagicMock()
    retriever.retrieve_policy_chunks_for_statute_chunks = AsyncMock(
        return_value=RetrievePolicyChunksResult(pairs=[], statute_chunks_considered=0)
    )
    mock_coll = MagicMock()
    mock_coll.count_documents.return_value = 0
    mongo = MagicMock()
    mongo.__getitem__.return_value.__getitem__.return_value = mock_coll
    service = _service(retriever=retriever, mongo_client=mongo)
    service._load_policy_text = AsyncMock(return_value=("text", None))

    with pytest.raises(ComplianceSuiteServiceError) as exc:
        await service.gap_analysis_v2(
            GapAnalysisRequest(
                policy_document_id="policy-1",
                applicable_jurisdictions=["CA"],
                save_results=False,
            )
        )
    assert exc.value.status_code == 400
    assert "Policy not indexed for gap analysis (v2)" in str(exc.value)
    assert service._config.policy_embeddings_collection in str(exc.value)
