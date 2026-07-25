"""Regression tests for health_score, consumer_rights_router, and shared helpers."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from compliance_config import load_config
from compliance_suite_schemas import (
    ConsumerRightsRouterRequest,
    GapAnalysisResponse,
    GapItem,
    GapSummary,
    HealthScoreRequest,
)
from compliance_suite_service import ComplianceSuiteService, ComplianceSuiteServiceError


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _service(**overrides: object) -> ComplianceSuiteService:
    config = load_config()
    config.default_jurisdictions = ["CA", "VA"]
    config.requirement_weights = {
        "right to know": 1.5,
        "right to delete": 2.0,
        "right to": 0.5,
    }
    config.partial_credit_percent = 0.5
    config.conflict_penalty_multiplier = 0.7
    rate_limiter = MagicMock()
    rate_limiter.allow = AsyncMock(return_value=True)
    storage = MagicMock()
    storage.write_compliance_result = AsyncMock(return_value=None)
    llm_client = MagicMock()
    llm_client.consumer_rights_router = AsyncMock(return_value=None)
    kwargs = {
        "mongo_client": MagicMock(),
        "config": config,
        "retriever": MagicMock(),
        "llm_client": llm_client,
        "storage": storage,
        "rate_limiter": rate_limiter,
        "gap_analysis_v4_service": MagicMock(),
    }
    kwargs.update(overrides)
    return ComplianceSuiteService(**kwargs)


def _gap(
    *,
    jurisdiction: str,
    requirement_summary: str,
    status: str,
    analysis_failed: bool = False,
) -> GapItem:
    return GapItem(
        jurisdiction=jurisdiction,
        statute_reference="§ 1798.100",
        requirement_summary=requirement_summary,
        status=status,
        analysis_failed=analysis_failed,
    )


def _gap_result(
    gaps: list[GapItem],
    *,
    company_name: str | None = "Acme",
    summary: GapSummary | None = None,
) -> GapAnalysisResponse:
    if summary is None:
        summary = GapSummary(
            total_requirements=len(gaps),
            addressed=sum(1 for g in gaps if g.status == "addressed"),
            missing=sum(1 for g in gaps if g.status == "missing"),
            conflicts=sum(1 for g in gaps if g.status == "conflict"),
            partial=sum(1 for g in gaps if g.status == "partial"),
            ambiguous=sum(1 for g in gaps if g.status == "ambiguous"),
            analysis_failures=sum(1 for g in gaps if g.analysis_failed),
        )
    return GapAnalysisResponse(
        policy_document_id="policy-1",
        company_name=company_name,
        applicable_jurisdictions=["CA", "VA"],
        analyzed_at="2026-07-25T10:00:00+00:00",
        gaps=gaps,
        summary=summary,
    )


# ----- health_score -----


@pytest.mark.anyio
async def test_health_score_longest_match_weight_and_partial_credit() -> None:
    """Longer weight keys win and partial status uses configured partial credit."""
    v4 = MagicMock()
    v4.run = AsyncMock(
        return_value=_gap_result(
            [
                _gap(
                    jurisdiction="CA",
                    requirement_summary="Consumers have the right to delete personal data.",
                    status="addressed",
                ),
                _gap(
                    jurisdiction="VA",
                    requirement_summary="Consumers have the right to know categories collected.",
                    status="partial",
                ),
                _gap(
                    jurisdiction="CA",
                    requirement_summary="Opt out of sale of personal information.",
                    status="missing",
                ),
            ]
        )
    )
    service = _service(gap_analysis_v4_service=v4)

    result = await service.health_score(
        HealthScoreRequest(policy_document_id="policy-1", save_results=False)
    )

    # weights: delete=2.0 addressed, know=1.5 * 0.5 partial, opt-out default=1.0 missing
    # weighted_sum = 2.0 + 0.75 + 0 = 2.75; total = 4.5; score = round(100 * 2.75/4.5) = 61
    assert result.privacy_health_score == 61
    assert result.error is None
    assert result.components["requirements_total"] == 3
    assert result.components["partial"] == 1
    assert result.components["conflict_penalty_applied"] is False
    assert result.score_breakdown["by_jurisdiction"]["CA"] == 67  # 2.0 / 3.0
    assert result.score_breakdown["by_jurisdiction"]["VA"] == 50  # 0.75 / 1.5
    service._storage.write_compliance_result.assert_not_awaited()


@pytest.mark.anyio
async def test_health_score_applies_conflict_penalty_and_persists() -> None:
    """Conflicts multiply the raw ratio and successful scores persist when enabled."""
    v4 = MagicMock()
    v4.run = AsyncMock(
        return_value=_gap_result(
            [
                _gap(
                    jurisdiction="CA",
                    requirement_summary="Right to know personal data categories.",
                    status="addressed",
                ),
                _gap(
                    jurisdiction="CA",
                    requirement_summary="Right to opt out of sale.",
                    status="conflict",
                ),
            ],
            summary=GapSummary(
                total_requirements=2,
                addressed=1,
                missing=0,
                conflicts=1,
                partial=0,
                ambiguous=0,
            ),
        )
    )
    service = _service(gap_analysis_v4_service=v4)
    service._config.requirement_weights = {"right to know": 1.5}

    result = await service.health_score(HealthScoreRequest(policy_document_id="policy-1"))

    # weights: know=1.5 addressed, opt out default=1.0 conflict (0 credit)
    # raw = 1.5/2.5 = 0.6; with 0.7 penalty => 0.42; score = 42
    assert result.privacy_health_score == 42
    assert result.components["conflict_penalty_applied"] is True
    assert result.components["conflicts"] == 1
    service._storage.write_compliance_result.assert_awaited_once()
    persisted = service._storage.write_compliance_result.await_args.args[0]
    assert persisted["privacy_health_score"] == 42
    assert persisted["run_types"] == ["health_score"]


@pytest.mark.anyio
async def test_health_score_no_gaps_returns_no_applicable_statutes() -> None:
    """Empty gap lists short-circuit to a null score with an explicit error."""
    v4 = MagicMock()
    v4.run = AsyncMock(return_value=_gap_result([], company_name=None))
    service = _service(gap_analysis_v4_service=v4)

    result = await service.health_score(
        HealthScoreRequest(policy_document_id="policy-1", save_results=False)
    )

    assert result.privacy_health_score is None
    assert result.error == "no_applicable_statutes"
    assert result.components["requirements_total"] == 0
    assert "Unable to assess" in (result.score_assessment or "")
    service._storage.write_compliance_result.assert_not_awaited()


@pytest.mark.anyio
async def test_health_score_skips_failed_gaps_for_insufficient_analysis() -> None:
    """Only analysis_failed gaps leave total_weight at zero and return insufficient_analysis."""
    v4 = MagicMock()
    v4.run = AsyncMock(
        return_value=_gap_result(
            [
                _gap(
                    jurisdiction="CA",
                    requirement_summary="Right to know",
                    status="missing",
                    analysis_failed=True,
                )
            ],
            summary=GapSummary(
                total_requirements=1,
                missing=1,
                analysis_failures=1,
            ),
        )
    )
    service = _service(gap_analysis_v4_service=v4)

    result = await service.health_score(
        HealthScoreRequest(policy_document_id="policy-1", save_results=False)
    )

    assert result.privacy_health_score is None
    assert result.error == "insufficient_analysis"
    assert result.components["requirements_total"] == 1


@pytest.mark.anyio
async def test_health_score_rate_limit_short_circuits_gap_analysis() -> None:
    """Rate limiting stops before gap analysis runs."""
    v4 = MagicMock()
    v4.run = AsyncMock()
    service = _service(gap_analysis_v4_service=v4)
    service._rate_limiter.allow = AsyncMock(return_value=False)

    with pytest.raises(ComplianceSuiteServiceError) as exc_info:
        await service.health_score(HealthScoreRequest(policy_document_id="policy-1"))

    assert exc_info.value.status_code == 429
    v4.run.assert_not_awaited()


# ----- consumer_rights_router -----


@pytest.mark.anyio
async def test_consumer_rights_router_mixed_llm_outcomes_and_jurisdiction_meta() -> None:
    """Mixed None/exception LLM outcomes stay safe and unknown jurisdictions get slug meta."""
    service = _service()
    service._fetch_v4_statute_docs = AsyncMock(
        return_value={
            "CA": [
                {
                    "header_text": "Cal. Civ. Code § 1798.105",
                    "subtopic_text": "Right to delete personal information.",
                }
            ],
            "ZZ": [{"requirement_summary": "Generic consumer right."}],
        }
    )
    service._llm_client.consumer_rights_router = AsyncMock(
        side_effect=[
            {
                "trees": {"california": {"steps": ["verify", "delete"]}},
                "policy_gaps": {
                    "california": {"covered": False, "gap": "No deletion form"},
                    "ignored": "not-a-dict",
                },
            },
            None,
            RuntimeError("llm down"),
        ]
    )

    result = await service.consumer_rights_router(
        ConsumerRightsRouterRequest(
            text="  Our privacy policy text.  ",
            applicable_jurisdictions=["CA", "ZZ"],
            request_types=["deletion", "access", "optout"],
            save_results=True,
        )
    )

    assert result.policy_document_id is None
    assert result.company_name is None
    assert result.applicable_jurisdictions == ["CA", "ZZ"]
    assert result.states["california"].abbr == "CA"
    assert result.states["zz"].name == "ZZ"
    assert result.states["zz"].abbr == "ZZ"
    assert result.trees["deletion"]["california"]["steps"] == ["verify", "delete"]
    assert result.policy_gaps["deletion"]["california"].covered is False
    assert result.policy_gaps["deletion"]["california"].gap == "No deletion form"
    assert "ignored" not in result.policy_gaps["deletion"]
    assert result.trees["access"] == {}
    assert result.policy_gaps["access"] == {}
    assert "optout" not in result.trees
    service._storage.write_compliance_result.assert_not_awaited()
    service._fetch_v4_statute_docs.assert_awaited_once_with(
        ["CA", "ZZ"], categories=["consumer_rights"]
    )


@pytest.mark.anyio
async def test_consumer_rights_router_loads_chunks_and_persists_when_requested() -> None:
    """Chunk-assembled policy text is used and results persist only with a document id."""
    service = _service()
    service._load_policy_from_chunks = AsyncMock(
        return_value=("Assembled policy body", "Acme Corp")
    )
    service._fetch_v4_statute_docs = AsyncMock(return_value={"CA": []})
    service._llm_client.consumer_rights_router = AsyncMock(
        return_value={"trees": {"california": {}}, "policy_gaps": {}}
    )

    result = await service.consumer_rights_router(
        ConsumerRightsRouterRequest(
            policy_document_id="policy-42",
            applicable_jurisdictions=["CA"],
            request_types=["access"],
            save_results=True,
        )
    )

    service._load_policy_from_chunks.assert_awaited_once_with(
        "policy-42",
        database="privacy-compliance",
        collection="policy_legal_embeddings",
    )
    assert result.policy_document_id == "policy-42"
    assert result.company_name == "Acme Corp"
    assert result.request_types["access"] == "Right to Access / Know"
    service._storage.write_compliance_result.assert_awaited_once()
    persisted = service._storage.write_compliance_result.await_args.args[0]
    assert persisted["result_type"] == "consumer_rights_router"
    assert persisted["policy_document_id"] == "policy-42"


@pytest.mark.anyio
async def test_consumer_rights_router_empty_policy_and_rate_limit() -> None:
    """Empty policy text and rate limits fail before LLM work."""
    service = _service()
    service._load_policy_from_chunks = AsyncMock(return_value=("", None))
    service._fetch_v4_statute_docs = AsyncMock()

    with pytest.raises(ComplianceSuiteServiceError) as empty_exc:
        await service.consumer_rights_router(
            ConsumerRightsRouterRequest(policy_document_id="missing-policy")
        )
    assert empty_exc.value.status_code == 400
    assert "not found or empty" in str(empty_exc.value)
    service._fetch_v4_statute_docs.assert_not_awaited()

    service._rate_limiter.allow = AsyncMock(return_value=False)
    with pytest.raises(ComplianceSuiteServiceError) as rate_exc:
        await service.consumer_rights_router(
            ConsumerRightsRouterRequest(text="policy text")
        )
    assert rate_exc.value.status_code == 429
    service._llm_client.consumer_rights_router.assert_not_awaited()


# ----- shared helpers -----


@pytest.mark.anyio
async def test_load_policy_text_falls_back_to_chunks_and_object_id() -> None:
    """Missing text falls back to policy_chunks; document lookup falls back to _id."""
    from bson import ObjectId

    oid = ObjectId()
    coll = MagicMock()
    coll.find_one.side_effect = [
        None,
        {
            "_id": oid,
            "policy_chunks": [
                {"chunk_text": "First chunk"},
                {"chunk_text": "Second chunk"},
                "skip-me",
            ],
            "company_name": "Acme",
        },
    ]
    db = MagicMock()
    db.__getitem__.return_value = coll
    mongo = MagicMock()
    mongo.__getitem__.return_value = db

    service = _service(mongo_client=mongo)
    text, company = await service._load_policy_text(str(oid))

    assert text == "First chunk\n\nSecond chunk"
    assert company == "Acme"
    assert coll.find_one.call_count == 2
    assert coll.find_one.call_args_list[0].args[0] == {
        service._config.policy_document_id_field: str(oid)
    }
    assert coll.find_one.call_args_list[1].args[0] == {"_id": oid}


@pytest.mark.anyio
async def test_fetch_v4_statute_docs_buckets_by_normalized_jurisdiction() -> None:
    """Docs without jurisdiction fan out; normalized aliases map into requested buckets."""
    coll = MagicMock()
    coll.find.return_value.limit.return_value = [
        {"category": "consumer_rights", "jurisdiction": "California", "requirement_summary": "CA right"},
        {"category": "consumer_rights", "jurisdiction": "", "requirement_summary": "shared"},
        {"category": "consumer_rights", "jurisdiction": "VA", "requirement_summary": "VA right"},
        {"category": "consumer_rights", "jurisdiction": "TX", "requirement_summary": "ignored"},
    ]
    db = MagicMock()
    db.__getitem__.return_value = coll
    mongo = MagicMock()
    mongo.__getitem__.return_value = db
    service = _service(mongo_client=mongo)

    result = await service._fetch_v4_statute_docs(["CA", "VA"], categories=["consumer_rights"])

    coll.find.assert_called_once_with({"category": {"$in": ["consumer_rights"]}})
    assert [d["requirement_summary"] for d in result["CA"]] == ["CA right", "shared"]
    assert [d["requirement_summary"] for d in result["VA"]] == ["shared", "VA right"]
