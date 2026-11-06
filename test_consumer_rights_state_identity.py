"""Consumer-rights state metadata is keyed by slug, not by the request token.

The router looks up `_JURISDICTION_META` with `token.upper()` only. It does not
use `normalize_jurisdiction`, so a spelled-out name such as "California" is an
unknown jurisdiction whose slug still collides with "CA".
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from compliance_config import load_config
from compliance_suite_schemas import ConsumerRightsRouterRequest
from compliance_suite_service import ComplianceSuiteService
from rate_limiter import RateLimiter


def _service() -> tuple[ComplianceSuiteService, AsyncMock]:
    config = load_config()
    config.llm_concurrency = 1
    llm = MagicMock()
    llm.consumer_rights_router = AsyncMock(
        return_value={"trees": {"california": {}}, "policy_gaps": {}}
    )
    service = ComplianceSuiteService(
        mongo_client=MagicMock(),
        config=config,
        retriever=MagicMock(),
        llm_client=llm,
        storage=MagicMock(),
        rate_limiter=RateLimiter(1000),
    )
    service._fetch_v4_statute_docs = AsyncMock(return_value={})
    return service, llm


def _route(jurisdictions: list[str]):
    service, llm = _service()
    request = ConsumerRightsRouterRequest(
        text="We honor consumer deletion requests.",
        applicable_jurisdictions=jurisdictions,
        request_types=["deletion"],
        save_results=False,
    )
    response = asyncio.run(service.consumer_rights_router(request))
    prompt = llm.consumer_rights_router.await_args.kwargs["jurisdictions"]
    service._storage.write_compliance_result.assert_not_called()
    service._fetch_v4_statute_docs.assert_awaited_once_with(
        jurisdictions, categories=["consumer_rights"]
    )
    return response, prompt


def test_spelled_out_california_overwrites_official_state_record() -> None:
    response, prompt = _route(["CA", "California", "TX"])

    assert response.applicable_jurisdictions == ["CA", "California", "TX"]
    assert set(response.states) == {"california", "texas"}
    assert response.states["california"].name == "California"
    assert response.states["california"].abbr == "CALIFORNIA"
    assert response.states["texas"].name == "Texas (TDPSA)"
    assert response.states["texas"].abbr == "TX"
    assert prompt == "\n".join(
        [
            "- CA: California (CCPA/CPRA) (slug: california)",
            "- CALIFORNIA: California (slug: california)",
            "- TX: Texas (TDPSA) (slug: texas)",
        ]
    )


def test_later_state_code_restores_official_california_record() -> None:
    response, prompt = _route(["California", "CA"])

    assert list(response.states) == ["california"]
    assert response.states["california"].name == "California (CCPA/CPRA)"
    assert response.states["california"].abbr == "CA"
    assert prompt == "\n".join(
        [
            "- CALIFORNIA: California (slug: california)",
            "- CA: California (CCPA/CPRA) (slug: california)",
        ]
    )


def test_lowercase_state_code_matches_official_record_and_prompt_repeats_it() -> None:
    response, prompt = _route(["ca", "CA"])

    assert list(response.states) == ["california"]
    assert response.states["california"].name == "California (CCPA/CPRA)"
    assert response.states["california"].abbr == "CA"
    assert prompt == "\n".join(
        [
            "- CA: California (CCPA/CPRA) (slug: california)",
            "- CA: California (CCPA/CPRA) (slug: california)",
        ]
    )
