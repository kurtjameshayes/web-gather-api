"""_fetch_v4_statute_docs empty-jurisdiction broadcast contract."""
from __future__ import annotations

import asyncio
import sys
from unittest.mock import MagicMock

sys.modules.setdefault("anthropic", MagicMock())
sys.modules.setdefault("sentence_transformers", MagicMock())

from compliance_config import load_config
from compliance_suite_service import ComplianceSuiteService


def _service(docs: list[dict]) -> ComplianceSuiteService:
    mongo = MagicMock()
    coll = MagicMock()
    coll.find.return_value.limit.return_value = docs
    mongo.__getitem__.return_value.__getitem__.return_value = coll
    config = load_config()
    return ComplianceSuiteService(
        mongo_client=mongo,
        config=config,
        retriever=MagicMock(),
        llm_client=MagicMock(),
        storage=MagicMock(),
        rate_limiter=MagicMock(),
    )


def test_empty_jurisdiction_statute_is_broadcast_to_all_requested() -> None:
    """A blank jurisdiction field is attached to every requested jurisdiction bucket."""
    blank = {"_id": "blank", "jurisdiction": "  ", "subtopic_text": "applies everywhere"}
    ca_only = {"_id": "ca", "jurisdiction": "CA", "subtopic_text": "california only"}
    va_only = {"_id": "va", "jurisdiction": "VA", "subtopic_text": "virginia only"}
    svc = _service([blank, ca_only, va_only])

    result = asyncio.run(svc._fetch_v4_statute_docs(["CA", "VA"]))

    assert blank in result["CA"]
    assert blank in result["VA"]
    assert ca_only in result["CA"]
    assert ca_only not in result["VA"]
    assert va_only in result["VA"]
    assert va_only not in result["CA"]


def test_california_alias_matches_ca_request_not_broadcast() -> None:
    """Stored 'California' normalizes to CA and is not copied into other jurisdictions."""
    stored = {"_id": "cal", "jurisdiction": "California", "subtopic_text": "ccpa"}
    svc = _service([stored])

    result = asyncio.run(svc._fetch_v4_statute_docs(["CA", "VA"]))

    assert stored in result["CA"]
    assert stored not in result["VA"]


def test_default_category_filter_is_analyze_categories() -> None:
    """Default fetch is category-only (consumer_rights/controller_duties), not jurisdiction-filtered."""
    svc = _service([])
    asyncio.run(svc._fetch_v4_statute_docs(["CA"]))

    coll = svc._mongo_client.__getitem__.return_value.__getitem__.return_value
    coll.find.assert_called_once_with(
        {"category": {"$in": ["consumer_rights", "controller_duties"]}}
    )
    coll.find.return_value.limit.assert_called_once_with(200)
