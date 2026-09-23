"""Regression tests for jurisdiction alias expansion used by retrieval and gap analysis."""
from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock

# Stub optional SDKs before compliance modules import them.
sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("anthropic", MagicMock())

from cache import SimpleLRUCache
from compliance_config import load_config
from compliance_suite_schemas import GapAnalysisRequest
from compliance_utils import jurisdiction_filter_values, normalize_jurisdiction
from gap_analysis_service_v3 import GapAnalysisServiceV3
from vector_retriever import VectorRetriever


def test_known_codes_expand_to_stored_aliases() -> None:
    """Normalized codes must match both codes and the spellings stored in Mongo."""
    assert jurisdiction_filter_values("CA") == [
        "CA",
        "california",
        "California",
        "ca",
        "Ca",
    ]
    assert jurisdiction_filter_values("US") == [
        "US",
        "usa",
        "Usa",
        "u.s.",
        "U.s.",
        "us",
        "Us",
        "united states",
        "United states",
    ]
    # str.capitalize() does not title-case later words, so this stored form is missed.
    assert "United States" not in jurisdiction_filter_values("US")
    assert jurisdiction_filter_values("EU") == [
        "EU",
        "european union",
        "European union",
        "eu",
        "Eu",
    ]
    assert "European Union" not in jurisdiction_filter_values("EU")
    assert jurisdiction_filter_values("UK") == [
        "UK",
        "uk",
        "Uk",
        "united kingdom",
        "United kingdom",
    ]
    assert "United Kingdom" not in jurisdiction_filter_values("UK")


def test_unnormalized_inputs_do_not_invent_aliases() -> None:
    """Callers must normalize first. Display names and unknown codes stay exact."""
    assert jurisdiction_filter_values("") == []
    assert jurisdiction_filter_values(None) == []  # type: ignore[arg-type]
    assert jurisdiction_filter_values("TX") == ["TX"]
    assert jurisdiction_filter_values("   ") == ["   "]
    assert jurisdiction_filter_values("California") == ["California"]
    assert "CA" not in jurisdiction_filter_values("ca")
    assert jurisdiction_filter_values("ca") == [
        "ca",
        "california",
        "California",
        "Ca",
    ]

    assert jurisdiction_filter_values(normalize_jurisdiction("California")) == jurisdiction_filter_values("CA")
    assert jurisdiction_filter_values(normalize_jurisdiction("u.s.")) == jurisdiction_filter_values("US")
    assert jurisdiction_filter_values(normalize_jurisdiction("United Kingdom")) == jurisdiction_filter_values("UK")
    assert normalize_jurisdiction("  ca  ") == "CA"


class _DummyEmbedder:
    def __init__(self, vector=None):
        self.vector = [0.1, 0.2] if vector is None else vector
        self.calls = []

    async def embed(self, text):
        self.calls.append(text)
        return self.vector


def _statute_retriever(aggregate_result):
    config = load_config()
    config.use_embeddings_collection = False
    collection = MagicMock()
    collection.aggregate.return_value = aggregate_result
    client = MagicMock()
    client.__getitem__.return_value.__getitem__.return_value = collection
    embedder = _DummyEmbedder()
    retriever = VectorRetriever(client, embedder, config, SimpleLRUCache(10, 60))
    return retriever, collection, embedder, config


def test_statute_retriever_expands_ca_and_keeps_unknown_codes_exact() -> None:
    retriever, collection, embedder, config = _statute_retriever(
        [
            {
                "_id": "stat-1",
                "jurisdiction": "California",
                "title": "Right to know",
                "section_text": "A consumer may request disclosure.",
                "section": "",
                "section_id": "1798.100",
                "score": 0.81,
                "chunk_header_text": "Civil Code",
            }
        ]
    )

    results = asyncio.run(
        retriever.retrieve(
            database="privacy-compliance",
            section_text="consumers can request their data",
            jurisdiction="CA",
            statute_corpus_id="ccpa",
            top_k=3,
        )
    )
    cached = asyncio.run(
        retriever.retrieve(
            database="privacy-compliance",
            section_text="consumers can request their data",
            jurisdiction="CA",
            statute_corpus_id="ccpa",
            top_k=3,
        )
    )

    assert results[0].statute_id == "stat-1"
    assert results[0].section_id == "1798.100"
    assert results[0].score == 0.81
    assert cached == results
    assert collection.aggregate.call_count == 1
    assert embedder.calls == ["consumers can request their data"]

    pipeline = collection.aggregate.call_args[0][0]
    vector_stage = pipeline[0]["$vectorSearch"]
    assert vector_stage["numCandidates"] == 30
    assert vector_stage["limit"] == 3
    jur_filter = vector_stage["filter"][config.statute_jurisdiction_field]
    assert jur_filter == {"$in": jurisdiction_filter_values("CA")}
    assert "California" in jur_filter["$in"]
    assert vector_stage["filter"]["$or"] == [
        {config.statute_corpus_field: "ccpa"},
        {f"metadata.{config.statute_corpus_field}": "ccpa"},
    ]

    unknown, unknown_collection, _, _ = _statute_retriever([])
    asyncio.run(
        unknown.retrieve(
            database="privacy-compliance",
            section_text="other state",
            jurisdiction="TX",
            statute_corpus_id=None,
            top_k=1,
        )
    )
    tx_filter = unknown_collection.aggregate.call_args[0][0][0]["$vectorSearch"]["filter"]
    assert tx_filter == {config.statute_jurisdiction_field: "TX"}

    blank, blank_collection, _, _ = _statute_retriever([])
    asyncio.run(
        blank.retrieve(
            database="privacy-compliance",
            section_text="default jurisdiction",
            jurisdiction="   ",
            statute_corpus_id=None,
            top_k=1,
        )
    )
    blank_filter = blank_collection.aggregate.call_args[0][0][0]["$vectorSearch"]["filter"]
    assert blank_filter[config.statute_jurisdiction_field] == {
        "$in": jurisdiction_filter_values(normalize_jurisdiction("California"))
    }


def test_empty_embedding_skips_search_and_is_not_cached() -> None:
    retriever, collection, embedder, _ = _statute_retriever([])
    embedder.vector = []

    first = asyncio.run(
        retriever.retrieve(
            database="privacy-compliance",
            section_text="blank",
            jurisdiction="CA",
            statute_corpus_id=None,
            top_k=2,
        )
    )
    embedder.vector = [0.4, 0.5]
    second = asyncio.run(
        retriever.retrieve(
            database="privacy-compliance",
            section_text="blank",
            jurisdiction="CA",
            statute_corpus_id=None,
            top_k=2,
        )
    )

    assert first == []
    assert second == []
    assert embedder.calls == ["blank", "blank"]
    assert collection.aggregate.call_count == 1


def test_embeddings_retriever_expands_jurisdiction_and_swallows_search_errors() -> None:
    config = load_config()
    config.use_embeddings_collection = True
    config.statute_collection_tag = ""
    collection = MagicMock()
    collection.aggregate.return_value = [
        {
            "document_id": "doc-9",
            "chunk_text": "Consumers may opt out.",
            "chunk_index": "4",
            "jurisdiction": "California",
            "score": 0.66,
        }
    ]
    client = MagicMock()
    client.__getitem__.return_value.__getitem__.return_value = collection
    retriever = VectorRetriever(client, _DummyEmbedder(), config, SimpleLRUCache(4, 60))

    results = asyncio.run(
        retriever.retrieve(
            database="privacy-compliance",
            section_text="opt out of sale",
            jurisdiction="California",
            statute_corpus_id=None,
            top_k=2,
        )
    )

    assert results[0].statute_id == "doc-9"
    assert results[0].jurisdiction == "California"
    assert results[0].chunk_text == "Consumers may opt out."
    pipeline = collection.aggregate.call_args[0][0]
    search_filter = pipeline[0]["$vectorSearch"]["filter"]
    assert config.embedding_collection_tag_field not in search_filter
    assert search_filter[config.statute_jurisdiction_field] == {
        "$in": jurisdiction_filter_values("CA")
    }
    assert "California" in search_filter[config.statute_jurisdiction_field]["$in"]

    failing = MagicMock()
    failing.aggregate.side_effect = RuntimeError("vector index missing")
    fail_client = MagicMock()
    fail_client.__getitem__.return_value.__getitem__.return_value = failing
    fail_retriever = VectorRetriever(fail_client, _DummyEmbedder(), config, SimpleLRUCache(4, 60))
    failed = asyncio.run(
        fail_retriever.retrieve(
            database="privacy-compliance",
            section_text="opt out of sale",
            jurisdiction="CA",
            statute_corpus_id=None,
            top_k=2,
        )
    )
    assert failed == []


def test_v3_gap_analysis_queries_expanded_jurisdiction_aliases() -> None:
    """V3 statute lookup must search stored spellings, not only the request code."""
    config = load_config()
    collections: dict[str, MagicMock] = {}

    def get_coll(name: str) -> MagicMock:
        if name not in collections:
            collections[name] = MagicMock()
        return collections[name]

    db = MagicMock()
    db.__getitem__.side_effect = get_coll
    mongo = MagicMock()
    mongo.__getitem__.return_value = db

    policies = get_coll(config.policies_collection)
    policies.find_one.return_value = {
        "document_id": "policy-1",
        "text": "We describe consumer rights.",
        "company_name": "Acme",
    }
    get_coll(config.policy_sub_embeddings_collection).count_documents.return_value = 1
    statute_coll = get_coll(config.statute_sub_embeddings_collection)
    statute_coll.find.return_value = []

    limiter = MagicMock()
    limiter.allow = AsyncMock(return_value=True)
    service = GapAnalysisServiceV3(
        mongo_client=mongo,
        config=config,
        retriever=MagicMock(),
        llm_client=MagicMock(),
        storage=MagicMock(),
        rate_limiter=limiter,
    )

    asyncio.run(
        service.run(
            GapAnalysisRequest(
                policy_document_id="policy-1",
                applicable_jurisdictions=["CA", "United States"],
                save_results=False,
            )
        )
    )

    statute_filter = statute_coll.find.call_args[0][0]
    queried = statute_filter[config.statute_jurisdiction_field]["$in"]
    assert "CA" in queried
    assert "California" in queried
    assert "united states" in queried
    assert "United states" in queried
    assert "United States" not in queried
    service.llm.gap_check_v3.assert_not_called()

    statute_coll.find.reset_mock()
    asyncio.run(
        service.run(
            GapAnalysisRequest(
                policy_document_id="policy-1",
                applicable_jurisdictions=[],
                save_results=False,
            )
        )
    )
    default_values = statute_coll.find.call_args[0][0][config.statute_jurisdiction_field]["$in"]
    assert "California" in default_values
    for code in ("VA", "CO", "CT"):
        assert code in default_values
