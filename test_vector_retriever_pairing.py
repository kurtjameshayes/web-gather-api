"""Regression tests for statute/policy vector pairing used by gap analysis v1 and v2.

These methods decide which policy text is compared to each statute chunk. A bad
vector must not abort the batch, an empty filtered search must not widen to
other policies, and a filter-index failure must fall back to a policy-scoped match.
"""
from __future__ import annotations

import asyncio
import sys
from unittest.mock import MagicMock

sys.modules.setdefault("sentence_transformers", MagicMock())

from compliance_config import load_config
from compliance_utils import jurisdiction_filter_values
from vector_retriever import VECTOR_SEARCH_JURISDICTION, VectorRetriever

DATABASE = "privacy-compliance"


def _vec(config, vector, **doc):
    doc[config.embedding_vector_field] = vector
    return doc


def _retriever():
    collections: dict[tuple[str, str], MagicMock] = {}
    databases: dict[str, MagicMock] = {}

    def get_coll(db_name: str, coll_name: str) -> MagicMock:
        key = (db_name, coll_name)
        if key not in collections:
            collections[key] = MagicMock()
        return collections[key]

    def get_db(name: str) -> MagicMock:
        if name not in databases:
            db = MagicMock()
            db.__getitem__.side_effect = lambda coll, db_name=name: get_coll(db_name, coll)
            databases[name] = db
        return databases[name]

    mongo = MagicMock()
    mongo.__getitem__.side_effect = get_db
    config = load_config()
    retriever = VectorRetriever(mongo, MagicMock(), config, MagicMock())

    def collection(name: str) -> MagicMock:
        return get_coll(DATABASE, name)

    return retriever, collection, config


def _assert_candidate_budget(vector_search: dict) -> None:
    assert vector_search["limit"] <= vector_search["numCandidates"]
    assert vector_search["limit"] >= 500


def test_subchunk_pairing_skips_bad_vectors_and_sanitizes_hits() -> None:
    retriever, collection, config = _retriever()
    statute_coll = collection(config.statute_sub_embeddings_collection)
    policy_coll = collection(config.policy_sub_embeddings_collection)
    statute_coll.find.return_value = [
        _vec(config, [1, True], _id="good", sub_chunk_text="duty"),
        _vec(config, "nope", _id="bad-type"),
        _vec(config, [1, None], _id="bad-number"),
        _vec(config, [0.2, 0.4], _id="no-hit"),
    ]
    seen_vectors: list[list[float]] = []

    def aggregate(pipe):
        vector_search = pipe[0]["$vectorSearch"]
        _assert_candidate_budget(vector_search)
        assert vector_search["index"] == config.vector_index_name
        assert vector_search["path"] == config.embedding_vector_field
        assert vector_search["filter"] == {config.policy_document_id_field: "policy-1"}
        assert pipe[-1] == {"$limit": 1}
        seen_vectors.append(vector_search["queryVector"])
        if vector_search["queryVector"] == [1.0, 1.0]:
            return [
                {
                    "_id": 42,
                    "score": 0.88,
                    config.embedding_vector_field: [9.0, 9.0],
                    "sub_chunk_text": "we collect",
                    "document_id": "policy-1",
                }
            ]
        return []

    policy_coll.aggregate.side_effect = aggregate

    result = asyncio.run(
        retriever.retrieve_policy_subchunks_for_statute_subchunks(
            database=DATABASE,
            policy_document_id="policy-1",
            applicable_jurisdictions=["CA", "EU"],
            statute_document_id="statute-9",
            top_k_per_statute=1,
        )
    )

    statute_filter = statute_coll.find.call_args[0][0]
    jurisdictions = statute_filter[config.statute_jurisdiction_field]["$in"]
    assert "CA" in jurisdictions
    assert "California" in jurisdictions
    assert "EU" in jurisdictions
    assert "european union" in jurisdictions
    assert "European union" in jurisdictions
    assert statute_filter["document_id"] == "statute-9"
    assert seen_vectors == [[1.0, 1.0], [0.2, 0.4]]
    assert result.statute_subchunks_considered == 4
    assert len(result.pairs) == 1
    pair = result.pairs[0]
    assert pair.score == 0.88
    assert pair.statute_doc["_id"] == "good"
    assert pair.policy_doc["_id"] == "42"
    assert config.embedding_vector_field not in pair.policy_doc
    assert pair.policy_doc["sub_chunk_text"] == "we collect"


def test_subchunk_empty_corpus_does_not_search() -> None:
    retriever, collection, config = _retriever()
    statute_coll = collection(config.statute_sub_embeddings_collection)
    policy_coll = collection(config.policy_sub_embeddings_collection)
    statute_coll.find.return_value = []

    result = asyncio.run(
        retriever.retrieve_policy_subchunks_for_statute_subchunks(
            database=DATABASE,
            policy_document_id="policy-1",
            applicable_jurisdictions=["ZZ"],
        )
    )

    assert result.pairs == []
    assert result.statute_subchunks_considered == 0
    policy_coll.aggregate.assert_not_called()


def test_subchunk_empty_filtered_search_does_not_widen() -> None:
    retriever, collection, config = _retriever()
    statute_coll = collection(config.statute_sub_embeddings_collection)
    policy_coll = collection(config.policy_sub_embeddings_collection)
    statute_coll.find.return_value = [_vec(config, [0.5, 0.5], _id="s1")]
    policy_coll.aggregate.return_value = []

    result = asyncio.run(
        retriever.retrieve_policy_subchunks_for_statute_subchunks(
            database=DATABASE,
            policy_document_id="policy-1",
            applicable_jurisdictions=["ZZ"],
            top_k_per_statute=1,
        )
    )

    assert policy_coll.aggregate.call_count == 1
    pipeline = policy_coll.aggregate.call_args[0][0]
    assert "filter" in pipeline[0]["$vectorSearch"]
    assert not any("$match" in stage for stage in pipeline)
    assert result.pairs == []
    assert result.statute_subchunks_considered == 1


def test_subchunk_filter_error_falls_back_to_policy_match() -> None:
    retriever, collection, config = _retriever()
    statute_coll = collection(config.statute_sub_embeddings_collection)
    policy_coll = collection(config.policy_sub_embeddings_collection)
    statute_coll.find.return_value = [_vec(config, [3, 4], _id="s1")]
    calls: list[list] = []

    def aggregate(pipe):
        calls.append(pipe)
        if len(calls) == 1:
            raise RuntimeError("index has no filter")
        return [
            {
                    "_id": 7,
                    "score": 0.4,
                    config.embedding_vector_field: [1, 1],
                    "sub_chunk_text": "fallback hit",
            }
        ]

    policy_coll.aggregate.side_effect = aggregate

    result = asyncio.run(
        retriever.retrieve_policy_subchunks_for_statute_subchunks(
            database=DATABASE,
            policy_document_id="policy-1",
            applicable_jurisdictions=["ZZ"],
            top_k_per_statute=1,
        )
    )

    assert "filter" in calls[0][0]["$vectorSearch"]
    assert "filter" not in calls[1][0]["$vectorSearch"]
    assert {"$match": {config.policy_document_id_field: "policy-1"}} in calls[1]
    _assert_candidate_budget(calls[1][0]["$vectorSearch"])
    assert len(result.pairs) == 1
    assert result.pairs[0].policy_doc["_id"] == "7"
    assert result.pairs[0].policy_doc["sub_chunk_text"] == "fallback hit"
    assert config.embedding_vector_field not in result.pairs[0].policy_doc


def test_subchunk_jurisdiction_filter_default_and_single_value() -> None:
    retriever, collection, config = _retriever()
    statute_coll = collection(config.statute_sub_embeddings_collection)
    statute_coll.find.return_value = []
    field = config.statute_jurisdiction_field

    asyncio.run(
        retriever.retrieve_policy_subchunks_for_statute_subchunks(
            database=DATABASE,
            policy_document_id="policy-1",
            applicable_jurisdictions=[],
        )
    )
    asyncio.run(
        retriever.retrieve_policy_subchunks_for_statute_subchunks(
            database=DATABASE,
            policy_document_id="policy-1",
            applicable_jurisdictions=["zz"],
        )
    )

    blank_filter = statute_coll.find.call_args_list[0][0][0]
    single_filter = statute_coll.find.call_args_list[1][0][0]
    raw_values = jurisdiction_filter_values(VECTOR_SEARCH_JURISDICTION)
    assert "document_id" not in blank_filter
    if len(raw_values) > 1:
        assert blank_filter[field] == {"$in": raw_values}
    else:
        assert blank_filter[field] == raw_values[0]
    # Omitted jurisdictions are not normalized before filtering, so "California"
    # stays an equality match and does not expand to the CA alias set.
    if VECTOR_SEARCH_JURISDICTION == "California":
        assert blank_filter[field] == "California"
    assert single_filter == {field: "ZZ"}


def test_chunk_pairing_skips_bad_vectors_and_falls_back() -> None:
    retriever, collection, config = _retriever()
    statute_coll = collection(config.statute_embeddings_collection)
    policy_coll = collection(config.policy_embeddings_collection)
    statute_coll.find.return_value = [
        _vec(config, ["nope"], _id="bad"),
        _vec(config, [8, 9], _id="good"),
    ]
    calls: list[list] = []

    def aggregate(pipe):
        calls.append(pipe)
        if "filter" in pipe[0]["$vectorSearch"]:
            raise RuntimeError("index has no filter")
        return [
            {
                "_id": 3,
                "score": 0.6,
                config.embedding_vector_field: [1],
                "chunk_text": "chunk hit",
            }
        ]

    policy_coll.aggregate.side_effect = aggregate

    result = asyncio.run(
        retriever.retrieve_policy_chunks_for_statute_chunks(
            database=DATABASE,
            policy_document_id="policy-1",
            applicable_jurisdictions=["ZZ"],
            top_k_per_statute=1,
        )
    )

    assert len(calls) == 2
    assert {"$match": {config.policy_document_id_field: "policy-1"}} in calls[1]
    assert result.statute_chunks_considered == 2
    assert len(result.pairs) == 1
    assert result.pairs[0].statute_doc["_id"] == "good"
    assert result.pairs[0].policy_doc["_id"] == "3"
    assert result.pairs[0].policy_doc["chunk_text"] == "chunk hit"
    assert config.embedding_vector_field not in result.pairs[0].policy_doc
