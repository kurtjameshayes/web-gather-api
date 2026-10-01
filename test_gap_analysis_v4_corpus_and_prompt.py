"""Regression tests for v4 corpus routing and statute-prompt assembly.

These paths decide which database is analyzed, which text the LLM sees, and
whether a critic failure can hide an otherwise successful gap run.
"""
from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock

sys.modules.setdefault("anthropic", MagicMock())

from compliance_config import load_config
from compliance_suite_schemas import GapAnalysisRequest
from gap_analysis_service_v4 import GapAnalysisServiceV4


def _config(**overrides):
    config = load_config()
    config.adaptive_feedback_enabled = False
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _llm_result():
    return {
        "status": "missing",
        "policy_quote": None,
        "statute_quote": "",
        "requirement_summary": "Requirement",
        "conflict_description": None,
        "confidence": "low",
        "_analysis_failed": False,
    }


def _mongo_router():
    """Route mongo[db][collection] to stable mocks and record database names."""
    buckets: dict[str, dict[str, MagicMock]] = {}
    seen: list[str] = []

    def get_db(name: str):
        seen.append(name)
        buckets.setdefault(name, {})
        bucket = buckets[name]
        db = MagicMock()

        def get_coll(coll_name: str):
            if coll_name not in bucket:
                coll = MagicMock()
                coll.find.return_value = []
                coll.find_one.return_value = None
                coll.count_documents.return_value = 0
                bucket[coll_name] = coll
            return bucket[coll_name]

        db.__getitem__.side_effect = get_coll
        return db

    mongo = MagicMock()
    mongo.__getitem__.side_effect = get_db
    return mongo, buckets, seen


def _coll(mongo, buckets, db_name: str, coll_name: str) -> MagicMock:
    """Create the collection mock before the service runs and return that same object."""
    _ = mongo[db_name][coll_name]
    return buckets[db_name][coll_name]


def _service(mongo, config, llm=None, critic=None):
    if llm is None:
        llm = MagicMock()
        llm.gap_check_v4 = AsyncMock(return_value=_llm_result())
    limiter = MagicMock()
    limiter.allow = AsyncMock(return_value=True)
    return GapAnalysisServiceV4(
        mongo_client=mongo,
        config=config,
        llm_client=llm,
        rate_limiter=limiter,
        critic=critic,
    ), llm


def _run(svc, **kwargs):
    kwargs.setdefault("applicable_jurisdictions", ["CA"])
    kwargs.setdefault("save_results", False)
    if "policy_document_id" not in kwargs and "policy_document_ids" not in kwargs:
        kwargs["policy_document_id"] = "p1"
    return asyncio.run(svc.run(GapAnalysisRequest(**kwargs)))


def _mapping(sub_topic="deletion", category="consumer_rights"):
    return {
        "statute_category": category,
        "policy_categories": ["right_to_delete"],
        "sub_topic": sub_topic,
    }


def test_v4_request_database_does_not_redirect_statute_corpus() -> None:
    """Policies and saved results follow the request database.

    Statute mappings, statute text, and policy-legal embeddings stay on
    statute_database. One indexed policy id is enough for a multi-id run.
    """
    config = _config(statute_database="statute-corpus")
    mongo, buckets, seen = _mongo_router()
    svc, llm = _service(mongo, config)

    policies = _coll(mongo, buckets, "tenant-db", config.policies_collection)
    legal = _coll(mongo, buckets, "statute-corpus", config.policy_legal_embeddings_collection)
    mappings = _coll(mongo, buckets, "statute-corpus", config.category_mapping_collection)
    statutes = _coll(mongo, buckets, "statute-corpus", config.statute_sub_topic_embeddings_collection)
    policies.find_one.side_effect = [
        {"document_id": "p1", "text": "Policy one.", "company_name": "Acme"},
        {"document_id": "p2", "text": "Policy two."},
    ]
    legal.count_documents.return_value = 1
    mappings.find.return_value = [_mapping()]
    statutes.find.return_value = [
        {"_id": "s1", "header_text": "§ 1798.105", "subtopic_text": "Delete data.", "jurisdiction": "CA"}
    ]
    legal.find.return_value = [{"chunk_text": "We delete personal information."}]
    seen.clear()

    result = _run(
        svc,
        policy_document_ids=["p1", "p2"],
        database="tenant-db",
        save_results=True,
    )

    assert set(seen) == {"tenant-db", "statute-corpus"}
    assert "privacy-compliance" not in seen
    policies.find_one.assert_any_call({"document_id": "p1"})
    policies.find_one.assert_any_call({"document_id": "p2"})
    legal.count_documents.assert_called_once_with({"document_id": {"$in": ["p1", "p2"]}})
    assert mappings.find.called
    assert statutes.find.called
    assert llm.gap_check_v4.await_count == 1

    results = _coll(mongo, buckets, "tenant-db", config.compliance_results_collection)
    run_log = _coll(mongo, buckets, "tenant-db", config.compliance_run_log_collection)
    assert results.insert_one.called
    assert run_log.insert_one.called
    saved = results.insert_one.call_args[0][0]
    assert saved["policy_document_ids"] == ["p1", "p2"]
    assert saved["policy_document_id"] == "p1"
    assert config.compliance_results_collection not in buckets["statute-corpus"]
    assert result.policy_document_ids == ["p1", "p2"]
    assert result.summary.total_requirements == 1


def test_v4_whitespace_database_names_fall_back_to_compliance_database() -> None:
    config = _config(statute_database="  \n")
    mongo, buckets, seen = _mongo_router()
    svc, llm = _service(mongo, config)
    db_name = config.compliance_database
    policies = _coll(mongo, buckets, db_name, config.policies_collection)
    legal = _coll(mongo, buckets, db_name, config.policy_legal_embeddings_collection)
    mappings = _coll(mongo, buckets, db_name, config.category_mapping_collection)
    statutes = _coll(mongo, buckets, db_name, config.statute_sub_topic_embeddings_collection)
    policies.find_one.return_value = {"document_id": "p1", "text": "Policy text."}
    legal.count_documents.return_value = 1
    mappings.find.return_value = [_mapping()]
    statutes.find.return_value = [
        {"_id": "s1", "subtopic_text": "Delete data.", "jurisdiction": "CA"}
    ]
    seen.clear()

    result = _run(svc, database="   ")

    assert set(seen) == {config.compliance_database}
    assert llm.gap_check_v4.await_count == 1
    assert result.summary.total_requirements == 1


def test_v4_prompt_assembles_requirement_context_and_reference_fallbacks() -> None:
    """Header plus subtopic is the requirement. Context headers are labeled.

    A missing document id skips the context query and stores statute_reference
    as the string "None". A blank sub_topic does not filter; a numeric one is
    stringified. An empty company_name string blocks a later policy's name.
    """
    config = _config()
    mongo, buckets, _seen = _mongo_router()
    svc, llm = _service(mongo, config)
    db_name = config.compliance_database
    statute_coll = _coll(mongo, buckets, db_name, config.statute_sub_topic_embeddings_collection)
    policies = _coll(mongo, buckets, db_name, config.policies_collection)
    legal = _coll(mongo, buckets, db_name, config.policy_legal_embeddings_collection)
    mappings = _coll(mongo, buckets, db_name, config.category_mapping_collection)
    context_docs = [
        {
            "category": "definitions",
            "header_text": "Personal information",
            "subtopic_text": "means information that identifies a consumer.",
        },
        {
            "category": "applicability",
            "header_text": "",
            "requirement_summary": "applies to businesses",
        },
        {"category": "definitions", "header_text": "   ", "subtopic_text": "   "},
        {"category": "definitions", "header_text": "Header only"},
    ]
    statute_docs = [
        {
            "_id": "s1",
            "document_id": "ccpa-1798",
            "header_text": "§ 1798.105",
            "subtopic_text": "A business shall delete.",
            "requirement_summary": "ignored when subtopic is present",
            "jurisdiction": "CA",
        },
        {
            "_id": "s2",
            "requirement_summary": "Keep data",
            "jurisdiction": "   ",
        },
    ]

    def statute_find(query, projection=None):
        if query.get("document_id"):
            assert projection == {
                "subtopic_text": 1,
                "requirement_summary": 1,
                "header_text": 1,
                "category": 1,
            }
            assert query["category"] == {"$in": ["definitions", "applicability"]}
            return context_docs
        return statute_docs

    statute_coll.find.side_effect = statute_find
    policies.find_one.side_effect = [
        {"document_id": "p1", "text": "First policy.", "company_name": ""},
        {"document_id": "p2", "text": "Second policy.", "company_name": "Beta"},
    ]
    legal.count_documents.return_value = 1
    mappings.find.return_value = [_mapping(sub_topic="  deletion  ")]
    legal.find.return_value = [
        {"chunk_text": "  "},
        {"chunk_text": "We delete personal information."},
        {"chunk_text": ""},
    ]

    result = _run(svc, policy_document_ids=["p1", "p2"], applicable_jurisdictions=["VA"])

    requirement_query = statute_coll.find.call_args_list[0].args[0]
    assert requirement_query == {"category": "consumer_rights", "sub_topic": "deletion"}
    context_queries = [
        call.args[0] for call in statute_coll.find.call_args_list if call.args[0].get("document_id")
    ]
    assert context_queries == [
        {
            "document_id": "ccpa-1798",
            "category": {"$in": ["definitions", "applicability"]},
        }
    ]

    first, second = [call.kwargs for call in llm.gap_check_v4.await_args_list]
    assert first["statutory_requirement"] == "§ 1798.105\n\nA business shall delete."
    assert "ignored when subtopic is present" not in first["statutory_requirement"]
    assert first["reference_context"] == (
        "[definitions]\nPersonal information\n\nmeans information that identifies a consumer."
        "\n\n---\n\n"
        "applies to businesses"
        "\n\n---\n\n"
        "[definitions]\nHeader only"
    )
    assert first["policy_text"] == "We delete personal information."
    assert first["adaptive_feedback"] == ""

    assert second["statutory_requirement"] == "Keep data"
    assert second["reference_context"] == "(No definitions or applicability context available.)"

    assert result.company_name == ""
    assert [gap.statute_reference for gap in result.gaps] == ["§ 1798.105", "None"]
    assert result.gaps[0].jurisdiction == "CA"
    assert result.gaps[1].jurisdiction == "   "
    assert result.gaps[0].statute_subchunk_text == "A business shall delete."
    assert result.gaps[1].statute_subchunk_text is None
    assert result.gaps[1].statute_chunk_text == "Keep data"


def test_v4_blank_sub_topic_is_unfiltered_and_numbers_are_stringified() -> None:
    config = _config()

    def _capture(sub_topic):
        mongo, buckets, _seen = _mongo_router()
        svc, _llm = _service(mongo, config)
        db_name = config.compliance_database
        policies = _coll(mongo, buckets, db_name, config.policies_collection)
        legal = _coll(mongo, buckets, db_name, config.policy_legal_embeddings_collection)
        mappings = _coll(mongo, buckets, db_name, config.category_mapping_collection)
        statute_coll = _coll(mongo, buckets, db_name, config.statute_sub_topic_embeddings_collection)
        policies.find_one.return_value = {"document_id": "p1", "text": "Policy."}
        legal.count_documents.return_value = 1
        mappings.find.return_value = [_mapping(sub_topic=sub_topic)]
        statute_coll.find.return_value = []
        _run(svc)
        return statute_coll.find.call_args.args[0]

    assert _capture("  \t") == {"category": "consumer_rights"}
    assert _capture(12) == {"category": "consumer_rights", "sub_topic": "12"}


def test_v4_num_rows_budget_spans_mappings_and_duplicates_still_count() -> None:
    config = _config()
    mongo, buckets, _seen = _mongo_router()
    svc, llm = _service(mongo, config)
    db_name = config.compliance_database
    policies = _coll(mongo, buckets, db_name, config.policies_collection)
    legal = _coll(mongo, buckets, db_name, config.policy_legal_embeddings_collection)
    mappings = _coll(mongo, buckets, db_name, config.category_mapping_collection)
    statutes = _coll(mongo, buckets, db_name, config.statute_sub_topic_embeddings_collection)
    policies.find_one.return_value = {"document_id": "p1", "text": "Policy."}
    legal.count_documents.return_value = 1
    mappings.find.return_value = [
        _mapping(sub_topic="deletion"),
        _mapping(sub_topic="access"),
    ]
    batches = [
        [
            {"_id": "d1", "subtopic_text": "one", "jurisdiction": "CA"},
            {"_id": "d2", "subtopic_text": "two", "jurisdiction": "CA"},
        ],
        [
            {"_id": "d3", "subtopic_text": "three", "jurisdiction": "CA"},
            {"_id": "d4", "subtopic_text": "four", "jurisdiction": "CA"},
        ],
    ]

    def statute_find(query, projection=None):
        assert "document_id" not in query
        return batches.pop(0)

    statutes.find.side_effect = statute_find

    result = _run(svc, num_rows=3)

    assert llm.gap_check_v4.await_count == 3
    assert result.retrieval_metadata.statute_items_considered == 3
    assert result.statute_chunk_ids_used == ["d1", "d2", "d3"]
    assert batches == []

    mongo2, buckets2, _seen2 = _mongo_router()
    svc2, llm2 = _service(mongo2, config)
    db_name = config.compliance_database
    policies2 = _coll(mongo2, buckets2, db_name, config.policies_collection)
    legal2 = _coll(mongo2, buckets2, db_name, config.policy_legal_embeddings_collection)
    mappings2 = _coll(mongo2, buckets2, db_name, config.category_mapping_collection)
    statutes2 = _coll(mongo2, buckets2, db_name, config.statute_sub_topic_embeddings_collection)
    policies2.find_one.return_value = {"document_id": "p1", "text": "Policy."}
    legal2.count_documents.return_value = 1
    mappings2.find.return_value = [_mapping()]
    statutes2.find.return_value = [
        {"_id": "dup", "subtopic_text": "first text", "jurisdiction": "CA"},
        {"_id": "dup", "subtopic_text": "second text", "jurisdiction": "CA"},
    ]

    dup = _run(svc2)

    assert llm2.gap_check_v4.await_count == 1
    assert llm2.gap_check_v4.await_args.kwargs["statutory_requirement"] == "first text"
    assert dup.retrieval_metadata.statute_items_considered == 2
    assert dup.summary.total_requirements == 1
    assert dup.statute_chunk_ids_used == ["dup"]


def test_v4_feedback_flag_is_snapshotted_and_critic_errors_are_swallowed() -> None:
    config = _config(adaptive_feedback_enabled=True)
    mongo, buckets, _seen = _mongo_router()
    critic = MagicMock()
    critic.get_active_feedback.return_value = []
    critic.evaluate = AsyncMock()
    critic.record_feedback_usage = AsyncMock()
    svc, _llm = _service(mongo, config, critic=critic)
    db_name = config.compliance_database
    policies = _coll(mongo, buckets, db_name, config.policies_collection)
    legal = _coll(mongo, buckets, db_name, config.policy_legal_embeddings_collection)
    mappings = _coll(mongo, buckets, db_name, config.category_mapping_collection)
    policies.find_one.return_value = {"document_id": "p1", "text": "Policy."}
    legal.count_documents.return_value = 1
    mappings.find.return_value = [_mapping()]

    config.adaptive_feedback_enabled = False
    result = _run(svc, save_results=True)

    critic.get_active_feedback.assert_not_called()
    critic.record_feedback_usage.assert_not_called()
    critic.evaluate.assert_awaited_once()
    assert critic.evaluate.await_args.args[0].policy_document_id == "p1"
    assert critic.evaluate.await_args.args[1]
    assert result.run_type == "gap_analysis_v4"
    results = _coll(mongo, buckets, db_name, config.compliance_results_collection)
    assert results.insert_one.called

    mongo2, buckets2, _seen2 = _mongo_router()
    critic2 = MagicMock()
    critic2.get_active_feedback.side_effect = RuntimeError("feedback store down")
    critic2.evaluate = AsyncMock(side_effect=RuntimeError("critic down"))
    critic2.record_feedback_usage = AsyncMock()
    config2 = _config(adaptive_feedback_enabled=True)
    svc2, llm2 = _service(mongo2, config2, critic=critic2)
    policies2 = _coll(mongo2, buckets2, config2.compliance_database, config2.policies_collection)
    legal2 = _coll(mongo2, buckets2, config2.compliance_database, config2.policy_legal_embeddings_collection)
    mappings2 = _coll(mongo2, buckets2, config2.compliance_database, config2.category_mapping_collection)
    statutes2 = _coll(mongo2, buckets2, config2.compliance_database, config2.statute_sub_topic_embeddings_collection)
    policies2.find_one.return_value = {"document_id": "p1", "text": "Policy."}
    legal2.count_documents.return_value = 1
    mappings2.find.return_value = [_mapping()]
    statutes2.find.return_value = [
        {"_id": "s1", "subtopic_text": "Delete data.", "jurisdiction": "CA"}
    ]

    survived = _run(svc2, save_results=True)

    assert llm2.gap_check_v4.await_count == 1
    assert llm2.gap_check_v4.await_args.kwargs["adaptive_feedback"] == ""
    results2 = _coll(
        mongo2, buckets2, config2.compliance_database, config2.compliance_results_collection
    )
    assert results2.insert_one.called
    critic2.record_feedback_usage.assert_not_called()
    critic2.evaluate.assert_awaited_once()
    assert survived.summary.total_requirements == 1
