"""Policy segmentation edges and adaptive-feedback audit queries.

Route tests mock history/log retrieval, so the Mongo filter itself needs
a direct check. Segmentation decides which policy text reaches comparison.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from adaptive_feedback_service import CriticService
from segmenter import PolicySegmenter


def test_segmenter_empty_text_and_empty_existing_sections() -> None:
    segmenter = PolicySegmenter(max_section_chars=2000)
    assert segmenter.segment("") == []
    assert segmenter.segment(None) == []
    # An empty list is falsy, so segmentation falls through to policy text.
    sections = segmenter.segment("DATA RETENTION\nKeep records.", existing_sections=[])
    assert len(sections) == 1
    assert sections[0].section_id.startswith("data_retention")


def test_segmenter_existing_sections_skip_blanks_and_split_long_text() -> None:
    segmenter = PolicySegmenter(max_section_chars=6)
    sections = segmenter.segment(
        "ignored because existing sections are present",
        existing_sections=[
            {"section_text": "   "},
            {"section_id": "short", "section_text": "abc"},
            {"title": "Data Retention", "text": "hello world"},
            {"section_id": "intro", "section_text": "abcdefghijkl"},
        ],
    )
    assert [section.section_id for section in sections] == [
        "short",
        "data_retention_1",
        "data_retention_2",
        "intro_1",
        "intro_2",
    ]
    # The cut rstrip/lstrip drops the boundary space.
    assert sections[1].section_text == "hello"
    assert sections[2].section_text == "world"
    assert sections[3].section_text == "abcdef"
    assert sections[4].section_text == "ghijkl"


def test_segmenter_numbered_heading_and_long_uppercase_stays_body() -> None:
    segmenter = PolicySegmenter(max_section_chars=500)
    text = (
        "1. Scope\n\nThis policy applies.\n\n"
        + ("A" * 121)
        + "\n\nSECURITY\n\nWe protect data."
    )
    sections = segmenter.segment(text)
    assert [section.section_id for section in sections] == ["1_scope", "security"]
    assert "This policy applies." in sections[0].section_text
    assert "A" * 121 in sections[0].section_text
    assert "We protect data." in sections[1].section_text


def _critic():
    feedback = MagicMock()
    log = MagicMock()
    db = MagicMock()

    def get_coll(name):
        if name == "adaptive_feedback":
            return feedback
        if name == "adaptive_feedback_log":
            return log
        raise AssertionError(name)

    db.__getitem__.side_effect = get_coll
    mongo = MagicMock()
    mongo.__getitem__.return_value = db
    cfg = MagicMock()
    cfg.compliance_database = "privacy-compliance"
    cfg.adaptive_feedback_collection = "adaptive_feedback"
    cfg.adaptive_feedback_log_collection = "adaptive_feedback_log"
    return CriticService(mongo, cfg, MagicMock()), feedback, log, mongo


def _cursor(docs):
    cursor = MagicMock()
    cursor.sort.return_value = cursor
    cursor.skip.return_value = cursor
    cursor.limit.return_value = docs
    return cursor


def test_feedback_history_query_active_filter_and_id_stringification() -> None:
    critic, feedback, _log, mongo = _critic()
    feedback.find.return_value = _cursor([{"_id": 15, "suggestions": []}])

    docs = critic.get_feedback_history(" pol-1 ", active_only=True, limit=2, offset=3)
    feedback.find.assert_called_once_with(
        {"policy_document_id": " pol-1 ", "superseded_by": None}
    )
    cursor = feedback.find.return_value
    cursor.sort.assert_called_once_with("created_at", -1)
    cursor.skip.assert_called_once_with(3)
    cursor.limit.assert_called_once_with(2)
    assert docs == [{"_id": "15", "suggestions": []}]
    mongo.__getitem__.assert_called_with("privacy-compliance")

    feedback.find.return_value = _cursor([])
    critic.get_feedback_history("pol-1", active_only=False, limit=0, offset=-1)
    feedback.find.assert_called_with({"policy_document_id": "pol-1"})
    cursor = feedback.find.return_value
    cursor.limit.assert_called_once_with(0)
    cursor.skip.assert_called_once_with(-1)


def test_feedback_log_run_filter_treats_blank_as_unfiltered() -> None:
    critic, _feedback, log, _mongo = _critic()
    log.find.return_value = _cursor([{"_id": "log-1", "run_id": "run-1"}])

    docs = critic.get_feedback_log(run_id=None, limit=50, offset=0)
    assert docs[0]["_id"] == "log-1"
    log.find.assert_called_with({})

    log.find.return_value = _cursor([])
    critic.get_feedback_log(run_id="", limit=10, offset=4)
    log.find.assert_called_with({})

    log.find.return_value = _cursor([{"run_id": "   "}])
    rows = critic.get_feedback_log(run_id="   ", limit=10, offset=4)
    log.find.assert_called_with({"run_id": "   "})
    cursor = log.find.return_value
    cursor.sort.assert_called_once_with("created_at", -1)
    cursor.skip.assert_called_once_with(4)
    cursor.limit.assert_called_once_with(10)
    assert rows == [{"run_id": "   "}]
