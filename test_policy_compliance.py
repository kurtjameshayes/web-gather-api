"""Tests for policy statute compliance service."""
import asyncio
import json
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest
from flask import Flask

# Avoid loading sentence_transformers in unit tests.
sys.modules["sentence_transformers"] = MagicMock()

from cache import SimpleLRUCache
from compliance_config import load_config
from compliance_evaluator import ComplianceEvaluator
from compliance_routes import compliance_bp, set_compliance_service
from compliance_service import ComplianceService
from llm_client import build_prompt
from redactor import Redactor
from segmenter import PolicySegmenter
from vector_retriever import StatuteCandidate, VectorRetriever
from rate_limiter import RateLimiter
from schemas import PolicyStatuteComplianceResponse


def test_segmenter_splits_on_headings():
    segmenter = PolicySegmenter(max_section_chars=2000)
    text = "DATA RETENTION\nWe keep data for analytics.\n\nSECURITY\nWe protect data."
    sections = segmenter.segment(text)
    assert len(sections) == 2
    assert sections[0].section_id.startswith("data_retention")
    assert "We keep data" in sections[0].section_text


def test_build_prompt_uses_template():
    candidate = StatuteCandidate(
        statute_id="stat-1",
        jurisdiction="US",
        title="Retention Limits",
        section_id="R-1",
        chunk_text="Data retention must not exceed 3 years.",
        score=0.87,
        chunk_id="chunk-1",
    )
    prompt = build_prompt("We retain user data for 10 years.", [candidate])
    assert "You are a privacy law analyst." in prompt
    assert "Candidate statutes (top K):" in prompt
    assert "[stat-1] US Retention Limits" in prompt
    assert "Output JSON exactly with keys:" in prompt


def test_evaluator_respects_confidence_threshold():
    evaluator = ComplianceEvaluator(evidence_score_threshold=0.2)
    candidates = [
        StatuteCandidate(
            statute_id="stat-1",
            jurisdiction="US",
            title="Retention Limits",
            section_id="R-1",
            chunk_text="Data retention must not exceed 3 years.",
            score=0.9,
            chunk_id="chunk-1",
        )
    ]
    llm_payload = json.dumps(
        {
            "section_id": "section_1",
            "applied_statutes": [
                {
                    "statute_id": "stat-1",
                    "jurisdiction": "US",
                    "title": "Retention Limits",
                    "matched_span": "10 years",
                    "evidence_score": 0.9,
                }
            ],
            "compliance": "compliant",
            "confidence": 0.5,
            "rationale": "Insufficient alignment.",
            "remediation_suggestions": ["Shorten retention."],
        }
    )
    result = evaluator.evaluate(
        section_id="section_1",
        section_text="We retain user data for 10 years.",
        candidates=candidates,
        llm_response_text=llm_payload,
        thresholds=load_config().confidence_thresholds,
        explainability=True,
    )
    assert result.compliance == "neither"


def test_retriever_returns_statute_candidates():
    class DummyEmbedder:
        async def embed(self, text):
            return [0.1, 0.2]

    mock_collection = MagicMock()
    mock_collection.aggregate.return_value = [
        {
            "_id": "stat-1",
            "jurisdiction": "US",
            "title": "Retention Limits",
            "section_text": "Retention must not exceed 3 years.",
            "section_id": "R-1",
            "score": 0.77,
        }
    ]
    mock_client = MagicMock()
    mock_client.__getitem__.return_value.__getitem__.return_value = mock_collection

    config = load_config()
    config.use_embeddings_collection = False

    retriever = VectorRetriever(
        mock_client,
        DummyEmbedder(),
        config,
        SimpleLRUCache(10, 60),
    )

    results = asyncio.run(
        retriever.retrieve(
            database="privacy_db",
            section_text="We retain data for 10 years.",
            jurisdiction="US",
            statute_corpus_id=None,
            top_k=1,
        )
    )
    assert results[0].statute_id == "stat-1"
    assert results[0].score == 0.77


class StubRetriever:
    async def retrieve(
        self,
        database,
        section_text,
        jurisdiction,
        statute_corpus_id,
        top_k,
    ):
        return [
            StatuteCandidate(
                statute_id="stat-901",
                jurisdiction=jurisdiction,
                title="Data Retention Limits",
                section_id="DR-3",
                chunk_text="Data retention for analytics must not exceed 3 years.",
                score=0.92,
                chunk_id="DR-3",
            )
        ]


class StubLLM:
    async def compare_section(self, section_id, section_text, candidates):
        return json.dumps(
            {
                "section_id": section_id,
                "applied_statutes": [
                    {
                        "statute_id": "stat-901",
                        "jurisdiction": "US",
                        "title": "Data Retention Limits",
                        "matched_span": "retain user data for 10 years",
                        "evidence_score": 0.92,
                    }
                ],
                "compliance": "non_compliant",
                "confidence": 0.86,
                "rationale": "Policy retention exceeds statutory limit.",
                "remediation_suggestions": [
                    "Reduce retention to 3 years or justify exception."
                ],
            }
        )


def test_policy_compliance_endpoint_retention_non_compliant():
    config = load_config()
    config.enable_audit_logging = False
    config.enable_redaction = True
    config.auth_required = False

    service = ComplianceService(
        mongo_client=MagicMock(),
        config=config,
        segmenter=PolicySegmenter(config.max_section_chars),
        retriever=StubRetriever(),
        llm_client=StubLLM(),
        evaluator=ComplianceEvaluator(config.evidence_score_threshold),
        redactor=Redactor(),
        audit_logger=MagicMock(log=AsyncMock()),
        rate_limiter=RateLimiter(1000),
    )

    set_compliance_service(service, config)
    app = Flask(__name__)
    app.register_blueprint(compliance_bp)
    client = app.test_client()
    response = client.post(
        "/policy-statute-compliance",
        json={
            "database": "privacy_db",
            "policy_collection": "policies",
            "text": "We retain user data for 10 years for analytics.",
            "jurisdiction": "US",
            "top_k_statutes": 5,
            "confidence_thresholds": {"compliant": 0.75, "non_compliant": 0.75},
            "options": {"explainability": True, "redact_pii": True},
        },
    )

    assert response.status_code == 200
    body = response.json()
    PolicyStatuteComplianceResponse.model_validate(body)
    assert body["sections"][0]["compliance"] == "non_compliant"
    assert body["sections"][0]["confidence"] >= 0.8
    assert "Reduce retention to 3 years" in body["sections"][0]["remediation_suggestions"][0]

    set_compliance_service(None)


def test_adversarial_ambiguous_language_neither():
    evaluator = ComplianceEvaluator(evidence_score_threshold=0.2)
    candidates = [
        StatuteCandidate(
            statute_id="stat-2",
            jurisdiction="US",
            title="Consent Requirements",
            section_id="C-1",
            chunk_text="Explicit consent is required for data sharing.",
            score=0.88,
            chunk_id="C-1",
        )
    ]
    llm_payload = json.dumps(
        {
            "section_id": "section_1",
            "applied_statutes": [],
            "compliance": "neither",
            "confidence": 0.4,
            "rationale": "The policy language is ambiguous.",
            "remediation_suggestions": ["Clarify consent requirements."],
        }
    )
    result = evaluator.evaluate(
        section_id="section_1",
        section_text="We may share data as appropriate.",
        candidates=candidates,
        llm_response_text=llm_payload,
        thresholds=load_config().confidence_thresholds,
        explainability=False,
    )
    assert result.compliance == "neither"


def test_precision_recall_f1_on_labeled_dataset():
    labeled = [
        ("retain 10 years", "non_compliant"),
        ("we obtain consent", "compliant"),
        ("ambiguous statement", "neither"),
    ]
    predicted = ["non_compliant", "compliant", "neither"]

    def metrics(labels, preds, target):
        tp = sum(1 for l, p in zip(labels, preds) if l == target and p == target)
        fp = sum(1 for l, p in zip(labels, preds) if l != target and p == target)
        fn = sum(1 for l, p in zip(labels, preds) if l == target and p != target)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return precision, recall, f1

    labels = [item[1] for item in labeled]
    precision, recall, f1 = metrics(labels, predicted, "non_compliant")
    assert precision == 1.0
    assert recall == 1.0
    assert f1 == 1.0
