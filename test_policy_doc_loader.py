"""Regression tests for gap-analysis policy document lookup helpers."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

from bson import ObjectId

# llm_client imports anthropic at module load; stub for lightweight unit tests.
sys.modules.setdefault("anthropic", MagicMock())

from compliance_config import load_config
from gap_analysis_service_v3 import GapAnalysisServiceV3
from gap_analysis_service_v4 import GapAnalysisServiceV4


def _bare_service(cls):
    """Construct service instances without running heavy __init__ dependencies."""
    service = cls.__new__(cls)
    service.cfg = load_config()
    return service


def test_v4_load_policy_doc_falls_back_to_object_id() -> None:
    """When document_id misses, v4 should retry with ObjectId(_id)."""
    service = _bare_service(GapAnalysisServiceV4)
    oid = ObjectId()
    policies = MagicMock()
    policies.find_one.side_effect = [
        None,
        {"_id": oid, "document_id": "pol-oid", "text": "Policy body"},
    ]

    doc = service._load_policy_doc(policies, str(oid))

    assert doc is not None
    assert doc["document_id"] == "pol-oid"
    assert policies.find_one.call_args_list[0].args[0] == {
        service.cfg.policy_document_id_field: str(oid)
    }
    assert policies.find_one.call_args_list[1].args[0] == {"_id": oid}


def test_v3_load_policy_doc_falls_back_to_raw_id_when_object_id_invalid() -> None:
    """Invalid ObjectId strings should fall back to a raw _id equality lookup."""
    service = _bare_service(GapAnalysisServiceV3)
    policies = MagicMock()
    policies.find_one.side_effect = [
        None,
        {"_id": "legacy-pol", "text": "Legacy policy"},
    ]

    doc = service._load_policy_doc(policies, "legacy-pol")

    assert doc is not None
    assert doc["text"] == "Legacy policy"
    assert policies.find_one.call_args_list[0].args[0] == {
        service.cfg.policy_document_id_field: "legacy-pol"
    }
    assert policies.find_one.call_args_list[1].args[0] == {"_id": "legacy-pol"}


def test_load_policy_doc_returns_none_when_all_lookups_miss() -> None:
    """Both v3 and v4 should return None when no policy document exists."""
    for cls in (GapAnalysisServiceV3, GapAnalysisServiceV4):
        service = _bare_service(cls)
        policies = MagicMock()
        policies.find_one.return_value = None

        assert service._load_policy_doc(policies, "missing-policy") is None
        assert policies.find_one.call_count >= 2
