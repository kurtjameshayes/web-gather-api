"""Case and whitespace contracts for the three citation-binding helpers.

v3 and v4 fold case and collapse whitespace before the substring check.
The suite helper collapses whitespace and stays case-sensitive, and a
whitespace-only quote does not bind.
"""
from __future__ import annotations

from compliance_suite_service import _citation_binding as bind_suite
from gap_analysis_service_v3 import _citation_binding as bind_v3
from gap_analysis_service_v4 import _citation_binding as bind_v4


def test_v3_and_v4_binding_is_case_insensitive() -> None:
    policy = "Prefix. We   COLLECT\n\tData. Suffix."
    quote = "we collect data."

    assert bind_v3(quote, policy) is True
    assert bind_v4(quote, policy) is True
    assert bind_v3("WE\tCOLLECT  DATA.", policy) is True
    assert bind_v4("WE\tCOLLECT  DATA.", policy) is True

    assert bind_v3("we collect cookies.", policy) is False
    assert bind_v4("we collect cookies.", policy) is False
    assert bind_v3(None, policy) is False
    assert bind_v4("we collect data.", "") is False
    assert bind_v4("we collect data.", None) is False


def test_suite_binding_keeps_case_and_rejects_blank_quotes() -> None:
    policy = "Prefix. We   COLLECT\n\tData. Suffix."

    assert bind_suite("We COLLECT Data.", policy) is True
    assert bind_suite("we collect data.", policy) is False
    assert bind_suite("We\nCOLLECT\tData.", "We COLLECT Data.") is True

    assert bind_suite(" \n\t ", policy) is False
    assert bind_suite("", policy) is False
    assert bind_suite(None, policy) is False
    assert bind_suite("We COLLECT Data.", "") is False
    assert bind_suite("We COLLECT Data.", None) is False
