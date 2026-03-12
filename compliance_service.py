"""Deprecated: policy-first compliance service.

The statute-policy-compliance endpoint now delegates to GapAnalysisServiceV4
(statute-first approach via category mappings). This module is retained only
for backward compatibility with any external imports.
"""
from __future__ import annotations


class ServiceError(Exception):
    """Retained for backward compatibility."""
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code
