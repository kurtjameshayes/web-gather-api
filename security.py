"""Authorization utilities for FastAPI endpoint."""
from __future__ import annotations

from fastapi import HTTPException, Request

from compliance_config import ComplianceConfig


def authorize_request(config: ComplianceConfig, request: Request) -> None:
    if not config.auth_required:
        return

    api_key = (request.headers.get("x-api-key") or "").strip()
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API key.")
    if config.api_key and api_key != config.api_key:
        raise HTTPException(status_code=403, detail="Invalid API key.")

    if config.allowed_roles:
        role_header = config.default_role_header or "x-role"
        role = (request.headers.get(role_header) or "").strip().lower()
        allowed = {r.lower() for r in config.allowed_roles}
        if role not in allowed:
            raise HTTPException(status_code=403, detail="Insufficient role.")
