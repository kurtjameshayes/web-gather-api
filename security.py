"""Authorization utilities for Flask endpoint."""
from __future__ import annotations

from compliance_config import ComplianceConfig


class AuthorizationError(Exception):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def authorize_request(config: ComplianceConfig, request) -> None:
    if not config.auth_required:
        return

    api_key = (request.headers.get("x-api-key") or "").strip()
    if not api_key:
        raise AuthorizationError("Missing API key.", 401)
    if config.api_key and api_key != config.api_key:
        raise AuthorizationError("Invalid API key.", 403)

    if config.allowed_roles:
        role_header = config.default_role_header or "x-role"
        role = (request.headers.get(role_header) or "").strip().lower()
        allowed = {r.lower() for r in config.allowed_roles}
        if role not in allowed:
            raise AuthorizationError("Insufficient role.", 403)
