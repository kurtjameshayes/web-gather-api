"""Authorization utilities for Flask endpoints."""
from __future__ import annotations

import hmac
import logging
import os
from functools import wraps
from typing import Optional

from flask import jsonify, request as flask_request

from compliance_config import ComplianceConfig

logger = logging.getLogger("web-gather-api")

# Global API key for core/db/util endpoints (set from APP_API_KEY env var).
_app_api_key: Optional[str] = None


def init_app_api_key() -> None:
    """Load application-level API key from environment."""
    global _app_api_key
    _app_api_key = (os.getenv("APP_API_KEY") or "").strip() or None
    if _app_api_key:
        logger.info("Application API key configured for core/db/util endpoints")
    else:
        logger.warning(
            "APP_API_KEY not set — core/db/util endpoints are unauthenticated. "
            "Set APP_API_KEY to require authentication."
        )


class AuthorizationError(Exception):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def _constant_time_compare(a: str, b: str) -> bool:
    """Compare two strings in constant time to prevent timing attacks."""
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def authorize_request(config: ComplianceConfig, request) -> None:
    """Authorize a compliance endpoint request using config-based auth."""
    if not config.auth_required:
        return

    api_key = (request.headers.get("x-api-key") or "").strip()
    if not api_key:
        raise AuthorizationError("Missing API key.", 401)
    if config.api_key and not _constant_time_compare(api_key, config.api_key):
        raise AuthorizationError("Invalid API key.", 403)

    if config.allowed_roles:
        role_header = config.default_role_header or "x-role"
        role = (request.headers.get(role_header) or "").strip().lower()
        allowed = {r.lower() for r in config.allowed_roles}
        if role not in allowed:
            raise AuthorizationError("Insufficient role.", 403)


def require_api_key(f):
    """Decorator that enforces APP_API_KEY authentication on core/db/util endpoints.

    When APP_API_KEY is not configured, requests pass through (backwards compatible).
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        if _app_api_key is not None:
            provided = (flask_request.headers.get("x-api-key") or "").strip()
            if not provided:
                return jsonify({"error": "Missing API key."}), 401
            if not _constant_time_compare(provided, _app_api_key):
                return jsonify({"error": "Invalid API key."}), 403
        return f(*args, **kwargs)
    return decorated


def require_api_key_async(f):
    """Async variant of require_api_key for async route handlers."""
    @wraps(f)
    async def decorated(*args, **kwargs):
        if _app_api_key is not None:
            provided = (flask_request.headers.get("x-api-key") or "").strip()
            if not provided:
                return jsonify({"error": "Missing API key."}), 401
            if not _constant_time_compare(provided, _app_api_key):
                return jsonify({"error": "Invalid API key."}), 403
        return await f(*args, **kwargs)
    return decorated
