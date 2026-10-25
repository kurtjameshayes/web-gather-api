"""Pin adaptive-feedback query coercion at the route, before storage.

Storage does not clamp limit. MongoDB treats limit 0 as no limit, so the route
floors limit at 1 and caps it at 200. active_only is a case-insensitive token
match and is not stripped, so a padded "yes" does not hide superseded rows.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

import compliance_routes
from compliance_config import load_config
from compliance_routes_v4 import compliance_v4_bp


@pytest.fixture
def client():
    config = load_config()
    config.auth_required = False
    app = Flask(__name__)
    app.register_blueprint(compliance_v4_bp, url_prefix="/api/v4/compliance")
    critic = MagicMock()
    critic.get_feedback_history.return_value = []
    critic.get_feedback_log.return_value = []
    with patch.object(compliance_routes, "_config", config), patch.object(
        compliance_routes, "_critic_service", critic
    ):
        yield app.test_client(), critic


@pytest.mark.parametrize(
    ("token", "active_only"),
    [
        ("YES", True),
        ("1", True),
        ("yes", True),
        ("false", False),
        ("0", False),
        ("no", False),
        (" true", False),
        ("yes ", False),
    ],
)
def test_active_only_token_is_not_stripped(client, token, active_only) -> None:
    http, critic = client
    response = http.get(
        "/api/v4/compliance/adaptive-feedback",
        query_string={"policy_document_id": "pol-1", "active_only": token},
    )
    assert response.status_code == 200
    critic.get_feedback_history.assert_called_once_with(
        policy_document_id="pol-1",
        active_only=active_only,
        limit=50,
        offset=0,
    )


def test_omitted_active_only_is_false(client) -> None:
    http, critic = client
    response = http.get(
        "/api/v4/compliance/adaptive-feedback",
        query_string={"policy_document_id": "pol-1"},
    )
    assert response.status_code == 200
    assert critic.get_feedback_history.call_args.kwargs["active_only"] is False


@pytest.mark.parametrize(
    ("limit", "offset", "expected_limit", "expected_offset"),
    [
        ("0", "0", 1, 0),
        ("-8", "0", 1, 0),
        ("201", "0", 200, 0),
        ("nope", "-2", 50, 0),
        ("10", "nope", 10, 0),
    ],
)
def test_feedback_history_limit_never_reaches_storage_as_zero(
    client, limit, offset, expected_limit, expected_offset
) -> None:
    http, critic = client
    response = http.get(
        "/api/v4/compliance/adaptive-feedback",
        query_string={
            "policy_document_id": "pol-1",
            "limit": limit,
            "offset": offset,
        },
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["limit"] == expected_limit
    assert body["offset"] == expected_offset
    critic.get_feedback_history.assert_called_once_with(
        policy_document_id="pol-1",
        active_only=False,
        limit=expected_limit,
        offset=expected_offset,
    )


@pytest.mark.parametrize(
    ("limit", "expected_limit"),
    [
        ("0", 1),
        ("500", 200),
    ],
)
def test_feedback_log_limit_is_clamped(client, limit, expected_limit) -> None:
    http, critic = client
    response = http.get(
        "/api/v4/compliance/adaptive-feedback/log",
        query_string={"run_id": "run-1", "limit": limit},
    )
    assert response.status_code == 200
    assert response.get_json()["limit"] == expected_limit
    critic.get_feedback_log.assert_called_once_with(
        run_id="run-1",
        limit=expected_limit,
        offset=0,
    )
