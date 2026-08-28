"""GET /api/compliance/runs `types` query parsing.

Whitespace-only or comma-only `types` currently becomes None (no type filter)
rather than an empty list that would match nothing. Pin that so a later
change cannot silently hide every run or, conversely, start treating `types=`
as a match-nothing filter.
"""
from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

from flask import Flask

sys.modules.setdefault("anthropic", MagicMock())
sys.modules.setdefault("sentence_transformers", MagicMock())

from compliance_config import load_config
from compliance_routes import compliance_bp
import compliance_routes


def _client(list_runs):
    config = load_config()
    config.auth_required = False
    mock_suite = MagicMock()
    mock_suite._storage.list_runs = list_runs
    app = Flask(__name__)
    app.register_blueprint(compliance_bp, url_prefix="/api/compliance")
    return app.test_client(), config, mock_suite


def test_list_runs_empty_types_is_unfiltered() -> None:
    """types=,,, is treated as no type filter (None), not an empty list."""
    list_runs = AsyncMock(return_value=([], 0))
    client, config, mock_suite = _client(list_runs)
    with patch.object(compliance_routes, "_config", config), patch.object(
        compliance_routes, "_get_suite_service", return_value=mock_suite
    ):
        response = client.get("/api/compliance/runs?types=,,%20,")

    assert response.status_code == 200
    list_runs.assert_called_once()
    kwargs = list_runs.call_args.kwargs
    assert kwargs["types"] is None
    assert kwargs["limit"] == 50
    assert kwargs["offset"] == 0


def test_list_runs_types_csv_is_stripped() -> None:
    """Comma-separated types are split and stripped before storage."""
    list_runs = AsyncMock(return_value=([], 0))
    client, config, mock_suite = _client(list_runs)
    with patch.object(compliance_routes, "_config", config), patch.object(
        compliance_routes, "_get_suite_service", return_value=mock_suite
    ):
        response = client.get("/api/compliance/runs?types=gap,%20health,")

    assert response.status_code == 200
    kwargs = list_runs.call_args.kwargs
    assert kwargs["types"] == ["gap", "health"]


def test_list_runs_omitted_types_is_none() -> None:
    """Omitting types also passes None (all run types)."""
    list_runs = AsyncMock(return_value=([], 0))
    client, config, mock_suite = _client(list_runs)
    with patch.object(compliance_routes, "_config", config), patch.object(
        compliance_routes, "_get_suite_service", return_value=mock_suite
    ):
        response = client.get("/api/compliance/runs")

    assert response.status_code == 200
    assert list_runs.call_args.kwargs["types"] is None
