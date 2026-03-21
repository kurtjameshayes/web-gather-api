"""Shared pytest fixtures and configuration for web-gather-api tests."""
from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pytest
from flask import Flask
from unittest.mock import MagicMock

# Avoid loading sentence_transformers in unit tests. Must run before any import of core.
sys.modules["sentence_transformers"] = MagicMock()


def _wire_mongo(mock_mongo: MagicMock, web_gather_db: str = "web-gather") -> dict[str, MagicMock]:
    """Wire a mock MongoDB client with web-gather and user databases."""
    wg_db = MagicMock()
    user_db = MagicMock()

    def _get_db(name: str) -> MagicMock:
        return wg_db if name == web_gather_db else user_db

    mock_mongo.__getitem__.side_effect = _get_db
    wg_db.__getitem__.return_value = MagicMock()
    user_db.__getitem__.return_value = MagicMock()
    return {"wg_db": wg_db, "user_db": user_db}


def _wire_mongo_core(
    mock_mongo: MagicMock,
    web_gather_db: str = "web-gather",
    index_db_name: str = "index_db",
) -> dict[str, MagicMock]:
    """Wire a mock MongoDB client for core endpoints (source, index, web-gather)."""
    source_db = MagicMock()
    index_db = MagicMock()
    wg_db = MagicMock()

    def _get_db(name: str) -> MagicMock:
        if name == web_gather_db:
            return wg_db
        if name == index_db_name:
            return index_db
        return source_db

    mock_mongo.__getitem__.side_effect = _get_db
    source_db.__getitem__.return_value = MagicMock()
    index_db.__getitem__.return_value = MagicMock()
    wg_db.__getitem__.return_value = MagicMock()
    return {"source_db": source_db, "index_db": index_db, "wg_db": wg_db}


class DummyModel:
    """Minimal embedding model for tests."""

    def encode(self, inputs, **kwargs: object) -> np.ndarray:
        n = len(inputs) if hasattr(inputs, "__len__") else 1
        return np.array([[1.0, 0.0]] * n)


@pytest.fixture
def mock_mongo_client():
    """Create a mock MongoDB client."""
    return MagicMock()


@pytest.fixture
def mock_firecrawl():
    """Create a mock Firecrawl client."""
    return MagicMock()


@pytest.fixture
def mock_anthropic():
    """Create a mock Anthropic client."""
    return MagicMock()


@pytest.fixture
def mock_clients_core():
    """Mock MongoDB, Firecrawl, Anthropic clients for core endpoints. Returns dict of mocks."""
    from core import init_core

    mock_mongo = MagicMock()
    mock_firecrawl = MagicMock()
    mock_anthropic = MagicMock()
    init_core(mock_mongo, mock_firecrawl, mock_anthropic)
    return {"mongo": mock_mongo, "firecrawl": mock_firecrawl, "anthropic": mock_anthropic}


@pytest.fixture
def app_core(mock_clients_core):
    """Minimal Flask app with core blueprint only (uses mock_clients_core)."""
    from core import core_bp

    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app


@pytest.fixture
def mock_mongo_db():
    """Mock MongoDB client for db blueprint. Use to configure DB responses."""
    from db import init_db

    mock_mongo = MagicMock()
    init_db(mock_mongo)
    return mock_mongo


@pytest.fixture
def app_db(mock_mongo_db):
    """Minimal Flask app with db blueprint only."""
    from db import db_bp

    app = Flask(__name__)
    app.register_blueprint(db_bp)
    return app


@pytest.fixture
def mock_mongo_util():
    """Mock MongoDB client for util blueprint. Use to configure DB responses."""
    from util import init_util

    mock_mongo = MagicMock()
    init_util(mock_mongo)
    return mock_mongo


@pytest.fixture
def app_util(mock_mongo_util):
    """Minimal Flask app with util blueprint only."""
    from util import util_bp

    app = Flask(__name__)
    app.register_blueprint(util_bp)
    return app


@pytest.fixture
def client_core(app_core):
    """Test client for core blueprint. Use with mock_clients_core to configure mocks."""
    return app_core.test_client()


@pytest.fixture
def client_db(app_db):
    """Test client for db blueprint."""
    return app_db.test_client()


@pytest.fixture
def client_util(app_util):
    """Test client for util blueprint."""
    return app_util.test_client()
