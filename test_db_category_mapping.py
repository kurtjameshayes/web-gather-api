"""Tests for db blueprint category-mapping endpoints."""
from __future__ import annotations

import json
from urllib.parse import quote

from bson import ObjectId
from flask import Flask
from unittest.mock import MagicMock

import pytest

from db import (
    CATEGORY_MAPPING_COLLECTION,
    PRIVACY_COMPLIANCE_DB,
    db_bp,
    init_db,
)


@pytest.fixture
def mock_mongo():
    """Mock MongoDB client for category-mapping (privacy-compliance DB)."""
    mock_mongo = MagicMock()
    init_db(mock_mongo)
    return mock_mongo


@pytest.fixture
def mock_coll(mock_mongo):
    """Mock category_mapping collection."""
    pc_db = MagicMock()
    mock_coll = MagicMock()
    pc_db.__getitem__.return_value = mock_coll
    mock_mongo.__getitem__.return_value = pc_db
    return mock_coll


@pytest.fixture
def app(mock_mongo):
    """Flask app with db blueprint."""
    app = Flask(__name__)
    app.register_blueprint(db_bp)
    return app


@pytest.fixture
def client(app):
    """Test client."""
    return app.test_client()


def test_get_category_mapping_success(client, mock_coll) -> None:
    """GET /category-mapping returns category mappings."""
    mock_coll.find.return_value = [
        {"_id": ObjectId(), "statute_category": "retention", "policy_categories": ["data_retention"]},
    ]
    response = client.get("/category-mapping")
    assert response.status_code == 200
    data = response.get_json()
    assert "category_mappings" in data
    assert len(data["category_mappings"]) == 1
    assert data["category_mappings"][0]["statute_category"] == "retention"


def test_get_category_mapping_filtered_by_statute_category(client, mock_coll) -> None:
    """GET /category-mapping?statute_category=X filters results."""
    mock_coll.find.return_value = [
        {"_id": ObjectId(), "statute_category": "retention", "policy_categories": ["data_retention"]},
    ]
    response = client.get("/category-mapping?statute_category=retention")
    assert response.status_code == 200
    mock_coll.find.assert_called_once_with({"statute_category": "retention"})


def test_get_category_mapping_filtered_by_sub_topic(client, mock_coll) -> None:
    """GET /category-mapping?sub_topic=X filters results."""
    mock_coll.find.return_value = []
    response = client.get("/category-mapping?sub_topic=right_to_delete")
    assert response.status_code == 200
    mock_coll.find.assert_called_once_with({"sub_topic": "right_to_delete"})


def test_get_category_mapping_invalid_query_json(client, mock_coll) -> None:
    """GET /category-mapping?query={bad} returns 400."""
    response = client.get("/category-mapping?query={bad}")
    assert response.status_code == 400
    assert "JSON" in response.get_json()["error"]


def test_get_category_mapping_rejects_dangerous_query_operator(client, mock_coll) -> None:
    """GET /category-mapping rejects raw Mongo operators before collection access."""
    encoded_query = quote(json.dumps({"$where": "this.statute_category == 'retention'"}))

    response = client.get(f"/category-mapping?query={encoded_query}")

    assert response.status_code == 400
    assert "$where" in response.get_json()["error"]
    mock_coll.find.assert_not_called()


def test_post_category_mapping_success(client, mock_coll) -> None:
    """POST /category-mapping creates a new mapping."""
    mock_coll.insert_one.return_value.inserted_id = ObjectId()
    response = client.post(
        "/category-mapping",
        json={
            "statute_category": "data_retention",
            "policy_categories": ["data_retention", "retention_periods"],
        },
    )
    assert response.status_code == 201
    data = response.get_json()
    assert "inserted_id" in data
    assert data["database"] == PRIVACY_COMPLIANCE_DB
    assert data["collection"] == CATEGORY_MAPPING_COLLECTION


def test_post_category_mapping_missing_statute_category(client) -> None:
    """POST /category-mapping without statute_category returns 400."""
    response = client.post(
        "/category-mapping",
        json={"policy_categories": ["x"]},
    )
    assert response.status_code == 400
    assert "statute_category" in response.get_json()["error"]


def test_post_category_mapping_missing_policy_categories(client) -> None:
    """POST /category-mapping without policy_categories returns 400."""
    response = client.post(
        "/category-mapping",
        json={"statute_category": "x"},
    )
    assert response.status_code == 400
    assert "policy_categories" in response.get_json()["error"]


def test_post_category_mapping_policy_categories_not_array(client) -> None:
    """POST /category-mapping with policy_categories not array returns 400."""
    response = client.post(
        "/category-mapping",
        json={"statute_category": "x", "policy_categories": "not-an-array"},
    )
    assert response.status_code == 400
    assert "array" in response.get_json()["error"]


def test_delete_category_mapping_by_id(client, mock_coll) -> None:
    """DELETE /category-mapping?_id=X deletes by ObjectId."""
    mock_coll.delete_many.return_value.deleted_count = 1
    oid = ObjectId()
    response = client.delete(f"/category-mapping?_id={oid}")
    assert response.status_code == 200
    data = response.get_json()
    assert data["deleted_count"] == 1


def test_delete_category_mapping_by_query(client, mock_coll) -> None:
    """DELETE /category-mapping?query={...} deletes by query."""
    mock_coll.delete_many.return_value.deleted_count = 3
    response = client.delete(
        "/category-mapping?query=%7B%22statute_category%22%3A%20%22old%22%7D"
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["deleted_count"] == 3
    mock_coll.delete_many.assert_called_once_with({"statute_category": "old"})


def test_delete_category_mapping_rejects_nested_query_operator(client, mock_coll) -> None:
    """DELETE /category-mapping rejects nested Mongo operators before deletion."""
    encoded_query = quote(json.dumps({"policy_categories": {"$regex": "retention"}}))

    response = client.delete(f"/category-mapping?query={encoded_query}")

    assert response.status_code == 400
    assert "$regex" in response.get_json()["error"]
    mock_coll.delete_many.assert_not_called()


def test_delete_category_mapping_both_id_and_query(client) -> None:
    """DELETE /category-mapping with both _id and query returns 400."""
    response = client.delete("/category-mapping?_id=123&query=%7B%7D")
    assert response.status_code == 400
    assert "both" in response.get_json()["error"].lower() or "either" in response.get_json()["error"].lower()


def test_delete_category_mapping_neither_id_nor_query(client) -> None:
    """DELETE /category-mapping with neither _id nor query returns 400."""
    response = client.delete("/category-mapping")
    assert response.status_code == 400
    assert "_id" in response.get_json()["error"].lower() or "query" in response.get_json()["error"].lower()


def test_delete_category_mapping_invalid_id(client) -> None:
    """DELETE /category-mapping?_id=invalid returns 400."""
    response = client.delete("/category-mapping?_id=not-valid-oid")
    assert response.status_code == 400
    assert "Invalid" in response.get_json()["error"]
