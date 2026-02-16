"""Utility endpoints for the Web Gather API.

Includes endpoints: embedding-models POST and GET.
"""
from __future__ import annotations

import logging
from typing import Any

from flask import Blueprint, jsonify, request

from db import (
    EMBEDDING_MODEL_COLLECTION,
    WEB_GATHER_DB,
    get_embedding_model_record,
    utc_now,
)

logger = logging.getLogger("web-gather-api")

VALID_SIMILARITY = frozenset(("euclidean", "cosine", "dotproduct"))

# Will be initialized by app.py
mongo_client = None

util_bp = Blueprint("util", __name__)


def init_util(client):
    """Initialize the util module with the MongoDB client."""
    global mongo_client
    mongo_client = client


def _validate_embedding_model_document(payload: Any) -> tuple[dict | None, str | None]:
    """Validate embedding_model document. Returns (payload, None) on success, (None, error_message) on failure."""
    if not isinstance(payload, dict):
        return None, "Request body must be a JSON object"
    database_name = payload.get("database_name")
    model_name = payload.get("model_name")
    fields = payload.get("fields")
    if not database_name or not isinstance(database_name, str) or not database_name.strip():
        return None, "database_name is required and must be a non-empty string"
    if not model_name or not isinstance(model_name, str) or not model_name.strip():
        return None, "model_name is required and must be a non-empty string"
    if not isinstance(fields, list) or len(fields) == 0:
        return None, "fields is required and must be a non-empty array"
    for i, f in enumerate(fields):
        if not isinstance(f, dict):
            return None, f"fields[{i}] must be an object"
        if f.get("type") != "vector":
            return None, f"fields[{i}].type must be exactly \"vector\""
        path = f.get("path")
        if not path or not isinstance(path, str) or not path.strip():
            return None, f"fields[{i}].path must be a non-empty string"
        num_dims = f.get("numDimensions")
        if num_dims is None or not isinstance(num_dims, (int, float)):
            return None, f"fields[{i}].numDimensions must be a number"
        sim = f.get("similarity")
        if not sim or not isinstance(sim, str):
            return None, f"fields[{i}].similarity is required and must be euclidean, cosine, or dotProduct"
        if sim.lower() not in VALID_SIMILARITY:
            return None, f"fields[{i}].similarity must be one of: euclidean, cosine, dotProduct"
    return payload, None


@util_bp.get("/embedding-models")
def list_embedding_models():
    """List embedding models."""
    logger.info("GET /embedding-models - Listing embedding models")
    database_name = request.args.get("database_name")
    logger.info("GET /embedding-models - Parameters: database_name=%s", database_name)
    wg_db = mongo_client[WEB_GATHER_DB]
    query = {"database_name": database_name} if database_name else {}
    models = list(wg_db[EMBEDDING_MODEL_COLLECTION].find(query, {"_id": 0}))
    logger.info("GET /embedding-models - Found %d models", len(models))
    return jsonify({"models": models})


@util_bp.post("/embedding-models")
def add_embedding_model():
    """Add or update an embedding model document. Body must include database_name, model_name, and fields (vector index definitions)."""
    logger.info("POST /embedding-models - Adding embedding model")
    payload = request.get_json(silent=True) or {}
    doc, err = _validate_embedding_model_document(payload)
    if err is not None:
        logger.warning("POST /embedding-models - Validation failed: %s", err)
        return jsonify({"error": err}), 400

    database_name = doc["database_name"]
    set_doc = {**doc, "updated_at": utc_now()}
    logger.info("POST /embedding-models - Setting model %s for database %s", doc.get("model_name"), database_name)
    wg_db = mongo_client[WEB_GATHER_DB]
    wg_db[EMBEDDING_MODEL_COLLECTION].update_one(
        {"database_name": database_name},
        {"$set": set_doc},
        upsert=True,
    )

    logger.info("POST /embedding-models - Successfully configured model for database %s", database_name)
    return jsonify(set_doc)


def _build_filter_field(path: str) -> dict:
    """Build a MongoDB vector search filter field definition."""
    return {"type": "filter", "path": path}


@util_bp.post("/create-vector-index")
def create_vector_index():
    """Create a vector search index on a collection using the embedding_model document from web-gather.

    Drops the index first if it exists, then creates it. Body: database_name (str), collection_name (str), index_name (str, optional),
    filter_fields (array of strings or objects, optional): field paths to index for pre-filtering
    (e.g. ["document_id", "jurisdiction"]). Each string is converted to {"type": "filter", "path": "..."}.
    Objects must have "path" key. If omitted, uses embedding_model.filter_fields when present.
    Looks up web-gather.embedding_model by database_name and uses its fields for the vector index.
    Requires Atlas (create_search_index).
    """
    logger.info("POST /create-vector-index - Creating vector index")
    payload = request.get_json(silent=True) or {}
    database_name = payload.get("database_name")
    collection_name = payload.get("collection_name")
    index_name = payload.get("index_name") or "vector_index"
    filter_fields_param = payload.get("filter_fields")

    if not database_name or not isinstance(database_name, str) or not database_name.strip():
        return jsonify({"error": "database_name is required and must be a non-empty string"}), 400
    if not collection_name or not isinstance(collection_name, str) or not collection_name.strip():
        return jsonify({"error": "collection_name is required and must be a non-empty string"}), 400
    if not index_name or not isinstance(index_name, str) or not index_name.strip():
        return jsonify({"error": "index_name must be a non-empty string when provided"}), 400

    record = get_embedding_model_record(database_name)
    if not record:
        return jsonify({
            "error": f"No embedding_model configured for database '{database_name}'. Use POST /embedding-models first."
        }), 400

    fields = list(record.get("fields") or [])
    if not fields:
        return jsonify({
            "error": f"embedding_model for '{database_name}' has no valid 'fields' array for vector index."
        }), 400

    # Resolve filter fields: request body overrides embedding_model
    filter_field_defs: list[dict] = []
    if filter_fields_param is not None:
        if isinstance(filter_fields_param, list):
            for item in filter_fields_param:
                if isinstance(item, str) and item.strip():
                    filter_field_defs.append(_build_filter_field(item.strip()))
                elif isinstance(item, dict) and item.get("path"):
                    path = str(item["path"]).strip()
                    if path:
                        filter_field_defs.append(_build_filter_field(path))
        else:
            return jsonify({"error": "filter_fields must be an array of field paths (strings) or objects with 'path'"}), 400
    elif isinstance(record.get("filter_fields"), list):
        for item in record["filter_fields"]:
            if isinstance(item, str) and item.strip():
                filter_field_defs.append(_build_filter_field(item.strip()))
            elif isinstance(item, dict) and item.get("path"):
                path = str(item["path"]).strip()
                if path:
                    filter_field_defs.append(_build_filter_field(path))

    fields = fields + filter_field_defs
    definition = {"fields": fields}
    db = mongo_client[database_name]
    try:
        drop_res = db.command(
            {"dropSearchIndex": collection_name, "name": index_name}
        )
        if drop_res.get("ok"):
            logger.info(
                "POST /create-vector-index - Dropped existing index %r on %s.%s",
                index_name, database_name, collection_name,
            )
    except Exception as drop_err:  # noqa: BLE001
        # Index may not exist; continue to creation
        logger.debug(
            "POST /create-vector-index - Drop index %r (may not exist): %s",
            index_name, drop_err,
        )
    try:
        res = db.command(
            {
                "createSearchIndexes": collection_name,
                "indexes": [
                    {
                        "name": index_name,
                        "type": "vectorSearch",
                        "definition": definition,
                    }
                ],
            }
        )
        if not res.get("ok"):
            return jsonify({"error": f"createSearchIndexes failed: {res}"}), 500
    except Exception as e:  # noqa: BLE001
        logger.exception("POST /create-vector-index - createSearchIndexes failed")
        return jsonify({"error": f"Failed to create vector index: {e!s}"}), 500

    logger.info(
        "POST /create-vector-index - Created index %r on %s.%s (filter_fields=%d)",
        index_name, database_name, collection_name, len(filter_field_defs),
    )
    return jsonify({
        "database_name": database_name,
        "collection_name": collection_name,
        "index_name": index_name,
        "definition": definition,
        "filter_fields_added": len(filter_field_defs),
    })
