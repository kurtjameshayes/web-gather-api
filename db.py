"""Database-related functions and endpoints.

Includes functionality for endpoints: all-collections, all-databases, collections, count-documents, databases, documents, write_to_collection, and category-mapping.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from bson import ObjectId
from bson.errors import InvalidId
from flask import Blueprint, Response, jsonify, request

logger = logging.getLogger("web-gather-api")

# Will be initialized by app.py
mongo_client = None
WEB_GATHER_DB = "web-gather"
DOCUMENTS_COLLECTION = "documents"
EMBEDDING_MODEL_COLLECTION = "embedding_model"
PRIVACY_COMPLIANCE_DB = "privacy-compliance"
CATEGORY_MAPPING_COLLECTION = "category_mapping"

# Application default embedding model (set at startup from web-gather for privacy-compliance).
application_embedding_model_name: str | None = None

db_bp = Blueprint("db", __name__)


def set_application_embedding_model(name: str | None) -> None:
    """Set the application default embedding model (used for privacy-compliance until restart)."""
    global application_embedding_model_name
    application_embedding_model_name = name


def get_application_embedding_model() -> str | None:
    """Return the application default embedding model, or None if not set."""
    return application_embedding_model_name


def init_db(client):
    """Initialize the database module with the MongoDB client."""
    global mongo_client
    mongo_client = client


def ensure_privacy_compliance_indexes(client, database_name: str = PRIVACY_COMPLIANCE_DB) -> None:
    """Create unique indexes on privacy-compliance collections if they do not exist.

    Idempotent: safe to call on every startup. Uses create_index which is a no-op
    when the index already exists.
    """
    db = client[database_name]
    index_specs: list[tuple[str, list[tuple[str, int]], bool]] = [
        ("statutes", [("document_id", 1)], True),
        ("policies", [("document_id", 1)], True),
        ("statute_chunks", [("document_id", 1), ("chunk_index", 1)], True),
        ("policy_chunks", [("document_id", 1), ("chunk_index", 1)], True),
        ("policy_sub_chunks", [("document_id", 1), ("subchunk_id", 1)], True),
        ("statute_sub_chunks", [("document_id", 1), ("subchunk_id", 1)], True),
        ("policy_sub_embeddings", [("document_id", 1), ("subchunk_id", 1)], True),
        ("statute_sub_embeddings", [("document_id", 1), ("subchunk_id", 1)], True),
    ]
    for coll_name, keys, unique in index_specs:
        try:
            coll = db[coll_name]
            coll.create_index(keys, unique=unique)
            logger.info("Ensured unique index on %s.%s: %s", database_name, coll_name, keys)
        except Exception as e:
            logger.warning(
                "Could not ensure index on %s.%s (%s): %s",
                database_name,
                coll_name,
                keys,
                e,
            )


def utc_now():
    """Return current UTC time."""
    return datetime.now(timezone.utc)


def _convert_extended_json(obj):
    """Recursively convert MongoDB Extended JSON types (e.g. {"$oid": "..."}) to native BSON types."""
    if isinstance(obj, dict):
        if len(obj) == 1 and "$oid" in obj:
            try:
                return ObjectId(obj["$oid"])
            except (InvalidId, TypeError):
                return obj
        return {k: _convert_extended_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_extended_json(item) for item in obj]
    return obj


def get_embedding_model_name(database_name: str):
    """Look up the embedding model configured for a database."""
    logger.info("Looking up embedding model for database: %s", database_name)
    wg_db = mongo_client[WEB_GATHER_DB]
    record = wg_db[EMBEDDING_MODEL_COLLECTION].find_one(
        {"database_name": database_name}
    )
    if not record:
        logger.warning("No embedding model configured for database: %s", database_name)
        return None
    model_name = record.get("model_name")
    logger.info("Found embedding model: %s for database: %s", model_name, database_name)
    return model_name


def get_embedding_model_record(database_name: str):
    """Return the full embedding_model document for a database, or None."""
    if not mongo_client:
        return None
    wg_db = mongo_client[WEB_GATHER_DB]
    return wg_db[EMBEDDING_MODEL_COLLECTION].find_one(
        {"database_name": database_name}
    )


@db_bp.get("/documents")
def list_documents():
    """List documents in a MongoDB collection."""
    logger.info("GET /documents - Listing documents")
    database_name = request.args.get("database_name")
    collection_name = request.args.get("collection_name")
    query_param = request.args.get("query")
    logger.info("GET /documents - Parameters: database_name=%s, collection_name=%s, query=%s", database_name, collection_name, query_param)
    if not database_name or not collection_name:
        logger.warning("GET /documents - Missing required parameters")
        return jsonify({"error": "database_name and collection_name are required"}), 400

    # Parse the query parameter if provided
    mongo_query = {}
    if query_param:
        try:
            mongo_query = json.loads(query_param)
            if isinstance(mongo_query, str):
                mongo_query = json.loads(mongo_query)
            if not isinstance(mongo_query, dict):
                logger.warning("GET /documents - Query must be a JSON object")
                return jsonify({"error": "query must be a JSON object"}), 400
            mongo_query = _convert_extended_json(mongo_query)
        except json.JSONDecodeError as e:
            logger.warning("GET /documents - Invalid JSON in query parameter: %s", str(e))
            return jsonify({"error": f"Invalid JSON in query parameter: {str(e)}"}), 400

    logger.info("GET /documents - Querying %s.%s with query: %s", database_name, collection_name, mongo_query)
    db = mongo_client[database_name]
    docs = list(db[collection_name].find(mongo_query))
    for doc in docs:
        if "_id" in doc:
            doc["_id"] = str(doc["_id"])
    logger.info("GET /documents - Found %d documents", len(docs))
    return jsonify({"documents": docs})


@db_bp.delete("/documents")
def delete_documents() -> Response:
    """Delete documents in a MongoDB collection."""
    logger.info("DELETE /documents - Deleting documents")
    database_name = request.args.get("database_name")
    collection_name = request.args.get("collection_name")
    query_param = request.args.get("query")
    logger.info(
        "DELETE /documents - Parameters: database_name=%s, collection_name=%s, "
        "query=%s",
        database_name,
        collection_name,
        query_param,
    )

    if not database_name or not collection_name:
        logger.warning("DELETE /documents - Missing required parameters")
        return jsonify({"error": "database_name and collection_name are required"}), 400

    mongo_query = {}
    if query_param:
        try:
            mongo_query = json.loads(query_param)
            if isinstance(mongo_query, str):
                mongo_query = json.loads(mongo_query)
            if not isinstance(mongo_query, dict):
                logger.warning("DELETE /documents - Query must be a JSON object")
                return jsonify({"error": "query must be a JSON object"}), 400
        except json.JSONDecodeError as exc:
            logger.warning(
                "DELETE /documents - Invalid JSON in query parameter: %s",
                str(exc),
            )
            return (
                jsonify({"error": f"Invalid JSON in query parameter: {str(exc)}"}),
                400,
            )

    logger.info(
        "DELETE /documents - Deleting from %s.%s with query: %s",
        database_name,
        collection_name,
        mongo_query,
    )
    db = mongo_client[database_name]
    result = db[collection_name].delete_many(mongo_query)
    logger.info(
        "DELETE /documents - Deleted %d documents",
        result.deleted_count,
    )

    return jsonify({
        "database_name": database_name,
        "collection_name": collection_name,
        "deleted_count": result.deleted_count,
        "message": "Documents deleted successfully",
    })


@db_bp.get("/databases")
def list_databases():
    """List all databases with uploaded documents."""
    logger.info("GET /databases - Listing databases")
    logger.info("GET /databases - Parameters: (none)")
    wg_db = mongo_client[WEB_GATHER_DB]
    databases = wg_db[DOCUMENTS_COLLECTION].distinct("database_name")
    logger.info("GET /databases - Found %d databases", len(databases))
    return jsonify({"databases": databases})


@db_bp.get("/all-databases")
def list_all_databases():
    """List all databases in MongoDB."""
    logger.info("GET /all-databases - Listing all MongoDB databases")
    logger.info("GET /all-databases - Parameters: (none)")
    databases = mongo_client.list_database_names()
    logger.info("GET /all-databases - Found %d databases", len(databases))
    return jsonify({"databases": databases})


@db_bp.get("/collections")
def list_collections():
    """List collections with uploaded documents."""
    logger.info("GET /collections - Listing collections")
    database_name = request.args.get("database_name")
    logger.info("GET /collections - Parameters: database_name=%s", database_name)
    if not database_name:
        logger.warning("GET /collections - Missing required parameter: database_name")
        return jsonify({"error": "database_name is required"}), 400

    logger.info("GET /collections - Querying collections for database: %s", database_name)
    wg_db = mongo_client[WEB_GATHER_DB]
    collections = wg_db[DOCUMENTS_COLLECTION].distinct(
        "collection_name", {"database_name": database_name}
    )
    logger.info("GET /collections - Found %d collections", len(collections))
    return jsonify({"database_name": database_name, "collections": collections})


@db_bp.get("/all-collections")
def list_all_collections():
    """List all collections in a MongoDB database."""
    logger.info("GET /all-collections - Listing all MongoDB collections")
    database_name = request.args.get("database_name")
    logger.info("GET /all-collections - Parameters: database_name=%s", database_name)
    if not database_name:
        logger.warning("GET /all-collections - Missing required parameter: database_name")
        return jsonify({"error": "database_name is required"}), 400

    logger.info("GET /all-collections - Querying all collections for database: %s", database_name)
    collections = mongo_client[database_name].list_collection_names()
    logger.info("GET /all-collections - Found %d collections", len(collections))
    return jsonify({"database_name": database_name, "collections": collections})


@db_bp.get("/count-documents")
def count_documents():
    """Count documents in a MongoDB collection.

    Returns the number of documents in the specified database and collection.
    """
    logger.info("GET /count-documents - Counting documents")
    database_name = request.args.get("database_name")
    collection_name = request.args.get("collection_name")
    logger.info("GET /count-documents - Parameters: database_name=%s, collection_name=%s", database_name, collection_name)

    if not database_name or not collection_name:
        logger.warning("GET /count-documents - Missing required parameters")
        return jsonify({"error": "database_name and collection_name are required"}), 400

    logger.info(
        "GET /count-documents - Counting documents in %s.%s",
        database_name,
        collection_name,
    )
    db = mongo_client[database_name]
    count = db[collection_name].count_documents({})
    logger.info("GET /count-documents - Found %d documents", count)

    return jsonify({
        "database_name": database_name,
        "collection_name": collection_name,
        "count": count,
    })


@db_bp.post("/write_to_collection")
def write_to_collection():
    """Write a JSON document to a MongoDB collection.

    Supports two modes:
    - append (default): Insert a new document into the collection
    - replace: Update an existing document by _id (requires update_id)
    """
    logger.info("POST /write_to_collection - Writing document to collection")
    data = request.get_json()

    database_name = data.get("database_name")
    collection_name = data.get("collection_name")
    mode = data.get("mode", "append")
    document = data.get("document")
    update_id = data.get("update_id")
    logger.info("POST /write_to_collection - Parameters: database_name=%s, collection_name=%s, mode=%s, update_id=%s, document=%s",
                database_name, collection_name, mode, update_id, "provided" if document else None)

    if not database_name or not collection_name:
        logger.warning("POST /write_to_collection - Missing required parameters")
        return jsonify({"error": "database_name and collection_name are required"}), 400

    if not document:
        logger.warning("POST /write_to_collection - Missing document")
        return jsonify({"error": "document is required"}), 400

    if mode not in ("append", "replace"):
        logger.warning("POST /write_to_collection - Invalid mode: %s", mode)
        return jsonify({"error": "mode must be 'append' or 'replace'"}), 400

    if mode == "replace" and not update_id:
        logger.warning("POST /write_to_collection - replace mode requires update_id")
        return jsonify({"error": "update_id is required when mode is 'replace'"}), 400

    logger.info(
        "POST /write_to_collection - Writing to %s.%s (mode=%s)",
        database_name,
        collection_name,
        mode,
    )

    db = mongo_client[database_name]
    collection = db[collection_name]

    if mode == "append":
        result = collection.insert_one(document)
        logger.info("POST /write_to_collection - Inserted document with _id: %s", result.inserted_id)
        return jsonify({
            "database_name": database_name,
            "collection_name": collection_name,
            "mode": mode,
            "inserted_id": str(result.inserted_id),
            "message": "Document inserted successfully",
        })
    else:  # mode == "replace"
        try:
            object_id = ObjectId(update_id)
        except InvalidId:
            logger.warning("POST /write_to_collection - Invalid update_id: %s", update_id)
            return jsonify({"error": f"Invalid update_id: {update_id}"}), 400

        result = collection.replace_one({"_id": object_id}, document)

        if result.matched_count == 0:
            logger.warning("POST /write_to_collection - Document not found with _id: %s", update_id)
            return jsonify({"error": f"Document not found with _id: {update_id}"}), 404

        logger.info(
            "POST /write_to_collection - Replaced document with _id: %s (modified: %d)",
            update_id,
            result.modified_count,
        )
        return jsonify({
            "database_name": database_name,
            "collection_name": collection_name,
            "mode": mode,
            "update_id": update_id,
            "matched_count": result.matched_count,
            "modified_count": result.modified_count,
            "message": "Document replaced successfully",
        })


def _get_category_mapping_coll():
    """Return the category_mapping collection in privacy-compliance database."""
    return mongo_client[PRIVACY_COMPLIANCE_DB][CATEGORY_MAPPING_COLLECTION]


def _serialize_doc(doc):
    """Convert _id to string for JSON serialization. Mutates doc in place."""
    if "_id" in doc:
        doc["_id"] = str(doc["_id"])


@db_bp.get("/category-mapping")
def get_category_mapping():
    """List category mappings. Optionally filter by statute_category or sub_topic."""
    logger.info("GET /category-mapping - Listing category mappings")
    statute_category = request.args.get("statute_category")
    sub_topic = request.args.get("sub_topic")
    query_param = request.args.get("query")

    mongo_query = {}
    if statute_category:
        mongo_query["statute_category"] = statute_category
    if sub_topic:
        mongo_query["sub_topic"] = sub_topic
    if query_param:
        try:
            mongo_query = json.loads(query_param)
            if not isinstance(mongo_query, dict):
                return jsonify({"error": "query must be a JSON object"}), 400
        except json.JSONDecodeError as e:
            return jsonify({"error": f"Invalid JSON in query: {str(e)}"}), 400

    coll = _get_category_mapping_coll()
    docs = list(coll.find(mongo_query))
    for doc in docs:
        _serialize_doc(doc)
    logger.info("GET /category-mapping - Found %d documents", len(docs))
    return jsonify({"category_mappings": docs})


@db_bp.post("/category-mapping")
def post_category_mapping():
    """Create a new category mapping. Requires statute_category and policy_categories."""
    logger.info("POST /category-mapping - Creating category mapping")
    payload = request.get_json(silent=True) or {}
    statute_category = payload.get("statute_category")
    policy_categories = payload.get("policy_categories")
    sub_topic = payload.get("sub_topic")
    description = payload.get("description")

    if not statute_category:
        return jsonify({"error": "statute_category is required"}), 400
    if not policy_categories:
        return jsonify({"error": "policy_categories is required"}), 400
    if not isinstance(policy_categories, list):
        return jsonify({"error": "policy_categories must be an array"}), 400

    doc = {
        "statute_category": str(statute_category).strip(),
        "policy_categories": [str(c).strip() for c in policy_categories],
    }
    if sub_topic is not None:
        doc["sub_topic"] = str(sub_topic).strip()
    if description is not None:
        doc["description"] = str(description).strip()

    coll = _get_category_mapping_coll()
    result = coll.insert_one(doc)
    logger.info("POST /category-mapping - Inserted document with _id: %s", result.inserted_id)
    return jsonify({
        "database": PRIVACY_COMPLIANCE_DB,
        "collection": CATEGORY_MAPPING_COLLECTION,
        "inserted_id": str(result.inserted_id),
        "message": "Category mapping created successfully",
    }), 201


@db_bp.delete("/category-mapping")
def delete_category_mapping():
    """Delete category mappings. Use _id for single delete, or query for bulk delete."""
    logger.info("DELETE /category-mapping - Deleting category mappings")
    _id_param = request.args.get("_id")
    query_param = request.args.get("query")

    if _id_param and query_param:
        return jsonify({"error": "Provide either _id or query, not both"}), 400
    if not _id_param and not query_param:
        return jsonify({"error": "Provide _id or query to specify which documents to delete"}), 400

    mongo_query = {}
    if _id_param:
        try:
            mongo_query["_id"] = ObjectId(_id_param)
        except InvalidId:
            return jsonify({"error": f"Invalid _id: {_id_param}"}), 400
    else:
        try:
            mongo_query = json.loads(query_param)
            if not isinstance(mongo_query, dict):
                return jsonify({"error": "query must be a JSON object"}), 400
        except json.JSONDecodeError as e:
            return jsonify({"error": f"Invalid JSON in query: {str(e)}"}), 400

    coll = _get_category_mapping_coll()
    result = coll.delete_many(mongo_query)
    logger.info("DELETE /category-mapping - Deleted %d documents", result.deleted_count)
    return jsonify({
        "database": PRIVACY_COMPLIANCE_DB,
        "collection": CATEGORY_MAPPING_COLLECTION,
        "deleted_count": result.deleted_count,
        "message": "Category mapping(s) deleted successfully",
    })
