"""Database-related functions and endpoints.

Includes functionality for endpoints: all-collections, all-databases, collections, count-documents, databases, documents, and write_to_collection.
"""
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

db_bp = Blueprint("db", __name__)


def init_db(client):
    """Initialize the database module with the MongoDB client."""
    global mongo_client
    mongo_client = client


def utc_now():
    """Return current UTC time."""
    return datetime.now(timezone.utc)


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
            if not isinstance(mongo_query, dict):
                logger.warning("GET /documents - Query must be a JSON object")
                return jsonify({"error": "query must be a JSON object"}), 400
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
