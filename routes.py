"""Endpoint routes and OpenAPI specification for the Web Gather API."""
from flask import Blueprint, Response, jsonify

routes_bp = Blueprint("routes", __name__)


def build_openapi_spec():
    """Build and return the OpenAPI specification."""
    return {
        "openapi": "3.0.3",
        "info": {
            "title": "Web Gather API",
            "version": "1.0.0",
            "description": (
                "Gather, load, index, and search web documents. The /ingest "
                "endpoint loads documents only - use /vector-index separately "
                "to create vector embeddings."
            ),
        },
        "components": {
            "schemas": {
                "ConfidenceThresholds": {
                    "type": "object",
                    "description": "Minimum confidence scores (0-1) for compliant and non_compliant determinations.",
                    "properties": {
                        "compliant": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "default": 0.75,
                            "description": "Lower bound for a compliant determination (0-1).",
                        },
                        "non_compliant": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "default": 0.75,
                            "description": "Lower bound for a non-compliant determination (0-1).",
                        },
                    },
                },
                "ComplianceOptions": {
                    "type": "object",
                    "description": "Request options for explainability and PII redaction.",
                    "properties": {
                        "explainability": {
                            "type": "boolean",
                            "default": True,
                            "description": "Include rationale and remediation suggestions in the response.",
                        },
                        "redact_pii": {
                            "type": "boolean",
                            "default": True,
                            "description": "Redact PII in returned text.",
                        },
                    },
                },
                "PolicyStatuteComplianceRequest": {
                    "type": "object",
                    "description": "Request body for policy statute compliance. Either policy_id or text must be provided.",
                    "properties": {
                        "database": {
                            "type": "string",
                            "description": "MongoDB database name containing the policy document.",
                        },
                        "policy_collection": {
                            "type": "string",
                            "description": "Collection name that holds the policy document.",
                        },
                        "policy_id": {
                            "type": "string",
                            "nullable": True,
                            "description": "Document ID of the policy in the database. Required if text is not provided.",
                        },
                        "text": {
                            "type": "string",
                            "nullable": True,
                            "description": "Raw policy text when not using a stored document. Required if policy_id is not provided.",
                        },
                        "jurisdiction": {
                            "type": "string",
                            "description": "Jurisdiction for statute comparison (e.g. GDPR, CCPA).",
                        },
                        "statute_corpus_id": {
                            "type": "string",
                            "nullable": True,
                            "description": "Identifier for which statute corpus to use.",
                        },
                        "top_k_statutes": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 50,
                            "description": "Maximum number of statute candidates to consider (1-50).",
                            "example": 50,
                        },
                        "confidence_thresholds": {
                            "$ref": "#/components/schemas/ConfidenceThresholds",
                            "description": "Optional thresholds for compliant/non-compliant confidence.",
                        },
                        "options": {
                            "$ref": "#/components/schemas/ComplianceOptions",
                            "description": "Optional request options (explainability, redact_pii).",
                        },
                    },
                    "required": ["database", "policy_collection", "jurisdiction"],
                },
                "AppliedStatute": {
                    "type": "object",
                    "properties": {
                        "statute_id": {"type": "string"},
                        "jurisdiction": {"type": "string"},
                        "title": {"type": "string"},
                        "section_id": {"type": "string"},
                        "matched_span": {"type": "string"},
                        "evidence_score": {"type": "number"},
                    },
                    "required": [
                        "statute_id",
                        "jurisdiction",
                        "title",
                        "section_id",
                        "matched_span",
                        "evidence_score",
                    ],
                },
                "PolicySectionResult": {
                    "type": "object",
                    "properties": {
                        "section_id": {"type": "string"},
                        "section_text": {"type": "string"},
                        "applied_statutes": {
                            "type": "array",
                            "items": {"$ref": "#/components/schemas/AppliedStatute"},
                        },
                        "compliance": {
                            "type": "string",
                            "enum": ["compliant", "non_compliant", "neither"],
                        },
                        "confidence": {"type": "number"},
                        "rationale": {"type": "string"},
                        "remediation_suggestions": {"type": "array", "items": {"type": "string"}},
                        "retrieval_trace": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": [
                        "section_id",
                        "section_text",
                        "applied_statutes",
                        "compliance",
                        "confidence",
                        "rationale",
                        "remediation_suggestions",
                        "retrieval_trace",
                    ],
                },
                "SummaryCounts": {
                    "type": "object",
                    "properties": {
                        "compliant": {"type": "integer"},
                        "non_compliant": {"type": "integer"},
                        "neither": {"type": "integer"},
                    },
                    "required": ["compliant", "non_compliant", "neither"],
                },
                "SummaryResult": {
                    "type": "object",
                    "properties": {
                        "overall_compliance": {
                            "type": "string",
                            "enum": ["compliant", "non_compliant", "mixed", "unknown"],
                        },
                        "counts": {"$ref": "#/components/schemas/SummaryCounts"},
                    },
                    "required": ["overall_compliance", "counts"],
                },
                "PolicyStatuteComplianceResponse": {
                    "type": "object",
                    "properties": {
                        "policy_id": {"type": "string", "nullable": True},
                        "jurisdiction": {"type": "string"},
                        "sections": {
                            "type": "array",
                            "items": {"$ref": "#/components/schemas/PolicySectionResult"},
                        },
                        "summary": {"$ref": "#/components/schemas/SummaryResult"},
                        "warnings": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["policy_id", "jurisdiction", "sections", "summary", "warnings"],
                },
                "ErrorResponse": {
                    "type": "object",
                    "properties": {"error": {"type": "string"}},
                },
            }
        },
        "paths": {
            "/gather": {
                "post": {
                    "summary": "Gather web documents based on query",
                    "description": "Search the web for documents matching the query. Returns search results with URLs that can be selected for crawling via the /ingest endpoint. Each result includes a relevance score (0-1) and percent_match (0-100) indicating how well the content matches the query for AI consumption. The response includes a next_step object describing the required and optional parameters for ingestion.",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {"query": {"type": "string"}},
                                    "required": ["query"],
                                }
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Search results with next step instructions",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "query": {"type": "string", "description": "The search query"},
                                            "results": {
                                                "type": "array",
                                                "description": "List of search results sorted by relevance",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "url": {"type": "string"},
                                                        "title": {"type": "string"},
                                                        "description": {"type": "string"},
                                                        "score": {
                                                            "type": "number",
                                                            "description": "Relevance score from Firecrawl search (0-1)",
                                                            "minimum": 0,
                                                            "maximum": 1,
                                                        },
                                                        "percent_match": {
                                                            "type": "number",
                                                            "description": "Relevance score as percentage (0-100)",
                                                            "minimum": 0,
                                                            "maximum": 100,
                                                        },
                                                    },
                                                },
                                            },
                                            "next_step": {
                                                "type": "object",
                                                "description": "Instructions for the next step: selecting a URL and calling /ingest",
                                                "properties": {
                                                    "action": {"type": "string"},
                                                    "endpoint": {"type": "string"},
                                                    "method": {"type": "string"},
                                                    "required_parameters": {
                                                        "type": "object",
                                                        "description": "Parameters that must be provided to /ingest",
                                                        "properties": {
                                                            "url": {"type": "object"},
                                                            "database": {"type": "object"},
                                                            "collection": {"type": "object"},
                                                        },
                                                    },
                                                    "optional_parameters": {
                                                        "type": "object",
                                                        "description": "Optional parameters for /ingest (depth, breadth, mode, index_database, index_collection)",
                                                    },
                                                },
                                            },
                                        },
                                    }
                                }
                            },
                        }
                    },
                }
            },
            "/ingest": {
                "post": {
                    "summary": "Load a document by crawling a URL or parsing a PDF",
                    "description": (
                        "Supports both web pages (crawled via Firecrawl) and PDF "
                        "files (downloaded and parsed). PDF files are "
                        "automatically detected by URL extension or Content-Type "
                        "header. Use /vector-index endpoint separately to "
                        "create vector embeddings."
                    ),
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "url": {"type": "string", "description": "URL to a web page or PDF file"},
                                        "depth": {"type": "integer", "description": "Crawl depth for web pages (ignored for PDFs)"},
                                        "breadth": {"type": "integer", "description": "Max pages to crawl for web pages (ignored for PDFs)"},
                                        "database": {"type": "string", "description": "Database name for storing raw crawled data"},
                                        "collection": {"type": "string", "description": "Collection name for storing raw crawled data"},
                                        "mode": {"type": "string", "enum": ["append", "overwrite"], "description": "How to handle existing data: 'append' adds to existing data (default), 'overwrite' clears existing data first"},
                                    },
                                    "required": ["url", "database", "collection"],
                                }
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Ingestion result",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "document_id": {"type": "string"},
                                            "document_type": {"type": "string", "enum": ["web", "pdf"]},
                                            "pages": {"type": "integer"},
                                            "database_name": {"type": "string"},
                                            "collection_name": {"type": "string"},
                                            "mode": {"type": "string"},
                                            "message": {"type": "string"},
                                            "overwritten": {"type": "boolean"},
                                            "previous_document_count": {"type": "integer"},
                                        },
                                    }
                                }
                            },
                        },
                        "400": {
                            "description": "Bad request - missing required parameters or invalid mode",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "error": {"type": "string"},
                                        },
                                    }
                                }
                            },
                        },
                    },
                }
            },
            "/documents": {
                "get": {
                    "summary": "List uploaded documents for a collection",
                    "description": "Retrieve documents from a MongoDB collection. Optionally filter results using a MongoDB query.",
                    "parameters": [
                        {
                            "name": "database_name",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                            "description": "The name of the database",
                        },
                        {
                            "name": "collection_name",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                            "description": "The name of the collection",
                        },
                        {
                            "name": "query",
                            "in": "query",
                            "required": False,
                            "schema": {"type": "string"},
                            "description": "MongoDB query as JSON string to filter documents (e.g., '{\"status\": \"active\"}' or '{\"age\": {\"$gt\": 25}}')",
                        },
                    ],
                    "responses": {
                        "200": {
                            "description": "Documents list",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "documents": {
                                                "type": "array",
                                                "items": {"type": "object"},
                                                "description": "List of documents matching the query",
                                            }
                                        },
                                    }
                                }
                            },
                        },
                        "400": {
                            "description": "Bad request - missing required parameters or invalid query JSON",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "error": {"type": "string"},
                                        },
                                    }
                                }
                            },
                        },
                    },
                },
                "delete": {
                    "summary": "Delete documents from a collection",
                    "description": "Delete documents from a MongoDB collection. Optionally filter deletions using a MongoDB query.",
                    "parameters": [
                        {
                            "name": "database_name",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                            "description": "The name of the database",
                        },
                        {
                            "name": "collection_name",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                            "description": "The name of the collection",
                        },
                        {
                            "name": "query",
                            "in": "query",
                            "required": False,
                            "schema": {"type": "string"},
                            "description": "MongoDB query as JSON string to filter documents (e.g., '{\"status\": \"inactive\"}' or '{\"age\": {\"$lt\": 18}}')",
                        },
                    ],
                    "responses": {
                        "200": {
                            "description": "Deletion result",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "database_name": {"type": "string"},
                                            "collection_name": {"type": "string"},
                                            "deleted_count": {
                                                "type": "integer",
                                                "description": "Number of documents deleted",
                                            },
                                            "message": {"type": "string"},
                                        },
                                    }
                                }
                            },
                        },
                        "400": {
                            "description": "Bad request - missing required parameters or invalid query JSON",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "error": {"type": "string"},
                                        },
                                    }
                                }
                            },
                        },
                    },
                },
            },
            "/databases": {
                "get": {
                    "summary": "List all databases with uploaded documents",
                    "responses": {
                        "200": {
                            "description": "Databases list",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "databases": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            }
                                        },
                                    }
                                }
                            },
                        }
                    },
                }
            },
            "/collections": {
                "get": {
                    "summary": "List collections with uploaded documents",
                    "parameters": [
                        {
                            "name": "database_name",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                        }
                    ],
                    "responses": {"200": {"description": "Collections list"}},
                }
            },
            "/all-databases": {
                "get": {
                    "summary": "List all databases in MongoDB",
                    "description": "Returns all databases in the MongoDB instance, not just those with uploaded documents",
                    "responses": {
                        "200": {
                            "description": "All databases list",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "databases": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            }
                                        },
                                    }
                                }
                            },
                        }
                    },
                }
            },
            "/all-collections": {
                "get": {
                    "summary": "List all collections in a MongoDB database",
                    "description": "Returns all collections in the specified database, not just those tracked in documents",
                    "parameters": [
                        {
                            "name": "database_name",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "All collections list",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "database_name": {"type": "string"},
                                            "collections": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            }
                                        },
                                    }
                                }
                            },
                        }
                    },
                }
            },
            "/count-documents": {
                "get": {
                    "summary": "Count documents in a MongoDB collection",
                    "description": "Returns the number of documents in the specified database and collection",
                    "parameters": [
                        {
                            "name": "database_name",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                            "description": "The name of the database",
                        },
                        {
                            "name": "collection_name",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                            "description": "The name of the collection",
                        },
                    ],
                    "responses": {
                        "200": {
                            "description": "Document count",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "database_name": {"type": "string"},
                                            "collection_name": {"type": "string"},
                                            "count": {
                                                "type": "integer",
                                                "description": "The number of documents in the collection",
                                            },
                                        },
                                    }
                                }
                            },
                        },
                        "400": {
                            "description": "Bad request - missing required parameters",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "error": {"type": "string"},
                                        },
                                    }
                                }
                            },
                        },
                    },
                }
            },
            "/write_to_collection": {
                "post": {
                    "summary": "Write a JSON document to a MongoDB collection",
                    "description": "Insert a new document (append mode) or replace an existing document by _id (replace mode). When using replace mode, the update_id parameter is required to specify which document to update.",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "database_name": {
                                            "type": "string",
                                            "description": "The name of the database",
                                        },
                                        "collection_name": {
                                            "type": "string",
                                            "description": "The name of the collection",
                                        },
                                        "mode": {
                                            "type": "string",
                                            "enum": ["append", "replace"],
                                            "default": "append",
                                            "description": "Write mode: 'append' inserts a new document (default), 'replace' updates an existing document by _id",
                                        },
                                        "document": {
                                            "type": "object",
                                            "description": "The JSON document to write to the collection",
                                        },
                                        "update_id": {
                                            "type": "string",
                                            "description": "The _id of the document to update (required when mode is 'replace')",
                                        },
                                    },
                                    "required": ["database_name", "collection_name", "document"],
                                }
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Document written successfully",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "database_name": {"type": "string"},
                                            "collection_name": {"type": "string"},
                                            "mode": {"type": "string"},
                                            "inserted_id": {
                                                "type": "string",
                                                "description": "The _id of the inserted document (append mode only)",
                                            },
                                            "update_id": {
                                                "type": "string",
                                                "description": "The _id of the updated document (replace mode only)",
                                            },
                                            "matched_count": {
                                                "type": "integer",
                                                "description": "Number of documents matched (replace mode only)",
                                            },
                                            "modified_count": {
                                                "type": "integer",
                                                "description": "Number of documents modified (replace mode only)",
                                            },
                                            "message": {"type": "string"},
                                        },
                                    }
                                }
                            },
                        },
                        "400": {
                            "description": "Bad request - missing required parameters, invalid mode, or invalid update_id",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "error": {"type": "string"},
                                        },
                                    }
                                }
                            },
                        },
                        "404": {
                            "description": "Document not found (replace mode only)",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "error": {"type": "string"},
                                        },
                                    }
                                }
                            },
                        },
                    },
                }
            },
            "/vector-index": {
                "post": {
                    "summary": "Index collection rows by chunk_text",
                    "description": "Index all rows in a source collection by embedding the chunk_text field and writing the results to an index collection. The embedding model is determined by the index_database_name.",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "source_database_name": {
                                            "type": "string",
                                            "description": "The name of the database containing the collection to be indexed",
                                        },
                                        "source_collection_name": {
                                            "type": "string",
                                            "description": "The name of the collection containing the data to be indexed",
                                        },
                                        "index_database_name": {
                                            "type": "string",
                                            "description": "The database of the index collection",
                                        },
                                        "index_collection_name": {
                                            "type": "string",
                                            "description": "The indexed data will be written to this collection",
                                        },
                                        "source_query": {
                                            "type": "string",
                                            "description": (
                                                "Optional Mongo query as JSON text to filter source rows"
                                            ),
                                        },
                                    },
                                    "required": [
                                        "source_database_name",
                                        "source_collection_name",
                                        "index_database_name",
                                        "index_collection_name",
                                    ],
                                }
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Indexing result",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "source_database_name": {"type": "string"},
                                            "source_collection_name": {"type": "string"},
                                            "index_database_name": {"type": "string"},
                                            "index_collection_name": {"type": "string"},
                                            "chunks_indexed": {"type": "integer"},
                                            "embedding_model": {"type": "string"},
                                            "skipped_rows": {"type": "integer"},
                                        },
                                    }
                                }
                            },
                        },
                        "400": {"description": "Missing required parameters, invalid parameters, or embedding model not configured"},
                        "404": {"description": "Document not found"},
                    },
                }
            },
            "/search": {
                "get": {
                    "summary": "Search vector-indexed collection",
                    "parameters": [
                        {
                            "name": "document_id",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                        },
                        {
                            "name": "query",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                        },
                    ],
                    "responses": {"200": {"description": "Search results"}},
                }
            },
            "/embedding-models": {
                "get": {
                    "summary": "List embedding models",
                    "parameters": [
                        {
                            "name": "database_name",
                            "in": "query",
                            "required": False,
                            "schema": {"type": "string"},
                        }
                    ],
                    "responses": {"200": {"description": "Embedding models"}},
                },
                "post": {
                    "summary": "Add embedding model",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "database_name": {"type": "string"},
                                        "model_name": {"type": "string"},
                                    },
                                    "required": ["database_name", "model_name"],
                                }
                            }
                        },
                    },
                    "responses": {"200": {"description": "Embedding model saved"}},
                },
            },
            "/parse-llm": {
                "post": {
                    "summary": "Parse document text from a collection using LLM",
                    "description": "Reads records from the specified database/collection and uses this as input for the LLM along with the parse_prompt. If document_id is provided, only that specific document is parsed; otherwise, all documents in the collection are concatenated. Returns structured JSON with parsed sections.",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "database": {
                                            "type": "string",
                                            "description": "The name of the database containing the documents to parse",
                                        },
                                        "collection": {
                                            "type": "string",
                                            "description": "The name of the collection containing the documents to parse",
                                        },
                                        "parse_prompt": {
                                            "type": "string",
                                            "description": "Instructions for how to parse the combined document text",
                                        },
                                        "document_id": {
                                            "type": "string",
                                            "description": "Optional MongoDB ObjectId of a specific document to parse. If not provided, all documents in the collection are concatenated for parsing.",
                                        },
                                    },
                                    "required": ["database", "collection", "parse_prompt"],
                                }
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Parsed document sections",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "parsed_doc": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "section": {"type": "string"},
                                                        "code_name": {"type": "string"},
                                                        "jurisdiction": {"type": "string"},
                                                        "parsed_header_text": {"type": "string"},
                                                        "parsed_text": {"type": "string"},
                                                    },
                                                },
                                            }
                                        },
                                    }
                                }
                            },
                        },
                        "400": {"description": "Missing required parameters or no documents found"},
                        "500": {"description": "LLM parsing failed"},
                    },
                }
            },
            "/policy-statute-compliance": {
                "post": {
                    "summary": "Compare policy sections to statutes for compliance",
                    "description": "Segments a policy into sections, retrieves relevant statutes, and returns compliance determinations with evidence.",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/PolicyStatuteComplianceRequest"},
                                "example": {
                                    "database": "string",
                                    "policy_collection": "string",
                                    "policy_id": "string",
                                    "text": "string",
                                    "jurisdiction": "string",
                                    "statute_corpus_id": "string",
                                    "top_k_statutes": 50,
                                    "confidence_thresholds": {
                                        "compliant": 1,
                                        "non_compliant": 1,
                                    },
                                    "options": {
                                        "explainability": True,
                                        "redact_pii": True,
                                    },
                                },
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Compliance analysis response",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/PolicyStatuteComplianceResponse"}
                                }
                            },
                        },
                        "400": {
                            "description": "Bad request",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            },
                        },
                        "401": {
                            "description": "Missing API key",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            },
                        },
                        "403": {
                            "description": "Unauthorized",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            },
                        },
                        "422": {
                            "description": "Validation error",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            },
                        },
                        "500": {
                            "description": "Internal server error",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            },
                        },
                    },
                }
            },
            "/crawl": {
                "post": {
                    "summary": "Crawl a URL with specified depth and breadth",
                    "description": "Crawl a URL and return combined text from all pages visited. Use depth to control how many levels of links to follow, and breadth to limit the maximum number of pages crawled.",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "url": {
                                            "type": "string",
                                            "description": "The URL to start crawling from",
                                        },
                                        "depth": {
                                            "type": "integer",
                                            "description": "How deep to follow links from the starting URL (default: 2)",
                                            "default": 2,
                                            "minimum": 1,
                                        },
                                        "breadth": {
                                            "type": "integer",
                                            "description": "Maximum number of pages to crawl (default: 10)",
                                            "default": 10,
                                            "minimum": 1,
                                        },
                                    },
                                    "required": ["url"],
                                }
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Crawl results with combined page text",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "url": {
                                                "type": "string",
                                                "description": "The starting URL that was crawled",
                                            },
                                            "depth": {
                                                "type": "integer",
                                                "description": "The depth parameter used",
                                            },
                                            "breadth": {
                                                "type": "integer",
                                                "description": "The breadth parameter used",
                                            },
                                            "pages_crawled": {
                                                "type": "integer",
                                                "description": "Number of pages successfully crawled",
                                            },
                                            "urls_crawled": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                                "description": "List of URLs that were crawled",
                                            },
                                            "combined_text": {
                                                "type": "string",
                                                "description": "Combined text content from all crawled pages",
                                            },
                                            "text_length": {
                                                "type": "integer",
                                                "description": "Length of the combined text in characters",
                                            },
                                        },
                                    }
                                }
                            },
                        },
                        "400": {
                            "description": "Bad request - missing URL, invalid parameters, or no content returned",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "error": {"type": "string"},
                                        },
                                    }
                                }
                            },
                        },
                        "500": {
                            "description": "Crawl failed",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "error": {"type": "string"},
                                        },
                                    }
                                }
                            },
                        },
                    },
                }
            },
        },
    }


@routes_bp.get("/openapi.json")
def openapi():
    """Return the OpenAPI specification."""
    return jsonify(build_openapi_spec())


@routes_bp.get("/docs")
def docs():
    """Render the Swagger UI documentation."""
    html = """
    <!doctype html>
    <html>
      <head>
        <title>Web Gather API Docs</title>
        <link
          rel="stylesheet"
          href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css"
        />
      </head>
      <body>
        <div id="swagger-ui"></div>
        <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
        <script>
          window.onload = () => {
            SwaggerUIBundle({
              url: "/openapi.json",
              dom_id: "#swagger-ui"
            });
          };
        </script>
      </body>
    </html>
    """
    return Response(html, mimetype="text/html")
