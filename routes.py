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
            "description": "Gather, load, index, and search web documents. The /ingest endpoint loads documents only - use /index separately to create vector embeddings.",
        },
        "paths": {
            "/gather": {
                "post": {
                    "summary": "Gather web documents based on query",
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
                    "responses": {"200": {"description": "Search results"}},
                }
            },
            "/ingest": {
                "post": {
                    "summary": "Load a document by crawling a URL or parsing a PDF",
                    "description": "Supports both web pages (crawled via Firecrawl) and PDF files (downloaded and parsed). PDF files are automatically detected by URL extension or Content-Type header. This endpoint only loads data - use /index to create vector embeddings.",
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
                                        "database": {"type": "string"},
                                        "collection": {"type": "string"},
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
                                            "overwritten": {"type": "boolean"},
                                            "previous_document_count": {"type": "integer"},
                                        },
                                    }
                                }
                            },
                        }
                    },
                }
            },
            "/documents": {
                "get": {
                    "summary": "List uploaded documents for a collection",
                    "parameters": [
                        {
                            "name": "database_name",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                        },
                        {
                            "name": "collection_name",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                        },
                    ],
                    "responses": {"200": {"description": "Documents list"}},
                }
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
            "/index": {
                "post": {
                    "summary": "Index a document by id",
                    "description": "Index a document from a source collection and write the indexed chunks to an index collection. The embedding model is determined by the index_collection_name.",
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
                                        "source_document_id": {
                                            "type": "string",
                                            "description": "The id of the document within the source collection",
                                        },
                                        "index_database_name": {
                                            "type": "string",
                                            "description": "The database of the index collection",
                                        },
                                        "index_collection_name": {
                                            "type": "string",
                                            "description": "The indexed data will be written to this collection",
                                        },
                                    },
                                    "required": [
                                        "source_database_name",
                                        "source_collection_name",
                                        "source_document_id",
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
                                            "source_document_id": {"type": "string"},
                                            "index_database_name": {"type": "string"},
                                            "index_collection_name": {"type": "string"},
                                            "chunks_indexed": {"type": "integer"},
                                            "embedding_model": {"type": "string"},
                                            "chunk_collection": {"type": "string"},
                                        },
                                    }
                                }
                            },
                        },
                        "400": {"description": "Missing required parameters or embedding model not configured"},
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
            "/parse_llm": {
                "get": {
                    "summary": "Parse document text using LLM",
                    "description": "Dynamically parse document text according to specific instructions using a language model. Returns structured JSON with parsed sections.",
                    "parameters": [
                        {
                            "name": "document",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                            "description": "The document text to parse",
                        },
                        {
                            "name": "parsing_prompt",
                            "in": "query",
                            "required": False,
                            "schema": {"type": "string"},
                            "description": "Instructions for how to parse the document. If not provided, defaults to parsing the document into logical sections based on document type.",
                        },
                        {
                            "name": "document_id",
                            "in": "query",
                            "required": False,
                            "schema": {"type": "string"},
                            "description": "Optional document identifier (defaults to doc_001)",
                        },
                    ],
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
                                                        "document_id": {"type": "string"},
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
                        "400": {"description": "Missing required parameters"},
                        "500": {"description": "LLM parsing failed"},
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
