"""Core endpoints for the Web Gather API.

Includes endpoints: gather, ingest, parse-llm, search, and index.
"""
import io
import json
import logging
import uuid
from urllib.parse import urlparse

import numpy as np
import requests
from flask import Blueprint, jsonify, request
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from db import (
    WEB_GATHER_DB,
    DOCUMENTS_COLLECTION,
    get_embedding_model_name,
    utc_now,
)

logger = logging.getLogger("web-gather-api")

# Will be initialized by app.py
mongo_client = None
firecrawl_client = None
anthropic_client = None

# Model cache for sentence transformers
_model_cache = {}

core_bp = Blueprint("core", __name__)


def init_core(mongo, firecrawl, anthropic):
    """Initialize the core module with required clients."""
    global mongo_client, firecrawl_client, anthropic_client
    mongo_client = mongo
    firecrawl_client = firecrawl
    anthropic_client = anthropic


def get_model(model_name: str) -> SentenceTransformer:
    """Get or create a cached sentence transformer model."""
    model = _model_cache.get(model_name)
    if model is None:
        model = SentenceTransformer(model_name)
        _model_cache[model_name] = model
    return model


def normalize_results(raw_result):
    """Normalize Firecrawl results to a list format."""
    logger.info("normalize_results: input type: %s", type(raw_result))

    # Handle Firecrawl CrawlJob response (has .data attribute)
    if hasattr(raw_result, "data") and isinstance(raw_result.data, list):
        logger.info("normalize_results: returning .data list with %d items", len(raw_result.data))
        return raw_result.data
    # Handle Firecrawl SearchData response (has .web attribute)
    if hasattr(raw_result, "web") and raw_result.web is not None:
        logger.info("normalize_results: returning .web with %d items", len(raw_result.web))
        return raw_result.web
    # Handle dict responses (legacy compatibility)
    if isinstance(raw_result, dict):
        logger.info("normalize_results: dict keys: %s", list(raw_result.keys()))
        if "data" in raw_result:
            logger.info("normalize_results: returning dict['data'] with %d items", len(raw_result["data"]))
            return raw_result["data"]
        if "results" in raw_result:
            logger.info("normalize_results: returning dict['results']")
            return raw_result["results"]
        if "pages" in raw_result:
            logger.info("normalize_results: returning dict['pages']")
            return raw_result["pages"]
    if isinstance(raw_result, list):
        logger.info("normalize_results: returning list with %d items", len(raw_result))
        return raw_result
    logger.warning("normalize_results: no matching pattern, returning empty list")
    return []


def get_page_attr(page, attr, default=""):
    """Get attribute from page object or dict."""
    if hasattr(page, attr):
        return getattr(page, attr, default) or default
    if isinstance(page, dict):
        return page.get(attr, default) or default
    return default


def combine_pages(pages):
    """Combine multiple pages into a single text."""
    logger.info("combine_pages: Processing %d pages", len(pages))
    combined = []
    for idx, page in enumerate(pages):
        logger.info("combine_pages: Page %d type: %s", idx, type(page))

        # Debug: log all available attributes
        if hasattr(page, '__dict__'):
            logger.info("combine_pages: Page %d attrs: %s", idx, list(vars(page).keys()))
        elif isinstance(page, dict):
            logger.info("combine_pages: Page %d dict keys: %s", idx, list(page.keys()))

        # Handle Firecrawl Document objects (url/title in metadata)
        if hasattr(page, "metadata") and page.metadata:
            url = getattr(page.metadata, "url", "") or ""
            title = getattr(page.metadata, "title", "") or ""
            logger.info("combine_pages: Page %d metadata - url: %s, title: %s", idx, url[:50] if url else "None", title[:50] if title else "None")
        else:
            url = get_page_attr(page, "url")
            title = get_page_attr(page, "title")
            logger.info("combine_pages: Page %d direct - url: %s, title: %s", idx, url[:50] if url else "None", title[:50] if title else "None")

        # Get content - try markdown first (Firecrawl), then fallbacks
        # Note: Firecrawl Document fields are: markdown, html, raw_html, summary
        markdown_content = get_page_attr(page, "markdown")
        html_content = get_page_attr(page, "html")
        raw_html_content = get_page_attr(page, "raw_html")
        summary_content = get_page_attr(page, "summary")

        logger.info("combine_pages: Page %d content lengths - markdown: %d, html: %d, raw_html: %d, summary: %d",
                   idx,
                   len(markdown_content) if markdown_content else 0,
                   len(html_content) if html_content else 0,
                   len(raw_html_content) if raw_html_content else 0,
                   len(summary_content) if summary_content else 0)

        content = markdown_content or html_content or raw_html_content or summary_content
        if not content:
            logger.warning("combine_pages: Page %d has no extractable content, skipping", idx)
            continue
        header = "Source"
        if title:
            header = f"{title} | Source"
        combined.append(f"{header}: {url}\n{content}".strip())
        logger.info("combine_pages: Page %d added with content length: %d", idx, len(content))

    result = "\n\n".join(combined).strip()
    logger.info("combine_pages: Final combined text length: %d", len(result))
    return result


def is_pdf_url(url: str) -> bool:
    """Check if a URL points to a PDF file."""
    parsed = urlparse(url)
    path_lower = parsed.path.lower()
    # Check file extension
    if path_lower.endswith(".pdf"):
        return True
    # Check common PDF query patterns
    if "pdf" in parsed.query.lower():
        return True
    return False


def detect_content_type(url: str) -> str:
    """Make a HEAD request to detect the content type of a URL."""
    try:
        logger.info("Detecting content type for URL: %s", url)
        response = requests.head(url, allow_redirects=True, timeout=10)
        content_type = response.headers.get("Content-Type", "").lower()
        logger.info("Content-Type detected: %s", content_type)
        return content_type
    except Exception as exc:
        logger.warning("Failed to detect content type for %s: %s", url, exc)
        return ""


def download_pdf(url: str) -> bytes:
    """Download a PDF file from a URL."""
    logger.info("Downloading PDF from: %s", url)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    content_length = len(response.content)
    logger.info("Downloaded PDF: %d bytes", content_length)
    return response.content


def extract_text_from_pdf(pdf_content: bytes) -> list[dict]:
    """
    Extract text from PDF content.
    Returns a list of page dictionaries with 'page_number' and 'text' keys.
    """
    logger.info("Extracting text from PDF (%d bytes)", len(pdf_content))
    pages = []

    try:
        pdf_file = io.BytesIO(pdf_content)
        reader = PdfReader(pdf_file)
        total_pages = len(reader.pages)
        logger.info("PDF has %d pages", total_pages)

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append({
                    "page_number": page_num,
                    "text": text,
                })
                logger.info("Extracted %d characters from page %d", len(text), page_num)
            else:
                logger.warning("Page %d has no extractable text", page_num)

        logger.info("Successfully extracted text from %d/%d pages", len(pages), total_pages)
    except Exception as exc:
        logger.error("Failed to extract text from PDF: %s", exc)
        raise

    return pages


def combine_pdf_pages(pages: list[dict], source_url: str) -> str:
    """Combine PDF page texts into a single document string."""
    combined = []
    for page in pages:
        page_num = page.get("page_number", "?")
        text = page.get("text", "")
        if text:
            header = f"Page {page_num} | Source: {source_url}"
            combined.append(f"{header}\n{text}")
    return "\n\n".join(combined).strip()


def chunk_text(text, chunk_size=1200, overlap=200):
    """Split text into overlapping chunks."""
    if not text:
        return []
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(text_len, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == text_len:
            break
        start = max(0, end - overlap)
    return chunks


def cosine_similarity(query_embedding, chunk_embeddings):
    """Compute cosine similarity between query and chunk embeddings."""
    if not chunk_embeddings:
        return np.array([])
    chunk_matrix = np.vstack(chunk_embeddings)
    return chunk_matrix @ query_embedding


def index_document_chunks(
    document_id: str,
    text: str,
    database_name: str,
    collection_name: str,
    index_database_name: str = None,
    index_collection_name: str = None,
):
    """
    Create vector-indexed chunks for a document.

    Args:
        document_id: The document identifier
        text: The text content to index
        database_name: The source database name (used for metadata updates)
        collection_name: The source collection name (used for metadata updates)
        index_database_name: The database where chunks will be stored (defaults to database_name)
        index_collection_name: The collection name for chunks (defaults to collection_name)

    Returns dict with indexing results or None if embedding model not configured.
    """
    # Use source values as defaults if index values not provided
    if index_database_name is None:
        index_database_name = database_name
    if index_collection_name is None:
        index_collection_name = collection_name

    logger.info("Starting indexing for document: %s", document_id)
    logger.info("Source: %s.%s, Index target: %s.%s",
                database_name, collection_name, index_database_name, index_collection_name)

    # Look up embedding model using the index database name
    model_name = get_embedding_model_name(index_database_name)
    if not model_name:
        logger.info("Skipping indexing - no embedding model configured for database: %s", index_database_name)
        return None

    logger.info("Chunking document text (length: %d characters)", len(text))
    chunks = chunk_text(text)
    if not chunks:
        logger.warning("Document %s has no content to chunk", document_id)
        return None

    logger.info("Created %d chunks from document %s", len(chunks), document_id)

    logger.info("Loading embedding model: %s", model_name)
    model = get_model(model_name)

    logger.info("Generating embeddings for %d chunks", len(chunks))
    embeddings = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    logger.info("Embeddings generated successfully (dimension: %d)", embeddings.shape[1])

    # Write chunks to the index database/collection (use collection name verbatim)
    index_db = mongo_client[index_database_name]
    chunk_collection = index_collection_name

    logger.info("Clearing existing chunks for document %s in %s.%s",
                document_id, index_database_name, chunk_collection)
    delete_result = index_db[chunk_collection].delete_many({"document_id": document_id})
    logger.info("Deleted %d existing chunks", delete_result.deleted_count)

    chunk_docs = []
    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        chunk_docs.append(
            {
                "document_id": document_id,
                "chunk_index": idx,
                "text": chunk,
                "embedding": embedding.tolist(),
                "created_at": utc_now(),
            }
        )

    logger.info("Inserting %d chunks into %s.%s", len(chunk_docs), index_database_name, chunk_collection)
    index_db[chunk_collection].insert_many(chunk_docs)

    wg_db = mongo_client[WEB_GATHER_DB]
    wg_db[DOCUMENTS_COLLECTION].update_one(
        {"document_id": document_id},
        {
            "$set": {
                "indexed_at": utc_now(),
                "chunk_count": len(chunk_docs),
                "embedding_model": model_name,
                "chunk_collection": chunk_collection,
                "index_database_name": index_database_name,
                "index_collection_name": index_collection_name,
            }
        },
    )

    logger.info("Indexing complete for document %s: %d chunks indexed with model %s to %s.%s",
                document_id, len(chunk_docs), model_name, index_database_name, chunk_collection)

    return {
        "chunks_indexed": len(chunk_docs),
        "embedding_model": model_name,
        "chunk_collection": chunk_collection,
    }


def serialize_search_result(result):
    """Convert Firecrawl search result object to dict."""
    if isinstance(result, dict):
        return result
    # Handle SearchResultWeb or Document objects
    data = {}
    if hasattr(result, "url"):
        data["url"] = result.url
    if hasattr(result, "title"):
        data["title"] = result.title
    if hasattr(result, "description"):
        data["description"] = result.description
    if hasattr(result, "markdown"):
        data["markdown"] = result.markdown
    if hasattr(result, "metadata") and result.metadata:
        data["url"] = getattr(result.metadata, "url", None) or data.get("url")
        data["title"] = getattr(result.metadata, "title", None) or data.get("title")
    return data


@core_bp.post("/gather")
def gather():
    """Gather web documents based on query."""
    logger.info("POST /gather - Starting web search")
    payload = request.get_json(silent=True) or {}
    query = payload.get("query")
    if not query:
        logger.warning("POST /gather - Missing required parameter: query")
        return jsonify({"error": "query is required"}), 400

    logger.info("POST /gather - Searching for: %s", query)
    results = firecrawl_client.search(
        query=query,
        limit=10,
    )
    normalized = normalize_results(results)
    serialized = [serialize_search_result(r) for r in normalized]
    logger.info("POST /gather - Found %d results for query: %s", len(serialized), query)

    # Build next_step information for the gather->ingest flow
    next_step = {
        "action": "Select a URL to crawl and call POST /ingest",
        "endpoint": "/ingest",
        "method": "POST",
        "required_parameters": {
            "url": {
                "type": "string",
                "description": "Select a URL from the results above to crawl",
            },
            "database": {
                "type": "string",
                "description": "Database name for storing raw crawled data",
            },
            "collection": {
                "type": "string",
                "description": "Collection name for storing raw crawled data",
            },
        },
        "optional_parameters": {
            "depth": {
                "type": "integer",
                "default": 1,
                "description": "Crawl depth for web pages",
            },
            "breadth": {
                "type": "integer",
                "default": 5,
                "description": "Max pages to crawl for web pages",
            },
            "mode": {
                "type": "string",
                "enum": ["append", "overwrite"],
                "default": "append",
                "description": "How to handle existing data: 'append' adds to existing data, 'overwrite' clears existing data first",
            },
            "index_database": {
                "type": "string",
                "description": "Database name for storing chunked/vectorized data (enables automatic indexing)",
            },
            "index_collection": {
                "type": "string",
                "description": "Collection name for storing chunked/vectorized data (enables automatic indexing)",
            },
        },
    }

    return jsonify({
        "query": query,
        "results": serialized,
        "next_step": next_step,
    })


@core_bp.post("/ingest")
def ingest():
    """Load a document by crawling a URL or parsing a PDF."""
    logger.info("POST /ingest - Starting document ingestion (load only)")
    payload = request.get_json(silent=True) or {}
    url = payload.get("url")
    depth = int(payload.get("depth", 1))
    breadth = int(payload.get("breadth", 5))
    database_name = payload.get("database")
    collection_name = payload.get("collection")
    mode = payload.get("mode", "append").lower()

    if not url or not database_name or not collection_name:
        logger.warning("POST /ingest - Missing required parameters")
        return (
            jsonify(
                {
                    "error": "url, database, and collection are required",
                }
            ),
            400,
        )

    # Validate mode parameter
    if mode not in ("append", "overwrite"):
        logger.warning("POST /ingest - Invalid mode: %s", mode)
        return (
            jsonify(
                {
                    "error": "mode must be 'append' or 'overwrite'",
                }
            ),
            400,
        )

    # Detect if URL is a PDF file
    is_pdf = is_pdf_url(url)
    if not is_pdf:
        # Check content type via HEAD request
        content_type = detect_content_type(url)
        is_pdf = "application/pdf" in content_type

    page_count = 0
    document_type = "web"

    if is_pdf:
        # Handle PDF file
        logger.info("POST /ingest - Detected PDF file, downloading and extracting text")
        document_type = "pdf"
        try:
            pdf_content = download_pdf(url)
            pdf_pages = extract_text_from_pdf(pdf_content)
            page_count = len(pdf_pages)

            if not pdf_pages:
                logger.warning("POST /ingest - PDF has no extractable text: %s", url)
                return jsonify({"error": "PDF has no extractable text"}), 400

            combined_text = combine_pdf_pages(pdf_pages, url)
            logger.info("POST /ingest - Extracted text from %d PDF pages", page_count)

        except requests.RequestException as exc:
            logger.error("POST /ingest - Failed to download PDF %s: %s", url, exc)
            return jsonify({"error": f"Failed to download PDF: {exc}"}), 500
        except Exception as exc:
            logger.error("POST /ingest - Failed to parse PDF %s: %s", url, exc)
            return jsonify({"error": f"Failed to parse PDF: {exc}"}), 500
    else:
        # Handle web page via Firecrawl
        logger.info("POST /ingest - Crawling web URL: %s (depth=%d, breadth=%d)", url, depth, breadth)
        try:
            crawl_result = firecrawl_client.crawl(
                url=url,
                limit=breadth,
            )
        except Exception as exc:
            logger.error("POST /ingest - Crawl failed for URL %s: %s", url, exc)
            return jsonify({"error": f"crawl failed: {exc}"}), 500

        # Debug: log crawl result structure
        logger.info("POST /ingest - crawl_result type: %s", type(crawl_result))
        if hasattr(crawl_result, '__dict__'):
            logger.info("POST /ingest - crawl_result attrs: %s", list(vars(crawl_result).keys()))
        if hasattr(crawl_result, 'data'):
            logger.info("POST /ingest - crawl_result.data: type=%s, len=%s",
                       type(crawl_result.data), len(crawl_result.data) if crawl_result.data else 0)
        if hasattr(crawl_result, 'status'):
            logger.info("POST /ingest - crawl_result.status: %s", crawl_result.status)

        pages = normalize_results(crawl_result)
        page_count = len(pages)
        logger.info("POST /ingest - Crawl returned %d pages", page_count)

        # Check if crawl returned no pages at all
        if page_count == 0:
            logger.error("POST /ingest - Crawl returned 0 pages for URL: %s", url)
            return jsonify({"error": "crawl returned no pages - website may be blocking crawlers"}), 400

        # Log any warnings from crawled pages
        for idx, page in enumerate(pages):
            if hasattr(page, 'warning') and page.warning:
                logger.warning("POST /ingest - Page %d warning: %s", idx, page.warning)
            if hasattr(page, 'metadata') and page.metadata:
                status_code = getattr(page.metadata, 'statusCode', None)
                if status_code and status_code != 200:
                    logger.warning("POST /ingest - Page %d statusCode: %s", idx, status_code)

        combined_text = combine_pages(pages)
        if not combined_text:
            logger.warning("POST /ingest - Crawl returned no content for URL: %s", url)
            return jsonify({"error": "crawl returned no content"}), 400

    logger.info("POST /ingest - Combined text length: %d characters", len(combined_text))

    db = mongo_client[database_name]
    wg_db = mongo_client[WEB_GATHER_DB]

    # Track previous document count for response
    previous_document_count = 0
    overwritten = False

    # Handle overwrite mode - clear existing data in collection
    if mode == "overwrite":
        logger.info("POST /ingest - Overwrite mode: checking for existing data in %s.%s", database_name, collection_name)
        previous_document_count = db[collection_name].count_documents({})

        if previous_document_count > 0:
            logger.info("POST /ingest - Overwrite mode: clearing %d existing documents", previous_document_count)

            # Delete documents from the main collection
            db[collection_name].delete_many({})

            # Delete document metadata records for this collection
            wg_db[DOCUMENTS_COLLECTION].delete_many({
                "database_name": database_name,
                "collection_name": collection_name,
            })

            overwritten = True
            logger.info("POST /ingest - Overwrite mode: existing data cleared successfully")

    document_id = str(uuid.uuid4())
    logger.info("POST /ingest - Created document ID: %s", document_id)

    logger.info("POST /ingest - Storing document in %s.%s (mode: %s)", database_name, collection_name, mode)
    db[collection_name].insert_one(
        {
            "_id": document_id,
            "source_url": url,
            "document_type": document_type,
            "text": combined_text,
            "created_at": utc_now(),
        }
    )

    description = combined_text[:300]
    logger.info("POST /ingest - Creating document metadata record")
    wg_db[DOCUMENTS_COLLECTION].insert_one(
        {
            "document_id": document_id,
            "database_name": database_name,
            "collection_name": collection_name,
            "source_url": url,
            "document_type": document_type,
            "description": description,
            "created_at": utc_now(),
        }
    )

    response_data = {
        "document_id": document_id,
        "document_type": document_type,
        "pages": page_count,
        "database_name": database_name,
        "collection_name": collection_name,
        "mode": mode,
        "message": "Document loaded successfully. Use /index endpoint to create vector embeddings.",
    }

    if mode == "overwrite":
        response_data["overwritten"] = overwritten
        response_data["previous_document_count"] = previous_document_count

    logger.info("POST /ingest - Document loaded successfully (mode: %s): %s", mode, document_id)
    return jsonify(response_data)


@core_bp.post("/index")
def index_document():
    """Index a document by id."""
    logger.info("POST /index - Starting document indexing")
    payload = request.get_json(silent=True) or {}

    # Get required parameters
    source_database_name = payload.get("source_database_name")
    source_collection_name = payload.get("source_collection_name")
    source_document_id = payload.get("source_document_id")
    index_database_name = payload.get("index_database_name")
    index_collection_name = payload.get("index_collection_name")

    # Validate required parameters
    missing_params = []
    if not source_database_name:
        missing_params.append("source_database_name")
    if not source_collection_name:
        missing_params.append("source_collection_name")
    if not source_document_id:
        missing_params.append("source_document_id")
    if not index_database_name:
        missing_params.append("index_database_name")
    if not index_collection_name:
        missing_params.append("index_collection_name")

    if missing_params:
        logger.warning("POST /index - Missing required parameters: %s", ", ".join(missing_params))
        return jsonify({
            "error": f"Missing required parameters: {', '.join(missing_params)}"
        }), 400

    logger.info("POST /index - Source: %s.%s, Document ID: %s",
                source_database_name, source_collection_name, source_document_id)
    logger.info("POST /index - Index target: %s.%s", index_database_name, index_collection_name)

    # Check if embedding model is configured for index_database_name
    model_name = get_embedding_model_name(index_database_name)
    if not model_name:
        logger.error("POST /index - No embedding model configured for database: %s", index_database_name)
        return jsonify({
            "error": f"No embedding model configured for index_database_name '{index_database_name}'. "
                     f"Use POST /embedding-models to configure one."
        }), 400

    # Fetch the source document
    source_db = mongo_client[source_database_name]
    document = source_db[source_collection_name].find_one({"_id": source_document_id})
    if not document:
        logger.warning("POST /index - Document not found: %s in %s.%s",
                       source_document_id, source_database_name, source_collection_name)
        return jsonify({
            "error": f"Document not found: {source_document_id} in {source_database_name}.{source_collection_name}"
        }), 404

    text = document.get("text", "")
    if not text:
        logger.warning("POST /index - Document has no text content: %s", source_document_id)
        return jsonify({"error": "document has no content"}), 400

    # Use shared indexing function with separate source and index parameters
    index_result = index_document_chunks(
        document_id=source_document_id,
        text=text,
        database_name=source_database_name,
        collection_name=source_collection_name,
        index_database_name=index_database_name,
        index_collection_name=index_collection_name,
    )

    if not index_result:
        logger.error("POST /index - Indexing failed for document: %s", source_document_id)
        return jsonify({"error": "indexing failed"}), 500

    logger.info("POST /index - Successfully indexed document: %s", source_document_id)
    return jsonify(
        {
            "source_database_name": source_database_name,
            "source_collection_name": source_collection_name,
            "source_document_id": source_document_id,
            "index_database_name": index_database_name,
            "index_collection_name": index_collection_name,
            "chunks_indexed": index_result["chunks_indexed"],
            "embedding_model": index_result["embedding_model"],
            "chunk_collection": index_result["chunk_collection"],
        }
    )


@core_bp.get("/search")
def search():
    """Search vector-indexed collection."""
    logger.info("GET /search - Starting vector search")
    document_id = request.args.get("document_id")
    query = request.args.get("query")
    if not document_id or not query:
        logger.warning("GET /search - Missing required parameters")
        return jsonify({"error": "document_id and query are required"}), 400

    logger.info("GET /search - Searching document %s for: %s", document_id, query)
    wg_db = mongo_client[WEB_GATHER_DB]
    doc_record = wg_db[DOCUMENTS_COLLECTION].find_one({"document_id": document_id})
    if not doc_record:
        logger.warning("GET /search - Document not found: %s", document_id)
        return jsonify({"error": "document_id not found"}), 404

    database_name = doc_record["database_name"]
    collection_name = doc_record["collection_name"]

    # Use index location from metadata if available, otherwise fall back to source location
    index_database_name = doc_record.get("index_database_name", database_name)
    chunk_collection = doc_record.get("chunk_collection", collection_name)

    model_name = get_embedding_model_name(index_database_name)
    if not model_name:
        logger.error("GET /search - No embedding model configured for database: %s", index_database_name)
        return jsonify({"error": "embedding model not configured"}), 400

    db = mongo_client[index_database_name]
    logger.info("GET /search - Loading chunks from %s.%s", index_database_name, chunk_collection)
    chunks = list(
        db[chunk_collection].find({"document_id": document_id}, {"_id": 0})
    )
    if not chunks:
        logger.warning("GET /search - No chunks indexed for document: %s", document_id)
        return jsonify({"error": "no chunks indexed for document"}), 400

    logger.info("GET /search - Found %d chunks, generating query embedding", len(chunks))
    model = get_model(model_name)
    query_embedding = model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    )[0]
    chunk_embeddings = [np.array(c["embedding"]) for c in chunks]

    logger.info("GET /search - Computing cosine similarity scores")
    scores = cosine_similarity(query_embedding, chunk_embeddings)

    results = []
    for chunk, score in zip(chunks, scores):
        results.append(
            {
                "text": chunk["text"],
                "score": float(score),
                "percent_match": max(0.0, min(100.0, float(score) * 100.0)),
            }
        )
    results.sort(key=lambda x: x["score"], reverse=True)

    top_results = results[:5]
    logger.info("GET /search - Returning top %d results (best score: %.3f)",
                len(top_results), top_results[0]["score"] if top_results else 0)
    return jsonify(
        {
            "document_id": document_id,
            "query": query,
            "results": top_results,
        }
    )


DEFAULT_PARSING_PROMPT = """Parse the document into sections logically based on the type of information in the document.
For example, legal documents should be parsed into section or clauses."""


@core_bp.get("/parse_llm")
def parse_llm():
    """Parse document text using LLM."""
    logger.info("GET /parse_llm - Starting LLM document parsing")
    document = request.args.get("document")
    parsing_prompt = request.args.get("parsing_prompt")
    document_id = request.args.get("document_id", "doc_001")

    if not document:
        logger.warning("GET /parse_llm - Missing required parameter: document")
        return jsonify({"error": "document is required"}), 400

    # Use default parsing prompt if not provided
    if not parsing_prompt:
        parsing_prompt = DEFAULT_PARSING_PROMPT
        logger.info("GET /parse_llm - Using default parsing prompt")

    logger.info("GET /parse_llm - Parsing document (length: %d) with prompt: %s...",
                len(document), parsing_prompt[:50])

    system_prompt = """You are a document parsing assistant. Your task is to parse documents according to specific instructions and return structured JSON output.

You must return ONLY valid JSON with the following structure:
- A top-level object with a "parsed_doc" key
- The "parsed_doc" value should be an array of objects
- Each object in the array represents one parsed section and must contain:
  - "document_id": A string identifier for the document (use the provided document_id)
  - "parsed_header_text": A string containing the header, title, or identifying text for this section
  - "parsed_text": A string containing the main content/body text for this section

Important guidelines:
- Preserve the original text accurately - do not paraphrase or summarize unless explicitly instructed
- Each section should be complete and meaningful on its own
- Escape special characters properly in the JSON (quotes, newlines, etc.)
- The parsed_header_text should be concise but descriptive enough to identify what the section contains
- The parsed_text should contain the substantive content of that section
- Return ONLY the JSON object, no additional text or markdown formatting"""

    user_message = f"""Parse the following document according to the parsing instructions provided.

<document>
{document}
</document>

<parsing_prompt>
{parsing_prompt}
</parsing_prompt>

<document_id>{document_id}</document_id>

Return ONLY the JSON output with the parsed document sections."""

    try:
        logger.info("GET /parse_llm - Calling Anthropic API (model: claude-3-haiku-20240307)")
        message = anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=4096,
            messages=[
                {"role": "user", "content": user_message}
            ],
            system=system_prompt,
        )

        response_text = message.content[0].text.strip()
        logger.info("GET /parse_llm - Received response (length: %d)", len(response_text))

        # Try to extract JSON if wrapped in code blocks
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            # Remove first and last lines (code block markers)
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            response_text = "\n".join(lines)

        parsed_result = json.loads(response_text)
        section_count = len(parsed_result.get("parsed_doc", []))
        logger.info("GET /parse_llm - Successfully parsed document into %d sections", section_count)
        return jsonify(parsed_result)

    except json.JSONDecodeError as e:
        logger.error("GET /parse_llm - Failed to parse LLM response as JSON: %s", e)
        return jsonify({"error": f"Failed to parse LLM response as JSON: {str(e)}"}), 500
    except Exception as e:
        logger.error("GET /parse_llm - LLM parsing failed: %s", e)
        return jsonify({"error": f"LLM parsing failed: {str(e)}"}), 500
