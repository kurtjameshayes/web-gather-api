# Web Gather API Requirements

This document is derived from the current codebase behavior and structure.

## Functional Requirements

### 1) Web Document Gathering & Ingestion

- FR-1.1 Web Search (`POST /gather`)
  - Search the web via Firecrawl and return results with relevance scores and
    percent match values.
  - Provide guidance for the next ingestion step.
  - Files: `core.py`
- FR-1.2 Document Ingestion (`POST /ingest`)
  - Ingest web pages (via Firecrawl) and PDFs (download and parse).
  - Auto-detect PDFs by URL extension or Content-Type.
  - Store raw documents in MongoDB with metadata and timestamps.
  - Support `append` (default) and `overwrite` (clear existing) modes.
  - Files: `core.py`
- FR-1.3 Web Crawling (`POST /crawl`)
  - Crawl URLs with configurable depth and breadth.
  - Prefer Puppeteer and fall back to Firecrawl on failure.
  - Return combined text from all crawled pages.
  - Files: `core.py`

### 2) Document Indexing & Vector Search

- FR-2.1 Document Indexing (`POST /index`)
  - Generate vector embeddings from ingested documents.
  - Support chunking strategies: character, sentence, paragraph, semantic.
  - Configurable `chunk_size` and `chunk_overlap`.
  - Store chunk embeddings in MongoDB.
  - Files: `core.py`, `embedder.py`
- FR-2.2 Vector Search (`GET /search`)
  - Search indexed documents by query using cosine similarity.
  - Return top results with similarity scores and percent match.
  - Files: `core.py`
- FR-2.3 Embedding Model Management
  - Configure embedding models per database (`POST /embedding-models`).
  - List configured embedding models (`GET /embedding-models`).
  - Files: `util.py`, `db.py`

### 3) Database Operations

- FR-3.1 Document Management
  - List documents with optional query filters.
  - Count documents in collections.
  - Write documents to collections with append or replace modes.
  - Files: `db.py`
- FR-3.2 Database and Collection Discovery
  - List databases with uploaded documents.
  - List all MongoDB databases and collections.
  - Files: `db.py`

### 4) LLM Document Parsing

- FR-4.1 LLM-Based Parsing (`POST /parse-llm`)
  - Parse documents using an LLM with a custom prompt.
  - Support single document or full collection parsing.
  - Return structured sections with parsed header and text.
  - Files: `core.py`, `llm_client.py`

### 5) Policy-Statute Compliance

- FR-5.1 Compliance Analysis (`POST /policy-statute-compliance`)
  - Segment policy text into sections.
  - Retrieve relevant statutes via vector search.
  - Compare policy sections to statutes using an LLM.
  - Classify compliance as compliant, non_compliant, or neither.
  - Return per-section results with confidence, rationale, and remediation.
  - Return a summary with overall compliance status and counts.
  - Files: `compliance_routes.py`, `compliance_service.py`
- FR-5.2 Statute Retrieval
  - Vector search in statutes or embeddings collections.
  - Filter by jurisdiction and optional corpus id.
  - Configurable top_k retrieval and evidence thresholds.
  - Files: `vector_retriever.py`
- FR-5.3 Compliance Evaluation
  - Parse LLM JSON responses with fallbacks.
  - Apply confidence thresholds per compliance type.
  - Apply rule-based checks where applicable.
  - Files: `compliance_evaluator.py`
- FR-5.4 PII Redaction
  - Redact emails, SSNs, phone numbers, and credit cards before processing.
  - Files: `redactor.py`
- FR-5.5 Audit Logging
  - Log compliance checks with section hashes.
  - Support optional encryption of audit records and payloads.
  - Files: `audit_logger.py`

## Non-Functional Requirements

### 1) Performance

- NFR-1.1 Caching
  - LRU cache for embeddings and retrieval results with TTL support.
  - Files: `cache.py`, `embedder.py`, `vector_retriever.py`
- NFR-1.2 Concurrency
  - Async I/O for LLM and retrieval workflows.
  - Semaphore-based concurrency control for LLM calls.
  - Files: `compliance_service.py`, `llm_client.py`
- NFR-1.3 Rate Limiting
  - Token-bucket rate limiting for requests.
  - Files: `rate_limiter.py`

### 2) Security

- NFR-2.1 Authentication and Authorization
  - Optional API key authentication via `x-api-key`.
  - Role-based access control via `x-role`.
  - Files: `security.py`, `compliance_config.py`
- NFR-2.2 Data Protection
  - PII redaction before processing.
  - Optional encrypted audit logs with hashed identifiers.
  - Files: `redactor.py`, `audit_logger.py`

### 3) Configuration

- NFR-3.1 Environment Variables
  - Required: `FIRECRAWL_API_KEY`, `MONGODB_URI`, `ANTHROPIC_API_KEY`
  - Optional: `COMPLIANCE_API_KEY`, `COMPLIANCE_CONFIG_PATH`,
    `EMBEDDING_MODEL_NAME`, `LLM_MODEL_NAME`, `AUDIT_LOG_KEY`
  - Files: `app.py`, `compliance_config.py`
- NFR-3.2 Configuration File
  - JSON config support via `policy_compliance_config.json`.
  - Environment variables override config defaults.
  - Files: `compliance_config.py`

### 4) Reliability

- NFR-4.1 Error Handling
  - Validate inputs and return appropriate 4xx errors.
  - Service errors return 5xx with diagnostics.
  - Files: route handlers in `core.py`, `db.py`, `compliance_routes.py`
- NFR-4.2 Logging
  - Structured logging with standard levels.
  - Files: modules throughout codebase

### 5) Data Management

- NFR-5.1 MongoDB Structure
  - Documents: `web-gather.documents`
  - Embedding models: `web-gather.embedding_model`
  - Compliance audit: `{database}.policy_compliance_audit`
  - Statutes: `{database}.statutes`
  - Embeddings: `{database}.embeddings`
  - Files: `db.py`, `compliance_config.py`
- NFR-5.2 Document Storage
  - Raw documents include source URL, type, text, and timestamps.
  - Indexed chunks include document id, chunk index, text, embedding.
  - Files: `core.py`, `db.py`

### 6) API Documentation

- NFR-6.1 OpenAPI Specification
  - OpenAPI 3.0.3 spec.
  - Swagger UI at `/docs` and JSON at `/openapi.json`.
  - Files: `routes.py`

## Constraints and Dependencies

- MongoDB is required, and Atlas Vector Search is used for compliance retrieval.
- External dependencies include Firecrawl and Anthropic APIs.
- Python dependencies include Flask, PyMongo, SentenceTransformers, PyPDF,
  Pyppeteer, and Anthropic SDK.
- Default server port is 7000 (configurable).
