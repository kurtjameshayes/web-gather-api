# web-gather-api

Flask API for gathering, ingesting, indexing, and searching web documents.

## Setup

1. Create a `.env` file in the repository root:

```
FIRECRAWL_API_KEY=your_firecrawl_key
MONGODB_URI=mongodb://localhost:27017
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. (Optional) For crawl endpoint: Playwright requires browser binaries. After pip install, run:

```
playwright install chromium
```

Selenium requires Chrome/Chromium and chromedriver (or use webdriver-manager). The crawl endpoint tries Playwright first, then Puppeteer (pyppeteer), then Selenium, then Firecrawl API.

4. Run the server:

```
python app.py
```

## Statute-Policy Compliance API (Flask)

This repository includes a Flask endpoint for statute-first compliance analysis.
It iterates statutory requirements via category mappings, finds matching policy
chunks via vector search, and evaluates compliance gaps.

### Required environment variables

```
MONGODB_URI=mongodb://localhost:27017
ANTHROPIC_API_KEY=your_anthropic_key
COMPLIANCE_API_KEY=your_api_key
```

Optional configuration (see `policy_compliance_config.json`):

```
COMPLIANCE_CONFIG_PATH=/workspace/policy_compliance_config.json
COMPLIANCE_AUTH_REQUIRED=true
COMPLIANCE_ALLOWED_ROLES=admin,compliance
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
LLM_MODEL_NAME=claude-sonnet-4-6
AUDIT_LOG_KEY=base64_fernet_key
```

### Compliance endpoint (served by Flask)

```
POST /statute-policy-compliance
```

Example request/response JSON files are provided in:

- `example_request.json`
- `example_response.json`

When `COMPLIANCE_AUTH_REQUIRED=true`, include headers:

```
x-api-key: your_api_key
x-role: admin
```

### Compliance retrieval (vector index and config)

The compliance endpoint retrieves statute chunks by vector similarity and jurisdiction. For it to return candidates, config must match your data:

- **Where the vector index lives**
  - **Option A (embeddings collection):** If you run `POST /create-embeddings` with source e.g. `statute_chunks` and index target `statute_embeddings`, create the MongoDB Atlas vector index on **statute_embeddings**. Set in `policy_compliance_config.json`: `"use_embeddings_collection": true`, `"embeddings_collection": "statute_embeddings"`. The index path must match the field name that `/create-embeddings` writes (e.g. `embedding`); set `embedding_vector_field` to that name.
  - **Option B (statutes collection):** If the vector index is on the **statutes** (or statute_chunks) collection, set `"use_embeddings_collection": false` and ensure that collection has both the vector field and a `jurisdiction` field (or whatever `statute_jurisdiction_field` is). The index path and `vector_field` must match (e.g. `vector` or `embedding`).
- **Vector field name:** `/create-embeddings` writes the vector as **`embedding`**. If your index is on that collection, use `vector_field` / `embedding_vector_field` equal to `"embedding"` so search uses the same field.
- **Atlas vector index:** The search index name in Atlas must match `vector_index_name` (e.g. `statute_vector_index`). The index definition must use the same **path** as `embedding_vector_field` (e.g. `embedding`) and **numDimensions** must match the embedder output (e.g. **384** for `all-MiniLM-L6-v2`). Example Atlas index definition for `statute_embeddings`: `{"fields":[{"type":"vector","path":"embedding","numDimensions":384,"similarity":"cosine"}]}`. If any of these differ (index name, path, or dimensions), `$vectorSearch` can return 0 results.
- **Jurisdiction:** Stored values are compared in normalized form (e.g. "California" and "CA" both normalize to "CA"). Use one normalized form consistently (e.g. two-letter codes) in stored documents and requests.
- **Statute database:** Statute and embedding retrieval use the **compliance database** by default. If you ran `POST /create-embeddings` and wrote `statute_embeddings` into a different MongoDB database, set `"statute_database"` in `policy_compliance_config.json` to that database name (the same as `index_database_name` in the create-embeddings request). Leave it empty or omit it to use the compliance database.
- **Compliance readiness:** To verify retrieval, call `POST /statute-policy-compliance` with a known `policy_id` and `jurisdiction`; if the response returns zero gaps with `retrieval_metadata.statute_items_considered: 0`, check that category mappings and statute embeddings exist for the given jurisdiction.

### Tests

```
pytest
```

## Endpoints

- `POST /gather`
- `POST /ingest`
- `GET /documents`
- `GET /collections`
- `GET /count-documents`
- `POST /create-embeddings`
- `POST /search`
- `GET /embedding-models`
- `POST /embedding-models`
- `POST /create-vector-index` (create Atlas vector index from web-gather embedding_model)
- `POST /statute-policy-compliance`