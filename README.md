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

3. Run the server:

```
python app.py
```

## Policy Statute Compliance API (Flask)

This repository includes a Flask endpoint for comparing privacy policy sections
against statute excerpts with compliance determinations.

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
LLM_MODEL_NAME=claude-3-5-haiku-20241022
AUDIT_LOG_KEY=base64_fernet_key
```

### Compliance endpoint (served by Flask)

```
POST /policy-statute-compliance
```

Example request/response JSON files are provided in:

- `example_request.json`
- `example_response.json`

When `COMPLIANCE_AUTH_REQUIRED=true`, include headers:

```
x-api-key: your_api_key
x-role: admin
```

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
- `POST /vector-index`
- `POST /search`
- `GET /embedding-models`
- `POST /embedding-models`
- `POST /policy-statute-compliance`