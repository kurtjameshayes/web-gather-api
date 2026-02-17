# AGENTS.md — Backend Processing Systems

## Stack & Environment

- **Language**: Python 3.11+
- **Database**: MongoDB (via `pymongo` or `motor` for async)
- **Task/Queue**: Prefer lightweight approaches first (in-process queues, `asyncio.Queue`) before reaching for Celery/Redis
- **Config**: Use environment variables via `python-dotenv`. Never hardcode secrets or connection strings.
- **Package management**: `pip` with `requirements.txt` or `pyproject.toml`

## Architecture Principles

### Design for Failure
Every backend process MUST assume it will crash mid-execution. Design accordingly:
- All database writes should be **idempotent**. Re-running the same input must produce the same result without duplication.
- Use MongoDB `update_one` with `$set` / `$setOnInsert` and upserts over blind `insert_one` where possible.
- Track processing state per-record (e.g., `status: pending | processing | completed | failed`) so work can resume after crashes.
- Write a `last_updated` timestamp on every mutation.

### Pipeline Pattern
Structure multi-step processing as explicit pipelines with discrete stages:

```python
# Good: Each stage is independently testable, retryable, and loggable
class Pipeline:
    def __init__(self, stages: list[Callable]):
        self.stages = stages

    def run(self, item: dict) -> dict:
        for stage in self.stages:
            item = stage(item)
        return item

# Bad: One giant function that fetches, transforms, validates, and writes
def do_everything(raw_input):
    ...
```

Each stage should:
1. Accept a dict/dataclass and return a dict/dataclass (explicit data flow)
2. Have a single responsibility
3. Be independently testable with no DB required
4. Log its inputs/outputs at DEBUG level

### Separation of Concerns
- **Never mix I/O and transformation logic.** Functions either read/write external systems OR transform data — not both.
- Data access goes in a dedicated `db/` or `repository/` layer. Processing logic never calls `pymongo` directly.
- HTTP/API calls go through a client wrapper with retry logic, not inline `requests.get()`.

## MongoDB Patterns

### Schema Design
- Always define your expected document shape in a docstring or dataclass, even though Mongo is schemaless.
- Include `created_at` and `updated_at` on every collection.
- Use `_id` strategically — if there's a natural unique key (e.g., `jurisdiction + statute_id`), use a compound `_id` instead of ObjectId. This gives you free uniqueness enforcement and upsert capability.

### Queries & Performance
- **Every query must have a supporting index.** Before writing a `find()`, confirm the index exists or add `ensure_index` / `create_index` to the setup code.
- Use **projections** — never fetch full documents when you only need 2 fields.
- Prefer `bulk_write` with ordered=False for batch operations over loops of `update_one`.
- Use the aggregation pipeline for complex reads instead of pulling data into Python and filtering.
- **Pagination**: Use `_id`-based cursor pagination, not `skip/limit`.

### Connection Management
```python
# Good: Single client, reused across the application
client = MongoClient(MONGO_URI)
db = client[DB_NAME]

# Bad: Creating a new MongoClient per function call
def get_data():
    client = MongoClient(MONGO_URI)  # DON'T
```

## Error Handling & Resilience

### Retry Strategy
Use exponential backoff with jitter for transient failures. Wrap in a decorator:

```python
import time, random
from functools import wraps

def retry(max_attempts=3, base_delay=1.0, exceptions=(Exception,)):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        raise
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(delay)
        return wrapper
    return decorator
```

### Error Classification
Distinguish between:
- **Transient**: Network timeouts, rate limits, DB connection drops → retry
- **Permanent**: Validation errors, missing required fields, bad data → log + skip + continue
- **Fatal**: Config errors, auth failures, schema mismatches → crash immediately with clear message

Never catch bare `Exception` without logging. Never silently swallow errors.

### Dead Letter Pattern
Records that fail after all retries should be written to a `_failed` collection or `status: failed` with the error message and timestamp, not just logged and forgotten.

```python
except PermanentError as e:
    db.failed_records.insert_one({
        "source_collection": "documents",
        "source_id": record["_id"],
        "error": str(e),
        "failed_at": datetime.utcnow(),
        "input_snapshot": record
    })
```

## Logging & Observability

- Use `logging` module, not `print()`. Configure structured logging (JSON) for anything that runs in production.
- Every processing run should log: start time, record count, success count, failure count, elapsed time.
- Use a `run_id` (UUID) per execution for tracing.
- Log at appropriate levels: DEBUG for per-record detail, INFO for stage transitions, WARNING for retries, ERROR for failures.

## Code Style

### Type Hints
Use type hints on all function signatures. Use `TypedDict` or `dataclass` for structured data instead of raw dicts where the shape is known.

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ProcessingResult:
    record_id: str
    status: str  # "completed" | "failed"
    output: Optional[dict] = None
    error: Optional[str] = None
```

### Naming
- Functions that hit the DB or network: prefix with verb describing the I/O (`fetch_`, `store_`, `send_`)
- Pure transformation functions: name describes the transformation (`normalize_`, `validate_`, `enrich_`)
- Boolean functions: prefix with `is_` or `has_`

### File Structure
```
project/
├── main.py              # Entry point, CLI args, orchestration
├── config.py            # Env vars, constants
├── pipeline/
│   ├── stages.py        # Pure transformation functions
│   └── runner.py        # Pipeline orchestration
├── db/
│   ├── client.py        # Connection setup
│   └── repositories.py  # All DB read/write operations
├── clients/
│   └── api_client.py    # External API wrappers with retry
├── models/
│   └── schemas.py       # Dataclasses / TypedDicts
└── tests/
    ├── test_stages.py   # Unit tests for transformations
    └── test_pipeline.py # Integration tests
```

## Anti-Patterns — DO NOT

1. **Don't process records one-at-a-time when batch is possible.** Always check if the API/DB supports batch operations.
2. **Don't use `time.sleep()` for rate limiting** — use a proper rate limiter (`ratelimit` lib or token bucket).
3. **Don't store processing config in code.** Use env vars or a config collection.
4. **Don't write "smart" code.** Prefer boring, explicit, readable code over clever one-liners. Backend processing code gets debugged at 2 AM.
5. **Don't nest try/except more than 2 levels.** If you need to, refactor into smaller functions.
6. **Don't use threads for I/O-bound work** — use `asyncio` or batch the operations.
7. **Don't fetch all records into memory.** Use cursors with `batch_size` or generator patterns for large collections.
8. **Don't ignore `pymongo` write concerns.** Use `w="majority"` for critical writes.

## Testing Expectations

- **Unit tests** for all transformation/validation logic — no DB, no network, fast.
- **Integration tests** for DB operations using a test database (not mocks).
- Test the failure paths: what happens when the API returns 500? When a required field is missing? When the DB is down?
- Include at least one end-to-end test that runs the full pipeline on fixture data.

## When Generating Code

1. **Start with the data model.** Define what goes in and what comes out before writing logic.
2. **Show the error path.** Every function that can fail should show how failures are handled.
3. **Include the index.** If you write a query, include the `create_index` call.
4. **Batch by default.** Default to batch processing unless there's a reason for single-record.
5. **Add docstrings** to public functions explaining what they do, what they raise, and any side effects.
6. **Provide a CLI entry point** using `argparse` or `click` with `--dry-run` support.
