# Gap Analysis Process Design

## 1. High-Level Flow

```
Statute Items (embeddings) ──► Vector Search ──► Candidate Policy Sections
                                                        │
                                                        ▼
                                              LLM Analysis (per pair)
                                                        │
                                                        ▼
                                              Citation Binding Check
                                                        │
                                                        ▼
                                              Aggregation & Response
```

The core loop is: **for each statute requirement, find the most relevant policy text via vector similarity, then ask the LLM to classify the relationship.**

---

## 2. Step-by-Step Process

### Step 0: Input Validation & Prerequisite Check

- Validate `policy_document_id` exists in `policies`.
- Confirm policy is indexed: query `policy_embeddings` (v2) or `policy_sub_embeddings` (v1) for at least one record matching `policy_document_id`. If none → return 400 with indexing instructions.
- Resolve jurisdictions: if `applicable_jurisdictions` provided, use them. Otherwise, call the applicability service or default to all available jurisdictions.

### Step 1: Retrieve Statute Items

Query the statute collection filtered by jurisdiction:

**v1 (subchunk):**
```python
statute_items = db.statute_sub_embeddings.find(
    {"jurisdiction": {"$in": applicable_jurisdictions}},
    {"embedding": 1, "text": 1, "parent_chunk_id": 1,
     "statute_reference": 1, "jurisdiction": 1}
).limit(num_rows)  # if num_rows specified
```

**v2 (chunk):**
```python
statute_items = db.statute_embeddings.find(
    {"jurisdiction": {"$in": applicable_jurisdictions}},
    {"embedding": 1, "text": 1, "statute_reference": 1, "jurisdiction": 1}
).limit(num_rows)
```

Each statute item represents one requirement to evaluate against the policy.

### Step 2: Vector Search — Find Relevant Policy Sections

For each statute item, run a vector similarity search against the policy embeddings to find the top-k most relevant policy sections.

**v2 (chunk-to-chunk):**
```python
pipeline = [
    {
        "$vectorSearch": {
            "index": "policy_embeddings_vector_index",
            "path": "embedding",
            "queryVector": statute_item["embedding"],
            "numCandidates": 50,
            "limit": 5,
            "filter": {"policy_document_id": policy_document_id}
        }
    },
    {
        "$project": {
            "text": 1,
            "score": {"$meta": "vectorSearchScore"},
            "section_id": 1
        }
    }
]
matches = list(db.policy_embeddings.aggregate(pipeline))
```

**v1 (subchunk-to-subchunk):**
Same structure but against `policy_sub_embeddings`. Additionally, for each matched subchunk, retrieve the parent chunk text for context:

```python
# After getting subchunk matches:
parent_chunk_ids = [m["parent_chunk_id"] for m in matches]
parent_chunks = {
    doc["_id"]: doc["text"]
    for doc in db.policy_embeddings.find({"_id": {"$in": parent_chunk_ids}})
}
# Attach parent context to each match
for m in matches:
    m["parent_context"] = parent_chunks.get(m["parent_chunk_id"], "")
```

**Similarity Threshold:** Discard matches below a configurable score threshold (e.g., 0.70). This prevents the LLM from being asked to analyze irrelevant policy text. If all matches fall below threshold, the pair still proceeds to the LLM but with an empty policy context — the LLM should classify it as `missing`.

**Output of this step:** A list of `(statute_item, [policy_matches])` pairs. Track `statute_items_considered` and `statute_pairs_matched` (pairs where at least one match exceeded the threshold).

### Step 3: LLM Analysis

For each `(statute_item, [policy_matches])` pair, send to the LLM.

**Why per-pair, not batch?** Each statute requirement needs focused analysis against its specific matched policy text. Batching multiple requirements into one prompt risks cross-contamination of reasoning and makes citation binding harder to validate.

#### v2 Prompt Structure (chunk-level):

```yaml
system: |
  You are a privacy compliance analyst. You compare a statutory requirement
  against a company's privacy policy text and determine compliance status.

  You MUST respond with valid JSON matching this schema:
  {
    "status": "addressed" | "missing" | "conflict",
    "policy_quote": "<exact verbatim substring from the policy text, or null>",
    "statute_quote": "<relevant excerpt from the statute text>",
    "requirement_summary": "<one-sentence summary of what the statute requires>",
    "conflict_description": "<explanation if status is conflict, else null>",
    "confidence": "high" | "medium" | "low"
  }

  Rules:
  - "addressed": The policy text clearly satisfies the statutory requirement.
    You MUST provide a policy_quote that is an EXACT verbatim substring of the
    policy text. Do not paraphrase or truncate.
  - "missing": The policy text does not address the requirement at all, or
    only partially/ambiguously addresses it. Partial matches are MISSING.
  - "conflict": The policy text directly contradicts the requirement.
    You MUST provide both a policy_quote AND a conflict_description.
  - If no policy text is provided, the status MUST be "missing".
  - policy_quote must be copy-paste identical to a substring in the policy.

user: |
  STATUTE REQUIREMENT:
  {STATUTE_CHUNK}

  POLICY TEXT:
  {POLICY_CHUNK}

  Analyze whether this policy text satisfies the statutory requirement.
```

Where `{POLICY_CHUNK}` is the concatenation of the top-k matched policy sections (joined with section separators), and `{STATUTE_CHUNK}` is the statute item text.

#### v1 Prompt Structure (subchunk-level):

Same schema, but the prompt includes parent chunk context:

```yaml
user: |
  STATUTE REQUIREMENT (subchunk):
  {STATUTE_SUBCHUNK}

  STATUTE CONTEXT (enclosing section):
  {STATUTE_PARENT_CHUNK}

  POLICY TEXT (subchunk matches):
  {POLICY_SUBCHUNK}

  POLICY CONTEXT (enclosing sections):
  {POLICY_PARENT_CHUNKS}

  Analyze whether this policy text satisfies the statutory requirement.
  Use the context sections for understanding, but quote only from the
  policy subchunk text.
```

#### LLM Output Parsing:

Parse the JSON response. If parsing fails or the response doesn't conform to the schema, set `analysis_failed: true` for that item and continue.

### Step 4: Citation Binding Validation

This is a critical post-processing step. For each item where `status == "addressed"` or `status == "conflict"` and `policy_quote` is not null:

```python
# Fetch the full policy text
policy_text = db.policies.find_one(
    {"document_id": policy_document_id}
)["text"]

# Normalize whitespace for comparison
def normalize(s):
    return " ".join(s.split()).lower()

policy_norm = normalize(policy_text)
quote_norm = normalize(item["policy_quote"])

if quote_norm not in policy_norm:
    # Citation binding failed
    if item["status"] == "addressed":
        item["status"] = "missing"
        item["policy_quote"] = None
        item["citation_binding_failed"] = True
    elif item["status"] == "conflict":
        # Keep conflict status but flag the quote
        item["citation_binding_failed"] = True
```

**Why normalize?** The LLM may introduce minor whitespace differences. Lowercasing and collapsing whitespace handles this without being too permissive. Do NOT do fuzzy matching — the spec says verbatim.

### Step 5: Aggregation

```python
summary = {
    "total_requirements": len(gap_items),
    "addressed": sum(1 for g in gap_items if g["status"] == "addressed" and not g.get("analysis_failed")),
    "missing": sum(1 for g in gap_items if g["status"] == "missing" and not g.get("analysis_failed")),
    "conflicts": sum(1 for g in gap_items if g["status"] == "conflict" and not g.get("analysis_failed")),
    "analysis_failures": sum(1 for g in gap_items if g.get("analysis_failed"))
}

retrieval_metadata = {
    "statute_items_considered": statute_items_considered,
    "statute_pairs_matched": statute_pairs_matched
}
```

### Step 6: Persistence (if `save_results == true`)

```python
result_doc = {
    "policy_document_id": policy_document_id,
    "company_name": company_name,
    "applicable_jurisdictions": applicable_jurisdictions,
    "analyzed_at": datetime.utcnow().isoformat() + "Z",
    "gaps": gap_items,
    "summary": summary,
    "retrieval_metadata": retrieval_metadata,
    "run_type": "gap_analysis_v2",  # or v1
    "version": "v2"
}
db.compliance_results.insert_one(result_doc)

run_log = {
    "policy_document_id": policy_document_id,
    "run_type": "gap_analysis_v2",
    "statute_item_ids": [s["_id"] for s in statute_items_used],
    "ran_at": datetime.utcnow().isoformat() + "Z",
    "summary": summary
}
db.compliance_run_log.insert_one(run_log)
```

---

## 3. Vector Search Design Details

### Index Requirements

| Collection | Index Name | Type | Path | Similarity |
|---|---|---|---|---|
| `policy_embeddings` | `policy_embeddings_vector_index` | vectorSearch | `embedding` | cosine |
| `policy_sub_embeddings` | `policy_sub_embeddings_vector_index` | vectorSearch | `embedding` | cosine |
| `statute_embeddings` | — | standard | `jurisdiction` | — |
| `statute_sub_embeddings` | — | standard | `jurisdiction` | — |

The vector search is always **statute → policy direction** (statute embedding as the query vector, searching against policy embeddings). This is intentional: we want to find what the policy says about each statutory requirement, not the other way around.

### Why Not Policy → Statute?

Searching policy → statute would tell us which statutes each policy section relates to, but would miss statutes that the policy doesn't address at all. Statute → policy ensures every requirement gets evaluated, including ones the policy is silent on (which correctly become `missing`).

### Tuning Parameters

| Parameter | Default | Notes |
|---|---|---|
| `top_k` | 5 | Number of policy matches per statute item. Higher = more context for LLM but more noise. |
| `num_candidates` | 50 | Broader candidate pool for ANN search accuracy. |
| `score_threshold` | 0.70 | Below this, matches are discarded. Tunable per deployment. |

---

## 4. LLM Integration Design

### Model Selection

Use the fine-tuned compliance model when available (your Fireworks deployment). Fall back to a general model (e.g., Claude or GPT-4) with the structured prompts above.

### Concurrency

Statute items are independent — process them in parallel with a concurrency pool:

```python
import asyncio
from asyncio import Semaphore

sem = Semaphore(10)  # max 10 concurrent LLM calls

async def analyze_pair(statute_item, policy_matches):
    async with sem:
        return await call_llm(statute_item, policy_matches)

tasks = [analyze_pair(s, p) for s, p in pairs]
results = await asyncio.gather(*tasks, return_exceptions=True)
```

### Retry & Failure Handling

- Retry LLM calls up to 2 times on transient errors (timeout, 5xx).
- On persistent failure or malformed JSON output, set `analysis_failed: true`.
- Failed items are excluded from summary denominators per BR-GA-4.

### Token Budget

For each LLM call, the input is roughly: system prompt (~300 tokens) + statute text (~200-500 tokens) + concatenated policy matches (~500-2000 tokens). Keep total input under the model's context window. If concatenated policy matches exceed a budget (e.g., 3000 tokens), truncate to the highest-scoring matches.

---

## 5. Requested Storage Changes

### New Collections

**`compliance_results`** — Stores gap analysis outputs.
```
{
  _id: ObjectId,
  policy_document_id: string,
  company_name: string,
  applicable_jurisdictions: [string],
  analyzed_at: string (ISO8601),
  gaps: [GapItem],
  summary: {total, addressed, missing, conflicts, analysis_failures},
  retrieval_metadata: {statute_items_considered, statute_pairs_matched},
  run_type: string,
  version: string
}
```
Indexes: `policy_document_id`, `analyzed_at`, compound `(policy_document_id, version)`.

**`compliance_run_log`** — Audit trail.
```
{
  _id: ObjectId,
  policy_document_id: string,
  run_type: string,
  statute_item_ids: [ObjectId],
  ran_at: string (ISO8601),
  summary: object
}
```
Index: `policy_document_id`, `ran_at`.

### Changes to Existing Collections

**`statute_embeddings` and `statute_sub_embeddings`:**
- **Add `jurisdiction` field** (string, e.g., "CA", "VA") if not already present. This is required for filtering by jurisdiction during retrieval. Index it.
- **Add `statute_reference` field** (string, e.g., "CCPA §1798.100(a)") for human-readable identification in gap items.

**`policy_embeddings` and `policy_sub_embeddings`:**
- Ensure `policy_document_id` field exists and is indexed (for the vector search pre-filter).
- **v1 only:** Ensure `parent_chunk_id` exists on `policy_sub_embeddings` pointing to the corresponding `policy_embeddings` record.

**`statute_sub_embeddings` (v1 only):**
- Ensure `parent_chunk_id` exists pointing to the corresponding `statute_embeddings` record (for enclosing context retrieval).

### Vector Search Index Definitions

If not already created, define Atlas vector search indexes:

```json
{
  "name": "policy_embeddings_vector_index",
  "type": "vectorSearch",
  "definition": {
    "fields": [
      {
        "type": "vector",
        "path": "embedding",
        "numDimensions": 1536,
        "similarity": "cosine"
      },
      {
        "type": "filter",
        "path": "policy_document_id"
      }
    ]
  }
}
```

Same pattern for `policy_sub_embeddings_vector_index`.

---

## 6. Accuracy Considerations

### Where Accuracy Is Won or Lost

1. **Retrieval quality (vector search):** If the vector search doesn't surface the right policy section for a given statute requirement, the LLM will incorrectly say "missing." Mitigation: use a generous top-k (5), tune the embedding model, and consider a **two-stage retrieval** — first vector search, then an LLM reranker that scores each candidate match for relevance before the main analysis call.

2. **Citation binding strictness:** The LLM often paraphrases or slightly modifies quotes. The normalization step (whitespace + case) helps, but consider adding a **fuzzy fallback**: if exact match fails, check if a high-similarity substring exists (e.g., Levenshtein ratio > 0.95). If found, use the actual substring from the policy as the quote and flag it as `quote_corrected: true`.

3. **LLM classification quality:** The fine-tuned model should outperform a generic model here. Key risk: the LLM being too generous with "addressed" when the policy only vaguely touches the topic. The prompt explicitly instructs that partial/ambiguous = missing, but this benefits from eval tuning.

4. **Statute granularity:** A single statute chunk may contain multiple distinct requirements. Consider an LLM pre-processing step that splits compound statute chunks into individual requirements before the main analysis loop. This improves precision at the cost of more LLM calls.

### Optional: LLM Reranker (Recommended)

After vector search returns top-k policy matches for a statute item, use a lightweight LLM call to rerank them:

```
Given this statutory requirement: {STATUTE_TEXT}
Rank these policy sections by relevance (1 = most relevant):
1. {POLICY_MATCH_1}
2. {POLICY_MATCH_2}
...
Return only the top 3 most relevant, or "NONE" if none are relevant.
```

This filters out false-positive vector matches before the main analysis, reducing noise and improving classification accuracy. Cost: ~1 additional cheap LLM call per statute item.

---

## 7. Process Diagram

```
                    ┌─────────────────────┐
                    │   API Request        │
                    │   policy_document_id │
                    │   jurisdictions      │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Step 0: Validate    │
                    │  - policy exists?    │
                    │  - policy indexed?   │
                    │  - resolve jurisd.   │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Step 1: Get Statute │
                    │  Items by jurisd.   │
                    │  (chunk or subchunk) │
                    └──────────┬──────────┘
                               │
              ┌────────────────▼────────────────┐
              │  Step 2: For EACH statute item   │
              │  ┌────────────────────────────┐  │
              │  │ Vector search policy embeds │  │
              │  │ Filter: policy_document_id  │  │
              │  │ Top-k matches + threshold   │  │
              │  └─────────────┬──────────────┘  │
              │                │                  │
              │  ┌─────────────▼──────────────┐  │
              │  │ (Optional) LLM Reranker     │  │
              │  └─────────────┬──────────────┘  │
              │                │                  │
              │  ┌─────────────▼──────────────┐  │
              │  │ Step 3: LLM Classification  │  │
              │  │ → status, quotes, summary   │  │
              │  └─────────────┬──────────────┘  │
              │                │                  │
              │  ┌─────────────▼──────────────┐  │
              │  │ Step 4: Citation Binding    │  │
              │  │ Verify policy_quote exists  │  │
              │  │ in actual policy text       │  │
              │  └────────────────────────────┘  │
              └────────────────┬────────────────┘
                               │ (parallel, semaphore-limited)
                    ┌──────────▼──────────┐
                    │  Step 5: Aggregate   │
                    │  summary + metadata  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Step 6: Persist     │
                    │  (if save_results)   │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Return Response     │
                    └─────────────────────┘
```
