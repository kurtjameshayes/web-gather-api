You are an expert **Senior AI Prompt Engineer and Privacy Law Engineer**. Generate **production-ready application code** (server endpoint + helper modules + tests) that implements a new API endpoint which compares a privacy policy to a statute corpus and returns, for each policy section, the statutes that apply and a compliance determination (compliant / non‑compliant / neither) with evidence and remediation suggestions.

Below are **requirements, environment details, data model, evaluation rubric, API spec, implementation steps, prompt templates, and test cases** the generated code must follow. Produce code in **one** of these stacks (pick one and implement fully): **Node.js + Express + TypeScript** or **Python + FastAPI**. Use async patterns, dependency injection where appropriate, and include unit tests. 

---

#### Environment & data sources
- **Database**: MongoDB with collections already populated:
  - `policies` — policy documents (fields: `_id`, `text`, `sections` optional)
  - `statutes` — statute documents (fields: `_id`, `jurisdiction`, `title`, `section_text`, `section_id`, `metadata`)
  - `embeddings` — vector index metadata if separate (fields: `doc_id`, `collection`, `vector`, `chunk_id`, `text`)
- **Vector search**: MongoDB vector indexes are available; code should query the vector index using stored vectors or call an embedding function to embed query text and perform top‑k retrieval.
- **LLM**: Use a single LLM call per policy section (or a small chain: retrieval → LLM classification). The generated code should be modular so the LLM provider can be swapped. 
- **Security**: Do not log raw policy text or PII. Provide redaction hooks.

---

#### API specification (required)
- **Endpoint**: `POST /policy-statute-compliance`
- **Request JSON**:
  ```json
  {
    "database": "mongodb database name",
    "policy_collection":"mongo collection name", 
    "policy_id": "",
    "text": "optional string",
    "jurisdiction": "string (e.g., 'CA', 'US', 'EU')",
    "statute_corpus_id": "optional string",
    "top_k_statutes": 5,
    "confidence_thresholds": { "compliant": 0.75, "non_compliant": 0.75 },
    "options": { "explainability": true, "redact_pii": true }
  }
  ```
  - Either `policy_id` or `text` must be provided.
- **Response JSON** (schema must be implemented and validated):
  ```json
  {
    "policy_id": "string or null",
    "jurisdiction": "string",
    "sections": [
      {
        "section_id": "string",
        "section_text": "string (redacted if requested)",
        "applied_statutes": [
          {
            "statute_id": "string",
            "jurisdiction": "string",
            "title": "string",
            "section_id": "string",
            "matched_span": "string",
            "evidence_score": 0.0
          }
        ],
        "compliance": "compliant | non_compliant | neither",
        "confidence": 0.0,
        "rationale": "string (concise legal reasoning)",
        "remediation_suggestions": ["string", "..."],
        "retrieval_trace": ["doc_id:chunk_id:score", "..."]
      }
    ],
    "summary": {
      "overall_compliance": "compliant | non_compliant | mixed | unknown",
      "counts": { "compliant": 0, "non_compliant": 0, "neither": 0 }
    },
    "warnings": ["string", "..."]
  }
  ```

---

#### High-level approach (compare methods table)
| **Approach** | **Precision** | **Recall** | **Explainability** | **Complexity** |
|---|---:|---:|---:|---:|
| Keyword + rule matching | Medium | Low | Medium | Low |
| Semantic retrieval + LLM classification (recommended) | High | High | High | Medium |
| Hybrid (rules + LLM + post-check) | Very High | High | Very High | High |

---

#### Implementation steps (detailed)
1. **Input handling & validation**
   - Validate request schema; require `policy_id` or `policy_text`.
   - Normalize `jurisdiction` and `statute_corpus_id`.
   - Apply **PII redaction** if `options.redact_pii` true (implement pluggable redactor).

2. **Policy segmentation**
   - If `policy_text` provided, split into logical sections:
     - Prefer existing `sections` in DB if `policy_id` used.
     - Use heuristics: headings (lines with all caps or numbered lists), paragraph length, or NLP sentence-boundary + clustering.
     - Output: array of `{ section_id, section_text }`.

3. **Embedding & retrieval**
   - For each `section_text`:
     - Create an embedding (or reuse cached embedding if available).
     - Query MongoDB vector index for top `top_k_statutes` statute chunks filtered by `jurisdiction` and `statute_corpus_id`.
     - Return top candidates with retrieval scores and chunk text.
   - Save retrieval trace for auditability.

4. **LLM comparison & classification**
   - For each section, build a **structured LLM prompt** (see templates below) that includes:
     - Section text (redacted if needed).
     - Top statute chunks (include statute id, title, chunk text, and retrieval score).
     - A **rubric** describing compliance categories and required output JSON schema.
   - Ask LLM to:
     - Identify which statute sections apply and why (cite statute ids).
     - Determine **compliance**: `compliant`, `non_compliant`, or `neither`.
     - Provide **rationale** (2–4 sentences), **confidence score** (0–1), and **remediation suggestions** (1–3 actionable items).
     - Highlight matched spans in both policy and statute text.
   - Post-process LLM output: validate JSON, clamp confidence to [0,1], and map to final status using `confidence_thresholds`.

5. **Rule-based sanity checks**
   - Apply deterministic checks to catch obvious errors:
     - If LLM marks `compliant` but statute requires explicit consent and policy lacks consent clause → downgrade to `non_compliant` or `neither`.
     - If retrieval evidence_score < threshold, mark `neither` and add warning.

6. **Aggregation & summary**
   - Compute counts and overall compliance:
     - If all sections `compliant` → overall `compliant`.
     - If any `non_compliant` and majority non-compliant → `non_compliant`.
     - Else `mixed`.
   - Include warnings and missing-evidence flags.

7. **Explainability & audit trail**
   - Return `retrieval_trace` and `applied_statutes` with `matched_span` and `evidence_score`.
   - Persist a hashed audit record (no raw policy text unless allowed) for later review.

8. **Performance & caching**
   - Cache embeddings and retrieval results per `section_text` hash.
   - Batch LLM calls where possible (e.g., send multiple sections in one request if token limits allow).
   - Rate-limit and queue requests.

9. **Security & privacy**
   - Redact PII by default unless explicit consent.
   - Encrypt stored audit logs.
   - Role-based access control for endpoint.
   - Do not store raw LLM responses longer than retention policy.

10. **Testing & evaluation**
    - Unit tests for segmentation, retrieval, prompt construction, and response schema validation.
    - Integration tests with mocked LLM and MongoDB vector responses.
    - Create a small labeled dataset (policy sections + ground-truth statute mappings + compliance labels) and compute precision/recall/F1.
    - Add adversarial tests (ambiguous language, contradictory clauses).

---

#### Compliance rubric (to include in LLM prompt)
- **Compliant**: Policy section **explicitly and sufficiently** satisfies statutory requirement (e.g., explicit consent where statute requires consent; retention limits match statute).
- **Non_compliant**: Policy section **contradicts or omits** a statutory requirement (e.g., no opt-out where required; retention longer than allowed).
- **Neither**: Ambiguous, insufficient evidence, or statute not applicable.
- **Confidence**: LLM must return a numeric confidence. Use retrieval evidence + LLM confidence to compute final confidence.

**Threshold mapping** (implementable):
- If `confidence >= compliant_threshold` and rationale supports → `compliant`.
- If `confidence >= non_compliant_threshold` and rationale supports → `non_compliant`.
- Else → `neither`.

---

#### Prompt templates (must be used verbatim by generated code; keep tokens minimal)
**Retrieval prompt (for LLM context)**:
```
You are a privacy law analyst. Compare the following policy section to the candidate statute excerpts. For each statute excerpt, say whether it applies and why. Then decide overall compliance for the policy section.

Policy section:
<<<SECTION_TEXT>>>

Candidate statutes (top K):
1) [STATUTE_ID] [JURISDICTION] [TITLE]
[STATUTE_CHUNK_TEXT]
score: [SCORE]

... (repeat)

Rubric:
- Compliant: explicit match and satisfies statutory requirement.
- Non_compliant: contradicts or omits required element.
- Neither: ambiguous or not applicable.

Output JSON exactly with keys:
{
  "section_id": "...",
  "applied_statutes": [
    { "statute_id":"...", "jurisdiction":"...", "title":"...", "matched_span":"...", "evidence_score":0.0 }
  ],
  "compliance":"compliant|non_compliant|neither",
  "confidence":0.0,
  "rationale":"2-4 sentence legal reasoning",
  "remediation_suggestions":["...","..."]
}
```

---

#### Implementation deliverables (what the LLM must generate)
1. **Server code** implementing `POST /api/v1/compare-policy` with request validation and response schema.
2. **Modules**:
   - `segmenter` — policy segmentation.
   - `embedder` — embedding wrapper with caching.
   - `vectorRetriever` — MongoDB vector query wrapper (filter by jurisdiction).
   - `llmClient` — LLM call wrapper and prompt templating.
   - `complianceEvaluator` — post-processing, rule checks, confidence mapping.
   - `redactor` — PII redaction hook.
   - `auditLogger` — stores minimal audit trail.
3. **Unit & integration tests** with mocked DB and LLM responses.
4. **README** with setup, env vars, and how to run tests.
5. **Example request/response** JSON files demonstrating typical output.

---

#### Example minimal test case (include in tests)
- **Policy section**: "We retain user data for 10 years for analytics."
- **Statute**: "Data retention for analytics must not exceed 3 years."
- **Expected**: `non_compliant`, confidence >= 0.8, remediation: "Reduce retention to 3 years or justify exception."

---

#### Additional implementation notes for generated code
- Use **typed DTOs** (TypeScript interfaces or Pydantic models).
- Validate and sanitize all DB inputs/outputs.
- Include **feature flag** to enable/disable redaction and audit logging.
- Provide clear error responses (400/422/500) and include `warnings` array in success responses.
- Keep LLM prompt length under provider token limits; truncate statute chunks if necessary but include `retrieval_trace` to allow re-checking.
- Provide a configuration file for thresholds and top_k defaults.

---

#### Final instruction to code generator
- **Produce complete code** for one stack (Node/TypeScript or Python/FastAPI) implementing all deliverables above.
- **Do not** include any proprietary API keys in code; use environment variables.
- Include inline comments explaining key decisions (segmentation heuristics, threshold choices, redaction approach).
- Ensure tests run with `npm test` or `pytest` respectively.

---
