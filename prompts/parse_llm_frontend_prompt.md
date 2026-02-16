# Parse-LLM Frontend Prompt Specification

This document provides everything the frontend needs to recreate the backend prompt functionality for the `POST /parse-llm` endpoint. The frontend passes application-specific instructions via the `parse_prompt` parameter. The backend retains only generic parsing guardrails.

---

## API Endpoint

**`POST /parse-llm`**

### Request Body (JSON)

| Parameter     | Type   | Required | Description                                                                 |
|---------------|--------|----------|-----------------------------------------------------------------------------|
| `database`    | string | Yes      | MongoDB database name containing the documents to parse                     |
| `collection`  | string | Yes      | MongoDB collection name containing the documents to parse                   |
| `parse_prompt`| string | Yes      | Application-specific instructions for parsing (see below)                  |
| `document_id` | string | No       | Optional. If provided, only that document is parsed; otherwise all documents |

### Response (200 OK)

```json
{
  "parsed_doc": [
    {
      "section": "§ 1798.100",
      "code_name": "Civil Code",
      "jurisdiction": "California",
      "parsed_header_text": "Right to know what personal information is collected",
      "parsed_text": "Full statutory text for this section..."
    }
  ]
}
```

---

## What the Backend Does (Generic Guardrails)

The backend:

1. Reads documents from `database.collection` (or a single document if `document_id` is provided)
2. Concatenates all `text` attributes with `\n\n` between documents
3. Splits into lines and prepends 1-indexed line numbers: `1: first line`, `2: second line`, etc.
4. Sends the numbered text to the LLM with a system prompt (generic parsing instructions) and user message
5. Uses the `identify_sections` tool to extract structured sections
6. Extracts text between section boundaries using line numbers and returns `parsed_doc`

---

## Backend System Prompt (Generic – Do Not Duplicate)

The backend uses this system prompt. The frontend does **not** send this; it is fixed in the backend. Documented here for reference:

```
You are a legal text parser specializing in statutory interpretation.

Task:
Parse the provided legal statute into a structured, machine-readable format by
identifying and extracting its hierarchical sections.

Instructions:
- Preserve the original statutory language verbatim.
- Do NOT summarize, paraphrase, or interpret the text.
- Do NOT infer missing structure; rely only on explicit markers in the text.
- Maintain the original order of sections.

Identify and extract the following elements when present:
- Jurisdiction (e.g., California, United States)
- Code name (e.g., Civil Code, Health and Safety Code)
- Section citation (e.g., § 1798.100)
- Title or Act name
- Chapter or Division
- Article
- Section number
- Subsection (e.g., (a), (b), (1), (A))
- Heading or caption
- Body text

Output Format:
Return a JSON array where each object represents the smallest logical statutory
unit (typically a section or subsection).

Each object must include:
- "jurisdiction": string
- "code_name": string
- "level": one of ["title", "chapter", "article", "section", "subsection"]
- "identifier": the official number or label (e.g., "§ 1798.100", "(a)",
  "Chapter 3")
- "heading": the heading text if present, otherwise null
- "text": the full statutory text for that unit
- "parent_identifier": the identifier of the immediately enclosing unit, or
  null if top-level

The document includes line numbers at the start of each line. Use those line
numbers to identify section boundaries. Call the identify_sections tool and
return an array of sections with:
- section
- code_name
- jurisdiction
- parsed_header_text
- start_line (1-indexed).
```

---

## User Message Template (Backend)

The backend constructs the user message as follows. The `parse_prompt` is inserted into the `<parse_prompt>` block:

```
Analyze the following document and identify the section boundaries according to the parsing instructions.

<document>
{numbered_text}
</document>

Exclude irrelevant sections from the results.
Any sections that do not pertain to company privacy policy should be omitted.

<parse_prompt>
{parse_prompt}
</parse_prompt>

Use the identify_sections tool to report the sections you identified.
```

**Note:** The lines "Exclude irrelevant sections from the results." and "Any sections that do not pertain to company privacy policy should be omitted." are application-specific. The backend will remove these; the frontend must include equivalent instructions in `parse_prompt` when recreating this behavior.

---

## What to Pass as `parse_prompt`

The frontend should build `parse_prompt` to include all application-specific instructions. To replicate the current backend behavior, include:

```
Exclude irrelevant sections from the results.
Any sections that do not pertain to company privacy policy should be omitted.
```

For other use cases, the frontend may provide different instructions, for example:

- Filtering criteria (e.g., only privacy-related sections, only certain jurisdictions)
- Section prioritization or ordering preferences
- Additional context about the document type or intended use

The `parse_prompt` is inserted into the user message inside `<parse_prompt>...</parse_prompt>` tags, so the LLM receives it as explicit instructions.

---

## Tool Schema (Backend – For Reference)

The backend uses this tool. The frontend does not define tools; the backend handles tool use.

**Tool name:** `identify_sections`

**Description:** Report the identified sections in the document with their starting line numbers

**Input schema:**

```json
{
  "type": "object",
  "properties": {
    "sections": {
      "type": "array",
      "description": "Array of identified sections",
      "items": {
        "type": "object",
        "properties": {
          "section": {
            "type": "string",
            "description": "The formal section citation (e.g., § 1798.100)"
          },
          "code_name": {
            "type": "string",
            "description": "The legal code name (e.g., Civil Code)"
          },
          "jurisdiction": {
            "type": "string",
            "description": "The jurisdiction for this section (e.g., California)"
          },
          "parsed_header_text": {
            "type": "string",
            "description": "The header, title, or identifying text for this section"
          },
          "start_line": {
            "type": "integer",
            "description": "The 1-indexed line number where this section begins"
          }
        },
        "required": ["section", "code_name", "jurisdiction", "parsed_header_text", "start_line"]
      }
    }
  },
  "required": ["sections"]
}
```

---

## Document Format

- Documents are concatenated with `\n\n` between them
- Each line is prefixed with `{line_number}: ` (1-indexed)
- Example:
  ```
  1: § 1798.100. (a) A consumer shall have the right to request that a business
  2: disclose the categories and specific pieces of personal information that
  3: it has collected about the consumer.
  4:
  5: § 1798.105. (a) A consumer shall have the right to request that a business
  6: delete any personal information about the consumer which the business has
  ```

---

## Frontend Implementation Checklist

1. **Build `parse_prompt`** – Include all application-specific instructions (e.g., filtering, exclusions, context).
2. **Call the API** – `POST /parse-llm` with `database`, `collection`, `parse_prompt`, and optionally `document_id`.
3. **Handle response** – `parsed_doc` is an array of `{ section, code_name, jurisdiction, parsed_header_text, parsed_text }`.
4. **No parameter changes** – Use the existing API contract; only the content of `parse_prompt` changes.

---

## Summary of Current Prompts Used for Parsing

| Location        | Content                                                                 |
|----------------|-------------------------------------------------------------------------|
| System prompt  | Legal text parser role, task, instructions, output format, tool usage  |
| User message   | "Analyze the following document...", `<document>`, exclusion rules, `<parse_prompt>`, tool instruction |
| `parse_prompt` | Passed by frontend – application-specific (e.g., "Exclude irrelevant sections...", "Any sections that do not pertain to company privacy policy should be omitted.") |
