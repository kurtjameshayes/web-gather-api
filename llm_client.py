"""LLM client wrapper and prompt templating."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Protocol

import anthropic

from compliance_config import ComplianceConfig
from compliance_utils import extract_json_block
from vector_retriever import StatuteCandidate

logger = logging.getLogger("policy-compliance")


async def _run_in_thread(func, *args, **kwargs):
    """Run sync function in a thread (Python 3.8 compat: asyncio.to_thread added in 3.9)."""
    if hasattr(asyncio, "to_thread"):
        return await asyncio.to_thread(func, *args, **kwargs)
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


RETRIEVAL_PROMPT_TEMPLATE = """You are a privacy law analyst. Compare the following policy section to the candidate statute excerpts. For each statute excerpt, say whether it applies and why. Then decide overall compliance for the policy section.

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

Output only valid JSON with no surrounding text. Use exactly these keys:
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
"""

# ----- Compliance suite prompts -----

APPLICABILITY_PROMPT = """You are a privacy law analyst. Based on the following privacy policy text, list the US state codes (e.g. CA, VA, CO, CT) and "US federal" for which this policy is likely intended. Use explicit or implicit references in the text.

Output only valid JSON with this exact structure:
{
  "applicable_jurisdictions": ["CA", "VA", ...],
  "confidence": { "CA": 0.95, "VA": 0.8, ... }
}
Use two-letter state codes. confidence values are between 0 and 1."""

GAP_CHECK_PROMPT = """You are a privacy law analyst. A statute chunk states a disclosure or obligation. Determine if the policy text addresses it, and if there is any conflict.

Statute chunk (requirement):
<<<STATUTE_CHUNK>>>

Policy text:
<<<POLICY_TEXT>>>

Answer:
(a) Is the required disclosure/obligation in the statute addressed in the policy? (yes/no)
(b) If yes, quote the exact policy phrase that addresses it. If no, use null.
(c) Is there any statement in the policy that conflicts with the statute? (yes/no)
(d) If conflict, briefly describe it. Otherwise null.

Output only valid JSON with this exact structure:
{
  "addressed": true or false,
  "policy_quote": "exact phrase from policy or null",
  "missing": true or false,
  "conflict": true or false,
  "conflict_description": "brief description or null"
}
"""

REQUIREMENT_EXTRACTION_PROMPT = """You are a privacy law analyst. From the following statute chunk, list each discrete privacy/consumer right or obligation. One per item: short label and one-sentence description.

Statute chunk:
<<<STATUTE_CHUNK>>>

Output only valid JSON:
{
  "requirements": [
    { "label": "Short label", "description": "One sentence." },
    ...
  ]
}
"""

STRICTNESS_PROMPT = """You are a privacy law analyst. For the following canonical requirement, compare how each jurisdiction formulates it. Which jurisdiction imposes the strictest or most expansive requirement?

Canonical requirement: <<<CANONICAL_ID>>>

Jurisdiction formulations:
<<<JURISDICTION_DESCRIPTIONS>>>

Output only valid JSON:
{
  "strictest_jurisdiction": "CA",
  "strictest_description": "...",
  "other_jurisdictions": [ { "jurisdiction": "VA", "description": "..." }, ... ]
}
If two are equally strict, set strictest_jurisdiction to the first and include both in other_jurisdictions with note.
"""

POLICY_ALIGNMENT_PROMPT = """You are a privacy law analyst. For one requirement, jurisdiction A and B have different formulations. Given the policy excerpt, does it satisfy both, only the strictest, or conflict with one?

Requirement in jurisdiction A: <<<REQ_A>>>
Requirement in jurisdiction B: <<<REQ_B>>>
Policy excerpt: <<<POLICY_EXCERPT>>>

Output only valid JSON:
{
  "policy_alignment": "satisfies_all|satisfies_strictest_only|conflict_between_jurisdictions|not_provided|unclear",
  "policy_note": "Brief explanation or null"
}
"""


class LLMClient(Protocol):
    async def compare_section(
        self, section_id: str, section_text: str, candidates: List[StatuteCandidate]
    ) -> str:
        ...


def build_prompt(section_text: str, candidates: List[StatuteCandidate]) -> str:
    candidate_lines = []
    for idx, candidate in enumerate(candidates, start=1):
        candidate_lines.append(
            f"{idx}) [{candidate.statute_id}] {candidate.jurisdiction} {candidate.title}\n"
            f"{candidate.chunk_text}\n"
            f"score: {candidate.score:.4f}\n"
        )
    candidate_block = "\n".join(candidate_lines).strip() or "1) [STATUTE_ID] [JURISDICTION] [TITLE]\n[STATUTE_CHUNK_TEXT]\nscore: [SCORE]\n"
    prompt = RETRIEVAL_PROMPT_TEMPLATE.replace("<<<SECTION_TEXT>>>", section_text)
    prompt = prompt.replace(
        "1) [STATUTE_ID] [JURISDICTION] [TITLE]\n[STATUTE_CHUNK_TEXT]\nscore: [SCORE]\n\n... (repeat)",
        candidate_block,
    )
    return prompt


class AnthropicLLMClient:
    def __init__(self, api_key: str, config: ComplianceConfig) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = config.llm_model_name
        self._max_tokens = config.llm_max_tokens

    async def compare_section(
        self, section_id: str, section_text: str, candidates: List[StatuteCandidate]
    ) -> str:
        prompt = build_prompt(section_text, candidates)
        logger.info("LLM compare_section for section_id=%s candidates=%d", section_id, len(candidates))
        reminder = "\n\nOutput only valid JSON with no surrounding text."

        def run_call(extra: str = "") -> str:
            full_prompt = prompt + extra
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=[{"role": "user", "content": full_prompt}],
            )
            content = []
            for block in response.content:
                if block.type == "text":
                    content.append(block.text)
            return "".join(content)

        text = await _run_in_thread(run_call)
        raw = extract_json_block(text)
        if raw:
            try:
                json.loads(raw)
                return text
            except json.JSONDecodeError:
                pass
        text = await _run_in_thread(lambda: run_call(reminder))
        return text

    async def _call_json(self, prompt: str, retry_with_reminder: bool = True) -> Optional[Dict[str, Any]]:
        """Call LLM with prompt, parse JSON from response. Retry once with reminder if parse fails."""
        reminder = "\n\nOutput only valid JSON with no surrounding text."

        def run_call(extra: str = "") -> str:
            full_prompt = prompt + extra
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=[{"role": "user", "content": full_prompt}],
            )
            content = []
            for block in response.content:
                if block.type == "text":
                    content.append(block.text)
            return "".join(content)

        text = await _run_in_thread(run_call)
        raw = extract_json_block(text)
        if raw:
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                pass
        if retry_with_reminder:
            text = await _run_in_thread(lambda: run_call(reminder))
            raw = extract_json_block(text)
            if raw:
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    pass
        return None

    async def applicability(self, policy_text: str) -> Dict[str, Any]:
        """Return { applicable_jurisdictions: [...], confidence: { "CA": 0.95, ... } } or empty dict on failure."""
        prompt = APPLICABILITY_PROMPT + "\n\nPrivacy policy text:\n" + (policy_text[:8000] or "")
        out = await self._call_json(prompt)
        if not out or not isinstance(out.get("applicable_jurisdictions"), list):
            return {"applicable_jurisdictions": [], "confidence": {}}
        return {
            "applicable_jurisdictions": list(out["applicable_jurisdictions"]),
            "confidence": out.get("confidence") if isinstance(out.get("confidence"), dict) else {},
        }

    async def gap_check(
        self, statute_chunk_text: str, statute_reference: str, policy_text: str
    ) -> Dict[str, Any]:
        """Return { addressed, policy_quote, missing, conflict, conflict_description }. On parse failure return addressed=False, missing=True, conflict=False."""
        prompt = (
            GAP_CHECK_PROMPT.replace("<<<STATUTE_CHUNK>>>", statute_chunk_text[:3000])
            .replace("<<<POLICY_TEXT>>>", (policy_text or "")[:6000])
        )
        out = await self._call_json(prompt)
        if not out:
            return {
                "addressed": False,
                "policy_quote": None,
                "missing": True,
                "conflict": False,
                "conflict_description": None,
            }
        return {
            "addressed": bool(out.get("addressed")),
            "policy_quote": out.get("policy_quote") if out.get("policy_quote") else None,
            "missing": bool(out.get("missing", True)),
            "conflict": bool(out.get("conflict")),
            "conflict_description": out.get("conflict_description") if out.get("conflict_description") else None,
        }

    async def requirement_extraction(self, statute_chunk_text: str) -> Dict[str, Any]:
        """Return { requirements: [ { label, description }, ... ] }."""
        prompt = REQUIREMENT_EXTRACTION_PROMPT.replace("<<<STATUTE_CHUNK>>>", statute_chunk_text[:3000])
        out = await self._call_json(prompt)
        if not out or not isinstance(out.get("requirements"), list):
            return {"requirements": []}
        return {"requirements": [{"label": str(r.get("label", "")), "description": str(r.get("description", ""))} for r in out["requirements"]]}

    async def strictness_comparison(
        self, canonical_id: str, jurisdiction_descriptions: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Return { strictest_jurisdiction, strictest_description, other_jurisdictions }."""
        lines = []
        for d in jurisdiction_descriptions:
            j = d.get("jurisdiction", "")
            desc = d.get("description", "")
            lines.append(f"- {j}: {desc}")
        block = "\n".join(lines) if lines else "None"
        prompt = STRICTNESS_PROMPT.replace("<<<CANONICAL_ID>>>", canonical_id).replace("<<<JURISDICTION_DESCRIPTIONS>>>", block)
        out = await self._call_json(prompt)
        if not out:
            return {"strictest_jurisdiction": "", "strictest_description": "", "other_jurisdictions": []}
        others = out.get("other_jurisdictions") or []
        if not isinstance(others, list):
            others = []
        return {
            "strictest_jurisdiction": str(out.get("strictest_jurisdiction", "")),
            "strictest_description": str(out.get("strictest_description", "")),
            "other_jurisdictions": [{"jurisdiction": str(o.get("jurisdiction", "")), "description": str(o.get("description", ""))} for o in others],
        }

    async def policy_alignment(
        self, requirement_a: str, requirement_b: str, policy_excerpt: str
    ) -> Dict[str, Any]:
        """Return { policy_alignment: str, policy_note: str | null }."""
        prompt = (
            POLICY_ALIGNMENT_PROMPT.replace("<<<REQ_A>>>", requirement_a[:1500])
            .replace("<<<REQ_B>>>", requirement_b[:1500])
            .replace("<<<POLICY_EXCERPT>>>", (policy_excerpt or "")[:3000])
        )
        out = await self._call_json(prompt)
        if not out:
            return {"policy_alignment": "unclear", "policy_note": None}
        align = out.get("policy_alignment", "unclear")
        if align not in ("satisfies_all", "satisfies_strictest_only", "conflict_between_jurisdictions", "not_provided", "unclear"):
            align = "unclear"
        return {
            "policy_alignment": align,
            "policy_note": out.get("policy_note") if out.get("policy_note") else None,
        }


class StubLLMClient:
    """Deterministic stub for tests."""

    def __init__(self, response_text: str) -> None:
        self._response_text = response_text

    async def compare_section(
        self, section_id: str, section_text: str, candidates: List[StatuteCandidate]
    ) -> str:
        return self._response_text
