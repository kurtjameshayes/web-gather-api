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
(b) If yes, quote the exact policy phrase that addresses it in policy_quote. If no, use null.
(c) Is there any statement in the policy that conflicts with the statute? (yes/no)
(d) If conflict, briefly describe it in conflict_description. If the policy contains a phrase that conflicts, also quote that exact phrase in policy_quote.
(e) policy_quote must be a verbatim substring of the policy text (for addressed or conflict cases).

Output only valid JSON with this exact structure:
{
  "addressed": true or false,
  "policy_quote": "exact phrase from policy or null",
  "missing": true or false,
  "conflict": true or false,
  "conflict_description": "brief description or null"
}
"""

SUBCHUNK_GAP_CHECK_PROMPT = """You are a privacy law analyst. Compare the following statute subchunk to the policy subchunk for compliance. Use the enclosing chunk text for context, but base your determination on the subchunk-to-subchunk comparison.

Statute subchunk (focus here):
<<<STATUTE_SUBCHUNK>>>

Statute enclosing chunk (context):
<<<STATUTE_CHUNK>>>

Policy subchunk (focus here):
<<<POLICY_SUBCHUNK>>>

Policy enclosing chunk (context):
<<<POLICY_CHUNK>>>

Answer:
(a) Is the required disclosure/obligation in the statute subchunk addressed in the policy subchunk? (yes/no)
(b) If yes, quote the exact policy phrase that addresses it in policy_quote. If no, use null.
(c) Is there any statement in the policy subchunk that conflicts with the statute subchunk? (yes/no)
(d) If conflict, briefly describe it in conflict_description. If the policy contains a phrase that conflicts, also quote that exact phrase in policy_quote.
(e) policy_quote must be a verbatim substring of the policy subchunk or policy enclosing chunk (for addressed or conflict cases).

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

CITATION_PROMPT = """You are a privacy law analyst. Does this policy excerpt cite or align with this statute? If yes, provide the exact policy phrase that corresponds and an optional statute snippet.

Policy excerpt:
<<<POLICY_EXCERPT>>>

Statute (<<<JURISDICTION>>>):
<<<STATUTE_CHUNK>>>

Output only valid JSON:
{
  "alignment": true or false,
  "policy_quote": "exact phrase from policy or null",
  "statute_excerpt": "optional short snippet from statute or null"
}
"""

SUGGEST_POLICY_PROMPT = """You are a privacy law compliance editor. Your task is to rewrite or add text to a privacy policy so that it becomes compliant with the given statute.

You are provided with:
1. The current policy text.
2. A gap analysis finding that describes a compliance gap.
3. The gap analysis match status (e.g. "missing", "conflict", "partial").
4. The authoritative statute text that the policy must comply with.

Instructions:
- Analyze the gap analysis finding to understand exactly what the policy is missing or where it conflicts with the statute.
- Reference the statute text as the authoritative requirement.
- Rewrite or extend the policy text to address the gap while preserving all existing compliant language. Do not remove or weaken language that is already compliant.
- If the gap is about missing language, add the necessary provisions in the most natural location within the policy.
- If the gap is a conflict, revise the conflicting language to align with the statute.
- If the gap is partial, strengthen the existing language to fully satisfy the requirement.

Current policy text:
<<<POLICY_TEXT>>>

Gap analysis finding:
<<<GAP_ANALYSIS_TEXT>>>

Gap analysis match status:
<<<GAP_ANALYSIS_MATCH>>>

Statute text (authoritative requirement):
<<<STATUTE_TEXT>>>

Output only valid JSON with this exact structure:
{
  "suggested_policy_text": "The full revised policy text with all modifications applied.",
  "modifications_description": "A plain-language summary of every change made and why each change was necessary to achieve compliance."
}
"""

RISK_ASSESSMENT_PROMPT = """You are a privacy law analyst. From the following policy text and statute requirements, fill a DPIA-style risk assessment. List processing purposes, data categories, risks, mitigations, and any gaps between policy and statute.

Policy text:
<<<POLICY_TEXT>>>

Statute requirements (by jurisdiction):
<<<STATUTE_SUMMARY>>>

Output only valid JSON with this exact structure:
{
  "processing_purposes": ["purpose 1", "purpose 2", ...],
  "data_categories": ["category 1", "category 2", ...],
  "risks": [{"description": "...", "severity": "low|medium|high", "mitigation": "..."}, ...],
  "mitigations": ["mitigation 1", ...],
  "gaps_from_statute": [{"requirement": "...", "jurisdiction": "...", "status": "missing|addressed|conflict"}, ...]
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
        self._config = config
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

    async def _call_json(self, prompt: str, retry_with_reminder: bool = True, max_tokens: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Call LLM with prompt, parse JSON from response. Retry once with reminder if parse fails."""
        reminder = "\n\nOutput only valid JSON with no surrounding text."
        tokens = max_tokens or self._max_tokens
        # region agent log
        _dbg_stop_reason = [None]
        # endregion

        def run_call(extra: str = "") -> str:
            full_prompt = prompt + extra
            response = self._client.messages.create(
                model=self._model,
                max_tokens=tokens,
                messages=[{"role": "user", "content": full_prompt}],
            )
            # region agent log
            _dbg_stop_reason[0] = getattr(response, "stop_reason", None)
            # endregion
            content = []
            for block in response.content:
                if block.type == "text":
                    content.append(block.text)
            return "".join(content)

        text = await _run_in_thread(run_call)
        raw = extract_json_block(text)
        # region agent log
        import time as _t; open("/Users/kurthayes/Dev/AI/web-gather-api/.cursor/debug-4b665c.log","a").write(json.dumps({"sessionId":"4b665c","hypothesisId":"A,B","location":"llm_client.py:_call_json:first_call","message":"first LLM call result","data":{"text_len":len(text),"raw_extracted":raw is not None,"raw_len":len(raw) if raw else 0,"stop_reason":_dbg_stop_reason[0],"max_tokens":tokens,"text_tail":text[-200:] if text else "","prompt_len":len(prompt)},"timestamp":int(_t.time()*1000)})+"\n")
        # endregion
        if raw:
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                # region agent log
                open("/Users/kurthayes/Dev/AI/web-gather-api/.cursor/debug-4b665c.log","a").write(json.dumps({"sessionId":"4b665c","hypothesisId":"A","location":"llm_client.py:_call_json:json_decode_fail","message":"JSON decode failed on first call","data":{"raw_head":raw[:300] if raw else "","raw_tail":raw[-300:] if raw else ""},"timestamp":int(_t.time()*1000)})+"\n")
                # endregion
                pass
        if retry_with_reminder:
            text = await _run_in_thread(lambda: run_call(reminder))
            raw = extract_json_block(text)
            # region agent log
            open("/Users/kurthayes/Dev/AI/web-gather-api/.cursor/debug-4b665c.log","a").write(json.dumps({"sessionId":"4b665c","hypothesisId":"A,B","location":"llm_client.py:_call_json:retry","message":"retry LLM call result","data":{"text_len":len(text),"raw_extracted":raw is not None,"raw_len":len(raw) if raw else 0,"stop_reason":_dbg_stop_reason[0],"max_tokens":tokens,"text_tail":text[-200:] if text else ""},"timestamp":int(_t.time()*1000)})+"\n")
            # endregion
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

    async def gap_check_v3(
        self,
        statute_chunk_text: str,
        policy_chunk_text: str,
        prompt_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """V3 gap analysis: Returns status, policy_quote, statute_quote, requirement_summary, conflict_description, confidence.

        Used for gap analysis v3 with citation binding validation against full policy text.
        """
        path = prompt_path or getattr(
            self._config, "gap_analysis_v3_prompt_path", "prompts/gap_analysis_v3.yaml"
        )
        from prompt_loader import load_prompt_yaml, render_prompt

        cfg = load_prompt_yaml(path)
        prompt = render_prompt(
            cfg["prompt"],
            STATUTE_CHUNK=statute_chunk_text or "",
            POLICY_CHUNK=policy_chunk_text or "",
        )
        out = await self._call_json(prompt)
        if not out:
            return {
                "status": "missing",
                "policy_quote": None,
                "statute_quote": None,
                "requirement_summary": "Requirement",
                "conflict_description": None,
                "confidence": "low",
                "_analysis_failed": True,
            }
        status = (out.get("status") or "missing").lower()
        if status not in ("addressed", "missing", "conflict", "partial", "ambiguous"):
            status = "missing"
        return {
            "status": status,
            "policy_quote": out.get("policy_quote") if out.get("policy_quote") else None,
            "statute_quote": out.get("statute_quote") or "",
            "requirement_summary": (out.get("requirement_summary") or "Requirement").strip(),
            "conflict_description": out.get("gap_description") or out.get("conflict_description") or None,
            "confidence": (out.get("confidence") or "low").lower()
            if out.get("confidence") in ("high", "medium", "low")
            else "low",
            "_analysis_failed": False,
        }

    async def gap_check_v4(
        self,
        reference_context: str,
        statutory_requirement: str,
        policy_text: str,
        prompt_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """V4 gap analysis: Returns status, policy_quote, statute_quote, requirement_summary, conflict_description, confidence.

        Uses REFERENCE_CONTEXT (definitions, applicability) + STATUTORY_REQUIREMENT + POLICY_TEXT.
        Same output schema as gap_check_v3.
        """
        path = prompt_path or getattr(
            self._config, "gap_analysis_v4_prompt_path", "prompts/gap_analysis_v4.yaml"
        )
        from prompt_loader import load_prompt_yaml, render_prompt

        cfg = load_prompt_yaml(path)
        prompt = render_prompt(
            cfg["prompt"],
            REFERENCE_CONTEXT=reference_context or "",
            STATUTORY_REQUIREMENT=statutory_requirement or "",
            POLICY_TEXT=policy_text or "",
        )
        out = await self._call_json(prompt)
        if not out:
            return {
                "status": "missing",
                "policy_quote": None,
                "statute_quote": None,
                "requirement_summary": "Requirement",
                "conflict_description": None,
                "confidence": "low",
                "_analysis_failed": True,
            }
        status = (out.get("status") or "missing").lower()
        if status not in ("addressed", "missing", "conflict", "partial", "ambiguous"):
            status = "missing"
        return {
            "status": status,
            "policy_quote": out.get("policy_quote") if out.get("policy_quote") else None,
            "statute_quote": out.get("statute_quote") or "",
            "requirement_summary": (out.get("requirement_summary") or "Requirement").strip(),
            "conflict_description": out.get("gap_description") or out.get("conflict_description") or None,
            "confidence": (out.get("confidence") or "low").lower()
            if out.get("confidence") in ("high", "medium", "low")
            else "low",
            "_analysis_failed": False,
        }

    async def gap_check_chunks(
        self,
        statute_chunk_text: str,
        policy_chunk_text: str,
        prompt_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compare statute chunk to policy chunk for compliance. Uses YAML-configurable prompt.

        Same output schema as gap_check: addressed, policy_quote, missing, conflict, conflict_description.
        """
        from prompt_loader import load_prompt_yaml, render_prompt

        path = prompt_path or getattr(self._config, "gap_analysis_chunk_prompt_path", "prompts/gap_analysis_chunk.yaml")
        cfg = load_prompt_yaml(path)
        prompt = render_prompt(
            cfg["prompt"],
            STATUTE_CHUNK=statute_chunk_text or "",
            POLICY_CHUNK=policy_chunk_text or "",
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

    async def gap_check_subchunks(
        self,
        statute_subchunk_text: str,
        statute_chunk_text: str,
        policy_subchunk_text: str,
        policy_chunk_text: str,
    ) -> Dict[str, Any]:
        """Compare subchunks for compliance with enclosing chunk context. Same schema as gap_check."""
        prompt = (
            SUBCHUNK_GAP_CHECK_PROMPT.replace("<<<STATUTE_SUBCHUNK>>>", (statute_subchunk_text or "")[:3000])
            .replace("<<<STATUTE_CHUNK>>>", (statute_chunk_text or "")[:3000])
            .replace("<<<POLICY_SUBCHUNK>>>", (policy_subchunk_text or "")[:3000])
            .replace("<<<POLICY_CHUNK>>>", (policy_chunk_text or "")[:3000])
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

    async def citation_check(
        self,
        policy_excerpt: str,
        statute_chunk_text: str,
        statute_reference: str,
        jurisdiction: str,
    ) -> Dict[str, Any]:
        """Return { alignment: bool, policy_quote: str | null, statute_excerpt: str | null }."""
        prompt = (
            CITATION_PROMPT.replace("<<<POLICY_EXCERPT>>>", (policy_excerpt or "")[:2000])
            .replace("<<<STATUTE_CHUNK>>>", (statute_chunk_text or "")[:3000])
            .replace("<<<JURISDICTION>>>", jurisdiction or "")
        )
        out = await self._call_json(prompt)
        if not out:
            return {"alignment": False, "policy_quote": None, "statute_excerpt": None}
        return {
            "alignment": bool(out.get("alignment")),
            "policy_quote": out.get("policy_quote") if out.get("policy_quote") else None,
            "statute_excerpt": out.get("statute_excerpt") if out.get("statute_excerpt") else None,
        }

    async def risk_assessment(self, policy_text: str, statute_summary: str) -> Dict[str, Any]:
        """Return { processing_purposes, data_categories, risks, mitigations, gaps_from_statute }."""
        prompt = (
            RISK_ASSESSMENT_PROMPT.replace("<<<POLICY_TEXT>>>", (policy_text or "")[:8000])
            .replace("<<<STATUTE_SUMMARY>>>", (statute_summary or "")[:6000])
        )
        out = await self._call_json(prompt)
        if not out:
            return {
                "processing_purposes": [],
                "data_categories": [],
                "risks": [],
                "mitigations": [],
                "gaps_from_statute": [],
            }
        return {
            "processing_purposes": out.get("processing_purposes") or [],
            "data_categories": out.get("data_categories") or [],
            "risks": out.get("risks") or [],
            "mitigations": out.get("mitigations") or [],
            "gaps_from_statute": out.get("gaps_from_statute") or [],
        }

    async def suggest_policy(
        self,
        policy_text: str,
        gap_analysis_text: str,
        gap_analysis_match: str,
        statute_text: str,
    ) -> Optional[Dict[str, Any]]:
        """Return { suggested_policy_text, modifications_description } or None on failure."""
        prompt = (
            SUGGEST_POLICY_PROMPT
            .replace("<<<POLICY_TEXT>>>", (policy_text or "")[:8000])
            .replace("<<<GAP_ANALYSIS_TEXT>>>", (gap_analysis_text or "")[:3000])
            .replace("<<<GAP_ANALYSIS_MATCH>>>", (gap_analysis_match or "")[:500])
            .replace("<<<STATUTE_TEXT>>>", (statute_text or "")[:6000])
        )
        out = await self._call_json(prompt, max_tokens=4096)
        # region agent log
        import time as _t; open("/Users/kurthayes/Dev/AI/web-gather-api/.cursor/debug-4b665c.log","a").write(json.dumps({"sessionId":"4b665c","hypothesisId":"C,D","location":"llm_client.py:suggest_policy:result","message":"suggest_policy LLM result","data":{"out_is_none":out is None,"has_suggested_text":isinstance(out.get("suggested_policy_text"),str) if out else False,"suggested_text_len":len(out.get("suggested_policy_text","")) if out else 0,"modifications_len":len(str(out.get("modifications_description",""))) if out else 0,"prompt_len":len(prompt)},"timestamp":int(_t.time()*1000)})+"\n")
        # endregion
        if not out or not isinstance(out.get("suggested_policy_text"), str):
            return None
        return {
            "suggested_policy_text": out["suggested_policy_text"],
            "modifications_description": str(out.get("modifications_description") or ""),
        }


class StubLLMClient:
    """Deterministic stub for tests."""

    def __init__(self, response_text: str) -> None:
        self._response_text = response_text

    async def compare_section(
        self, section_id: str, section_text: str, candidates: List[StatuteCandidate]
    ) -> str:
        return self._response_text
