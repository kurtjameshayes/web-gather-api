"""LLM client wrapper and prompt templating."""
from __future__ import annotations

import asyncio
import logging
from typing import List, Protocol

import anthropic

from compliance_config import ComplianceConfig
from vector_retriever import StatuteCandidate

logger = logging.getLogger("policy-compliance")


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

        def run_call() -> str:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            content = []
            for block in response.content:
                if block.type == "text":
                    content.append(block.text)
            return "".join(content)

        return await asyncio.to_thread(run_call)


class StubLLMClient:
    """Deterministic stub for tests."""

    def __init__(self, response_text: str) -> None:
        self._response_text = response_text

    async def compare_section(
        self, section_id: str, section_text: str, candidates: List[StatuteCandidate]
    ) -> str:
        return self._response_text
