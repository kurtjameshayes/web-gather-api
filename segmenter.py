"""Policy segmentation heuristics."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional

from compliance_utils import slugify


HEADING_PATTERN = re.compile(r"^([A-Z][A-Z0-9\s\-]{3,}|[0-9IVX]+\.)")


@dataclass
class PolicySection:
    section_id: str
    section_text: str


class PolicySegmenter:
    def __init__(self, max_section_chars: int = 2000) -> None:
        self._max_section_chars = max_section_chars

    def segment(
        self,
        policy_text: str,
        existing_sections: Optional[Iterable[dict]] = None,
    ) -> List[PolicySection]:
        if existing_sections:
            sections = []
            for idx, section in enumerate(existing_sections, start=1):
                text = (section.get("section_text") or section.get("text") or "").strip()
                if not text:
                    continue
                section_id = section.get("section_id") or slugify(section.get("title", ""), f"section_{idx}")
                sections.extend(self._split_long_section(section_id, text))
            return sections

        text = (policy_text or "").strip()
        if not text:
            return []

        paragraphs = self._to_paragraphs(text)
        sections: List[PolicySection] = []
        current_title = None
        current_body: List[str] = []

        for paragraph in paragraphs:
            if self._is_heading(paragraph):
                if current_body or current_title:
                    section_id = slugify(current_title or "", f"section_{len(sections) + 1}")
                    section_text = self._merge_section_text(current_title, current_body)
                    sections.extend(self._split_long_section(section_id, section_text))
                    current_body = []
                current_title = paragraph
            else:
                current_body.append(paragraph)

        if current_body or current_title:
            section_id = slugify(current_title or "", f"section_{len(sections) + 1}")
            section_text = self._merge_section_text(current_title, current_body)
            sections.extend(self._split_long_section(section_id, section_text))

        return sections

    def _to_paragraphs(self, text: str) -> List[str]:
        lines = [line.strip() for line in text.splitlines()]
        paragraphs = []
        buffer: List[str] = []
        for line in lines:
            if not line:
                if buffer:
                    paragraphs.append(" ".join(buffer))
                    buffer = []
                continue
            buffer.append(line)
        if buffer:
            paragraphs.append(" ".join(buffer))
        return paragraphs

    def _is_heading(self, paragraph: str) -> bool:
        # Heuristic: short uppercase/numbered lines tend to be section headers.
        if len(paragraph) > 120:
            return False
        if HEADING_PATTERN.match(paragraph):
            return True
        return paragraph.isupper()

    def _merge_section_text(self, title: Optional[str], body: List[str]) -> str:
        if title and body:
            return f"{title}\n\n" + "\n\n".join(body)
        if title:
            return title
        return "\n\n".join(body)

    def _split_long_section(self, section_id: str, text: str) -> List[PolicySection]:
        if len(text) <= self._max_section_chars:
            return [PolicySection(section_id=section_id, section_text=text)]

        chunks = []
        remaining = text
        index = 1
        while remaining:
            chunk = remaining[: self._max_section_chars].rstrip()
            if not chunk:
                break
            chunks.append(PolicySection(section_id=f"{section_id}_{index}", section_text=chunk))
            remaining = remaining[len(chunk) :].lstrip()
            index += 1
        return chunks
