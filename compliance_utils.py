"""Shared helpers for policy statute compliance service."""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import List, Optional


JURISDICTION_MAP = {
    "usa": "US",
    "u.s.": "US",
    "us": "US",
    "united states": "US",
    "california": "CA",
    "ca": "CA",
    "european union": "EU",
    "eu": "EU",
    "uk": "UK",
    "united kingdom": "UK",
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    return max(min_value, min(max_value, value))


def normalize_jurisdiction(value: str) -> str:
    normalized = (value or "").strip().lower()
    return JURISDICTION_MAP.get(normalized, value.strip().upper())


def jurisdiction_filter_values(normalized: str) -> List[str]:
    """Return values to use in a DB filter so both normalized (e.g. CA) and stored variants (e.g. California) match."""
    if not normalized:
        return []
    values = [normalized]
    normalized_lower = normalized.lower()
    for key, val in JURISDICTION_MAP.items():
        if val == normalized or val.lower() == normalized_lower:
            values.append(key)
            if key and key[0].isalpha():
                values.append(key.capitalize())
    return list(dict.fromkeys(values))


def safe_truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def truncate_at_sentence(text: str, max_chars: int = 350) -> str:
    """Truncate text at the last sentence boundary before max_chars.

    Finds the last . ! or ? before max_chars and returns the substring up to
    and including that character. If no sentence boundary found, returns
    text[:max_chars].rstrip().
    """
    if not text or max_chars <= 0:
        return text or ""
    if len(text) <= max_chars:
        return text
    segment = text[:max_chars]
    last_sent = max(
        segment.rfind("."),
        segment.rfind("!"),
        segment.rfind("?"),
    )
    if last_sent >= 0:
        return text[: last_sent + 1].strip()
    return segment.rstrip()


def slugify(value: str, fallback: str = "section") -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    cleaned = cleaned.strip("_")
    return cleaned or fallback


def extract_json_block(text: str) -> Optional[str]:
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else None


def validate_collection_name(name: str) -> bool:
    if not name:
        return False
    return re.match(r"^[A-Za-z0-9_\-]+$", name) is not None
