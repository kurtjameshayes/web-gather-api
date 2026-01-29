"""Shared helpers for policy statute compliance service."""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Optional


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


def safe_truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


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
