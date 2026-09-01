"""Unit tests for core cosine_similarity used by GET /search ranking."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np

sys.modules.setdefault("pyppeteer", MagicMock())
sys.modules.setdefault("pypdf", MagicMock())
sys.modules.setdefault("sentence_transformers", MagicMock())

from core import cosine_similarity  # noqa: E402


def test_cosine_similarity_empty_chunks_returns_empty_array() -> None:
    query = np.array([1.0, 0.0])
    scores = cosine_similarity(query, [])
    assert scores.shape == (0,)


def test_cosine_similarity_dot_product_ranking() -> None:
    query = np.array([1.0, 0.0])
    chunks = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([-1.0, 0.0])]
    scores = cosine_similarity(query, chunks)
    assert scores.tolist() == [1.0, 0.0, -1.0]
