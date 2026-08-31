"""Regression tests for prompt_loader load/render contracts.

YAML must be a dict with a non-empty prompt. render_prompt replaces
<<<KEY>>> placeholders and truncates each value at 3000 characters.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from prompt_loader import load_prompt_yaml, render_prompt


def test_load_prompt_yaml_reads_prompt_and_defaults(tmp_path: Path) -> None:
    path = tmp_path / "gap.yaml"
    path.write_text(
        "prompt: |\n  Hello <<<NAME>>>\noutput_schema:\n  type: object\n",
        encoding="utf-8",
    )

    loaded = load_prompt_yaml(str(path))

    assert loaded["prompt"] == "Hello <<<NAME>>>"
    assert loaded["placeholders"] == []
    assert loaded["output_schema"] == {"type": "object"}
    assert loaded["description"] == ""


def test_load_prompt_yaml_missing_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "nope.yaml"
    with pytest.raises(FileNotFoundError, match="Prompt config not found"):
        load_prompt_yaml(str(missing))


def test_load_prompt_yaml_rejects_non_dict_and_empty_prompt(tmp_path: Path) -> None:
    listed = tmp_path / "list.yaml"
    listed.write_text("- just a list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a dict"):
        load_prompt_yaml(str(listed))

    empty = tmp_path / "empty.yaml"
    empty.write_text("description: no prompt here\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must have 'prompt' key"):
        load_prompt_yaml(str(empty))


def test_render_prompt_replaces_placeholders_and_truncates() -> None:
    template = "A: <<<LEFT>>>\nB: <<<RIGHT>>>"
    rendered = render_prompt(template, LEFT="ok", RIGHT="x" * 4000, MISSING=None)

    assert "A: ok" in rendered
    assert "B: " + ("x" * 3000) in rendered
    assert "x" * 3001 not in rendered
    assert "<<<RIGHT>>>" not in rendered
    assert rendered.count("x") == 3000
