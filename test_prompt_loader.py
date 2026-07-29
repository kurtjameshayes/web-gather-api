"""Regression tests for YAML prompt loading and placeholder rendering."""
from __future__ import annotations

from pathlib import Path

import pytest

from prompt_loader import load_prompt_yaml, render_prompt


def test_load_prompt_yaml_reads_required_fields(tmp_path: Path) -> None:
    path = tmp_path / "sample.yaml"
    path.write_text(
        "description: sample\n"
        "placeholders:\n  - POLICY_TEXT\n"
        "output_schema:\n  status: string\n"
        "prompt: |\n  Analyze <<<POLICY_TEXT>>>\n",
        encoding="utf-8",
    )

    loaded = load_prompt_yaml(str(path))
    assert loaded["description"] == "sample"
    assert loaded["placeholders"] == ["POLICY_TEXT"]
    assert loaded["output_schema"] == {"status": "string"}
    assert "Analyze <<<POLICY_TEXT>>>" in loaded["prompt"]


def test_load_prompt_yaml_rejects_missing_prompt(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("description: no prompt\n", encoding="utf-8")
    with pytest.raises(ValueError, match="prompt"):
        load_prompt_yaml(str(path))


def test_render_prompt_replaces_and_truncates_values() -> None:
    template = "Policy:\n<<<POLICY_TEXT>>>\nStatute:\n<<<STATUTE_TEXT>>>"
    long_policy = "A" * 3500
    rendered = render_prompt(template, POLICY_TEXT=long_policy, STATUTE_TEXT="Keep me")
    assert "Statute:\nKeep me" in rendered
    assert "A" * 3000 in rendered
    assert "A" * 3001 not in rendered
    assert "<<<POLICY_TEXT>>>" not in rendered
