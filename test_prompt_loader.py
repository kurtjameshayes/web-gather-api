"""Regression tests for shared YAML prompt loading and placeholder rendering.

LLM callers mock ``load_prompt_yaml`` / ``render_prompt``, so these tests pin
the real contracts: invalid prompt files fail closed, each substituted value is
silently capped at 3000 characters, and replacement is sequential. That cap is
what actually reaches the model, including when a caller pre-slices a longer
policy (consumer-rights router passes 12000 characters). Later kwargs rewrite
placeholders that appear inside earlier values; earlier placeholders left
inside a later value stay literal.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from prompt_loader import load_prompt_yaml, render_prompt


V4_PLACEHOLDERS = (
    "ADAPTIVE_FEEDBACK",
    "REFERENCE_CONTEXT",
    "STATUTORY_REQUIREMENT",
    "POLICY_TEXT",
)


def test_load_prompt_yaml_strips_prompt_and_defaults_optional_fields(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompts" / "gap.yaml"
    prompt_file.parent.mkdir()
    prompt_file.write_text(
        "description:\n"
        "prompt: '  compare this  '\n"
        "placeholders:\n"
        "output_schema:\n",
        encoding="utf-8",
    )

    loaded = load_prompt_yaml("prompts/gap.yaml", base_dir=str(tmp_path))

    assert loaded["prompt"] == "compare this"
    assert loaded["placeholders"] == []
    assert loaded["output_schema"] == {}
    assert loaded["description"] == ""


def test_load_prompt_yaml_keeps_declared_metadata(tmp_path: Path) -> None:
    prompt_file = tmp_path / "gap.yaml"
    prompt_file.write_text(
        "description: gap check\n"
        "prompt: compare\n"
        "placeholders:\n"
        "  - STATUTE_CHUNK\n"
        "output_schema:\n"
        "  status: string\n",
        encoding="utf-8",
    )

    loaded = load_prompt_yaml(str(prompt_file), base_dir=str(tmp_path / "unused"))

    assert loaded["prompt"] == "compare"
    assert loaded["placeholders"] == ["STATUTE_CHUNK"]
    assert loaded["output_schema"] == {"status": "string"}
    assert loaded["description"] == "gap check"


def test_load_prompt_yaml_rejects_missing_blank_and_non_dict(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    with pytest.raises(FileNotFoundError, match="Prompt config not found"):
        load_prompt_yaml(str(missing))

    (tmp_path / "list.yaml").write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a dict"):
        load_prompt_yaml("list.yaml", base_dir=str(tmp_path))

    (tmp_path / "empty.yaml").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a dict"):
        load_prompt_yaml("empty.yaml", base_dir=str(tmp_path))

    (tmp_path / "blank.yaml").write_text("prompt: '   '\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must have 'prompt' key"):
        load_prompt_yaml("blank.yaml", base_dir=str(tmp_path))

    (tmp_path / "no-prompt.yaml").write_text("description: only\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must have 'prompt' key"):
        load_prompt_yaml("no-prompt.yaml", base_dir=str(tmp_path))


def test_render_prompt_caps_each_value_at_3000_characters() -> None:
    """Callers may pre-slice longer text; the renderer still keeps 3000."""
    policy = "P" * 12000
    rendered = render_prompt(
        "POLICY:\n<<<POLICY_TEXT>>>\nSTATUTE:\n<<<STATUTE_CONTEXT>>>",
        POLICY_TEXT=policy,
        STATUTE_CONTEXT="S" * 3000,
    )

    policy_line = rendered.split("POLICY:\n", 1)[1].split("\nSTATUTE:", 1)[0]
    statute_line = rendered.split("STATUTE:\n", 1)[1]
    assert policy_line == "P" * 3000
    assert statute_line == "S" * 3000
    assert "<<<POLICY_TEXT>>>" not in rendered
    assert "<<<STATUTE_CONTEXT>>>" not in rendered


def test_render_prompt_clears_empty_values_and_preserves_whitespace_and_braces() -> None:
    assert render_prompt("A<<<POLICY_TEXT>>>B", POLICY_TEXT=None) == "AB"
    assert render_prompt("A<<<POLICY_TEXT>>>B", POLICY_TEXT="") == "AB"
    assert render_prompt("<<<POLICY_TEXT>>>", POLICY_TEXT="  ") == "  "
    assert render_prompt("<<<POLICY_TEXT>>>", POLICY_TEXT='{"a": 1} 100%') == '{"a": 1} 100%'
    assert (
        render_prompt("<<<POLICY_TEXT>>> / <<<POLICY_TEXT>>>", POLICY_TEXT="keep")
        == "keep / keep"
    )
    assert "<<<UNFILLED>>>" in render_prompt("<<<UNFILLED>>>", POLICY_TEXT="x")


def test_render_prompt_later_values_rewrite_placeholders_inside_earlier_values() -> None:
    """Matches gap_check_v4 kwargs order: POLICY_TEXT is substituted last."""
    rendered = render_prompt(
        "FB=<<<ADAPTIVE_FEEDBACK>>>\nREQ=<<<STATUTORY_REQUIREMENT>>>\nPOL=<<<POLICY_TEXT>>>",
        ADAPTIVE_FEEDBACK="note <<<POLICY_TEXT>>>",
        STATUTORY_REQUIREMENT="must honor <<<POLICY_TEXT>>>",
        POLICY_TEXT="we delete data",
    )

    assert rendered == (
        "FB=note we delete data\n"
        "REQ=must honor we delete data\n"
        "POL=we delete data"
    )


def test_render_prompt_leaves_earlier_placeholders_inside_later_values() -> None:
    rendered = render_prompt(
        "REQ=<<<STATUTORY_REQUIREMENT>>>\nPOL=<<<POLICY_TEXT>>>",
        STATUTORY_REQUIREMENT="delete on request",
        POLICY_TEXT="quoted requirement <<<STATUTORY_REQUIREMENT>>> stays literal",
    )

    assert rendered == (
        "REQ=delete on request\n"
        "POL=quoted requirement <<<STATUTORY_REQUIREMENT>>> stays literal"
    )


@pytest.mark.parametrize(
    ("relative_path", "placeholders"),
    [
        ("prompts/gap_analysis_v4.yaml", V4_PLACEHOLDERS),
        ("prompts/gap_analysis_v3.yaml", ("STATUTE_CHUNK", "POLICY_CHUNK")),
        ("prompts/gap_analysis_chunk.yaml", ("STATUTE_CHUNK", "POLICY_CHUNK")),
        ("prompts/adaptive_critic.yaml", ("SUMMARY", "GAP_ANALYSIS_RESULTS")),
        (
            "prompts/consumer_rights_router.yaml",
            (
                "REQUEST_TYPE",
                "REQUEST_TYPE_LABEL",
                "POLICY_TEXT",
                "STATUTE_CONTEXT",
                "JURISDICTIONS",
            ),
        ),
    ],
)
def test_shipped_prompts_still_declare_renderer_placeholders(
    relative_path: str, placeholders: tuple[str, ...]
) -> None:
    loaded = load_prompt_yaml(relative_path)
    assert loaded["prompt"]
    for name in placeholders:
        assert f"<<<{name}>>>" in loaded["prompt"]


def test_shipped_v4_prompt_substitutes_every_slot() -> None:
    loaded = load_prompt_yaml("prompts/gap_analysis_v4.yaml")
    policy = "P" * 3001
    rendered = render_prompt(
        loaded["prompt"],
        ADAPTIVE_FEEDBACK="feedback-note",
        REFERENCE_CONTEXT="definitions block",
        STATUTORY_REQUIREMENT="must delete personal information",
        POLICY_TEXT=policy,
    )

    for name in V4_PLACEHOLDERS:
        assert f"<<<{name}>>>" not in rendered
    assert "feedback-note" in rendered
    assert "definitions block" in rendered
    assert "must delete personal information" in rendered
    assert "P" * 3000 in rendered
    assert "P" * 3001 not in rendered
