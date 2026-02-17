"""Load prompts from YAML configuration files."""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


def load_prompt_yaml(path: str, base_dir: Optional[str] = None) -> Dict[str, Any]:
    """Load a prompt configuration from a YAML file.

    Args:
        path: Relative path (e.g. prompts/gap_analysis_chunk.yaml) or absolute path.
        base_dir: Base directory for relative paths. Defaults to project root.

    Returns:
        Dict with keys: prompt (str), placeholders (list), output_schema (dict), description (str).
    """
    if yaml is None:
        raise RuntimeError("PyYAML is required for prompt loading. Install with: pip install pyyaml")

    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_dir, path) if not os.path.isabs(path) else path

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Prompt config not found: {full_path}")

    with open(full_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Prompt YAML must be a dict: {full_path}")

    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        raise ValueError(f"Prompt YAML must have 'prompt' key: {full_path}")

    return {
        "prompt": prompt,
        "placeholders": data.get("placeholders") or [],
        "output_schema": data.get("output_schema") or {},
        "description": data.get("description") or "",
    }


def render_prompt(template: str, **kwargs: str) -> str:
    """Replace <<<PLACEHOLDER>>> in template with kwargs values."""
    result = template
    for key, value in kwargs.items():
        placeholder = f"<<<{key}>>>"
        result = result.replace(placeholder, (value or "")[:3000])
    return result
