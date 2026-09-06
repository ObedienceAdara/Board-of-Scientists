"""Artifact persistence and provenance helpers.

Reporting owns persistence of generated artifacts. It does not depend on
execution internals or graph orchestration.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def sanitize_relative_path(filename: str) -> str:
    """Normalize a relative artifact path and reject traversal components."""
    if not filename or "\x00" in filename:
        return "unnamed_module.py"

    cleaned = filename.replace("\\", "/").strip().lstrip("/")
    if len(cleaned) > 1 and cleaned[1] == ":":
        cleaned = cleaned[2:].lstrip("/")

    parts = [part for part in cleaned.split("/") if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        return "unnamed_module.py"
    return "/".join(parts)


def save_code_file(output_dir: str, filename: str, code: str) -> str:
    """Persist one generated artifact without allowing path escape."""
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    relative = sanitize_relative_path(filename)
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"Rejected artifact path outside output directory: {filename}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code, encoding="utf-8")
    return str(path)


def save_all_modules(output_dir: str, code_modules: dict) -> list[str]:
    """Persist all generated code/documentation modules."""
    saved = []
    for filename, module in code_modules.items():
        code = module.get("code", "") if isinstance(module, dict) else str(module)
        saved.append(save_code_file(output_dir, filename, code))
    return saved


def save_message_board(output_dir: str, messages: list) -> str:
    """Persist the inter-agent communication log as JSON."""
    path = Path(output_dir) / "team_communications.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(messages, indent=2), encoding="utf-8")
    return str(path)


__all__ = ["sanitize_relative_path", "save_code_file", "save_all_modules", "save_message_board"]
