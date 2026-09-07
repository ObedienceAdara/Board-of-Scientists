"""Artifact persistence and provenance helpers.

Reporting owns persistence of generated artifacts. It does not depend on
execution internals or graph orchestration.
"""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath


class ArtifactPathError(ValueError):
    """Raised when a generated artifact name is unsafe or invalid."""


def sanitize_relative_path(filename: str) -> str:
    """Normalize a safe relative artifact path or raise ``ArtifactPathError``."""
    if not isinstance(filename, str) or not filename.strip():
        raise ArtifactPathError("Artifact filename must be a non-empty string.")
    if "\x00" in filename:
        raise ArtifactPathError("Artifact filename contains a NUL byte.")

    cleaned = filename.replace("\\", "/").strip()
    if cleaned.startswith("/"):
        raise ArtifactPathError(f"Absolute artifact paths are forbidden: {filename!r}")
    if len(cleaned) > 1 and cleaned[1] == ":":
        raise ArtifactPathError(f"Drive-qualified artifact paths are forbidden: {filename!r}")

    parts = [part for part in PurePosixPath(cleaned).parts if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        raise ArtifactPathError(f"Artifact path traversal is forbidden: {filename!r}")
    if any("\x00" in part for part in parts):
        raise ArtifactPathError("Artifact filename contains a NUL byte.")
    return "/".join(parts)


def save_code_file(output_dir: str, filename: str, code: str) -> str:
    """Persist one generated artifact without allowing path escape."""
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    relative = sanitize_relative_path(filename)
    path = (root / Path(*PurePosixPath(relative).parts)).resolve()
    if path != root and root not in path.parents:
        raise ArtifactPathError(f"Rejected artifact path outside output directory: {filename!r}")
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
    path = Path(output_dir).resolve() / "team_communications.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(messages, indent=2), encoding="utf-8")
    return str(path)


__all__ = [
    "ArtifactPathError",
    "sanitize_relative_path",
    "save_code_file",
    "save_all_modules",
    "save_message_board",
]
