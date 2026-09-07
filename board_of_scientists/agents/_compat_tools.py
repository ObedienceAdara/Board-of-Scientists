"""Compatibility facade for the historical agent runtime.

The legacy runtime still imports a module named ``tools``. This facade routes
analysis/execution capabilities to canonical owners while intentionally keeping
artifact persistence out of the agents dependency boundary.
"""

from ..execution.experiments import format_measured_results, run_codebase_validation
from ..execution.sandbox import execute_python_code
from ..ingestion.equations import extract_equations
from ..ingestion.pdf import extract_pdf_pages, get_paper_metadata


def _artifact_persistence_removed(*_args, **_kwargs):
    raise RuntimeError(
        "Artifact persistence is graph-owned. Use board_of_scientists.reports.provenance."
    )


save_all_modules = _artifact_persistence_removed
save_code_file = _artifact_persistence_removed
save_message_board = _artifact_persistence_removed

def sanitize_relative_path(filename: str) -> str:
    """Compatibility sanitizer; graph/report code owns actual persistence."""
    from pathlib import PurePosixPath
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("Artifact filename must be non-empty")
    cleaned = filename.replace("\\", "/").strip()
    if cleaned.startswith("/") or (len(cleaned) > 1 and cleaned[1] == ":"):
        raise ValueError(f"Unsafe absolute artifact path: {filename!r}")
    parts = [part for part in PurePosixPath(cleaned).parts if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        raise ValueError(f"Unsafe artifact path: {filename!r}")
    return "/".join(parts)

__all__ = [
    "extract_pdf_pages", "get_paper_metadata", "extract_equations",
    "save_all_modules", "save_code_file", "save_message_board",
    "sanitize_relative_path", "run_codebase_validation", "format_measured_results",
    "execute_python_code",
]
