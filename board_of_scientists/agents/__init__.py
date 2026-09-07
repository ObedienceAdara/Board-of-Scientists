"""Specialized research-agent boundaries."""

from importlib import import_module
from pathlib import PurePosixPath
import sys

_COMPAT_IMPORTS = {
    "state": "board_of_scientists.schemas.state",
    "schemas": "board_of_scientists.schemas.agents",
    "prompts": "board_of_scientists.agents.prompts",
    "agent_registry": "board_of_scientists.agents.registry",
    "llm_provider": "board_of_scientists.agents._llm",
    "tools": "board_of_scientists.agents._compat_tools",
}

_prompt_module = import_module(".prompts", __name__)
_EXTERNAL_RESEARCH_GUARD = """

SOURCE-BOUNDARY RULE:
Use only evidence and references supplied in this prompt. Do not invent,
guess, or fabricate papers, authors, DOIs, arXiv URLs, benchmark numbers,
experimental results, or external citations. You do not have implicit web
access. When an external fact is required but not supplied, explicitly mark it
as [EXTERNAL-REFERENCE-REQUIRED] rather than making it up.
"""
for _prompt_name in (
    "CRO_READING_NOTES_PROMPT", "CRO_IMPLEMENTATION_PLAN_PROMPT", "THEORIST_PROMPT",
    "ARCHITECT_PROMPT", "ENGINEER_PROMPT", "REVIEWER_PROMPT",
    "EXPERIMENT_ENGINEER_PROMPT", "WRITER_PROMPT",
):
    setattr(_prompt_module, _prompt_name, getattr(_prompt_module, _prompt_name) + _EXTERNAL_RESEARCH_GUARD)

for _name, _module_path in _COMPAT_IMPORTS.items():
    sys.modules.setdefault(_name, import_module(_module_path))

_runtime = import_module("._runtime", __name__)
import_module("._runtime_hardening", __name__).install(_runtime)

from ..ingestion.pdf import extract_pdf_pages, get_paper_metadata
from ..ingestion.equations import extract_equations
from ..execution.experiments import format_measured_results, run_codebase_validation
from ..execution.sandbox import execute_python_code

_runtime.extract_pdf_pages = extract_pdf_pages
_runtime.get_paper_metadata = get_paper_metadata
_runtime.extract_equations = extract_equations
_runtime.run_codebase_validation = run_codebase_validation
_runtime.format_measured_results = format_measured_results
_runtime.execute_python_code = execute_python_code

from .analyst import analyst_agent
from .theorist import theorist_agent
from .architect import architect_agent
from .engineer import engineer_agent
from .reviewer import reviewer_agent
from .experiment import experiment_engineer_agent
from .writer import writer_agent
from .cro import cro_read_paper, cro_create_plan, cro_evaluate_agent, cro_final_verdict

_runtime.cro_read_paper = cro_read_paper
_runtime.cro_create_plan = cro_create_plan
_runtime.cro_evaluate_agent = cro_evaluate_agent
_runtime.cro_final_verdict = cro_final_verdict


def sanitize_relative_path(filename: str) -> str:
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("Artifact filename must be non-empty")
    cleaned = filename.replace("\\", "/").strip()
    if cleaned.startswith("/") or (len(cleaned) > 1 and cleaned[1] == ":"):
        raise ValueError(f"Unsafe absolute artifact path: {filename!r}")
    parts = [part for part in PurePosixPath(cleaned).parts if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        raise ValueError(f"Unsafe artifact path: {filename!r}")
    return "/".join(parts)


def _safe_manifest(file_manifest):
    """Sanitize an Architect manifest while rejecting only unsafe entries."""
    seen = set()
    sanitized = []
    for spec in file_manifest:
        raw = spec.model_dump() if hasattr(spec, "model_dump") else dict(spec)
        try:
            filename = sanitize_relative_path(raw.get("filename", ""))
        except (TypeError, ValueError):
            continue
        if filename in seen:
            continue
        seen.add(filename)
        sanitized.append({
            "filename": filename,
            "description": raw.get("description", ""),
            "depends_on": list(raw.get("depends_on") or []),
            "group": (raw.get("group") or "").strip(),
        })
    valid = {entry["filename"] for entry in sanitized}
    for entry in sanitized:
        clean_deps = []
        for dep in entry["depends_on"]:
            try:
                dep = sanitize_relative_path(dep)
            except (TypeError, ValueError):
                continue
            if dep in valid and dep != entry["filename"] and dep not in clean_deps:
                clean_deps.append(dep)
        entry["depends_on"] = clean_deps
    return sanitized

_runtime.sanitize_relative_path = sanitize_relative_path
_runtime._sanitize_manifest = _safe_manifest

_compat_tools = import_module("._compat_tools", __name__)
for _name in ("save_all_modules", "save_code_file", "save_message_board"):
    setattr(_runtime, _name, getattr(_compat_tools, _name))

__all__ = [
    "analyst_agent", "theorist_agent", "architect_agent", "engineer_agent", "reviewer_agent",
    "experiment_engineer_agent", "writer_agent", "cro_read_paper", "cro_create_plan",
    "cro_evaluate_agent", "cro_final_verdict",
]
