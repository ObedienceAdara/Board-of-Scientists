"""Specialized research-agent boundaries.

Historical absolute imports inside ``_runtime`` are isolated to this adapter
layer. Concrete capabilities are routed to their canonical package owners.
"""

from importlib import import_module
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
    "CRO_READING_NOTES_PROMPT",
    "CRO_IMPLEMENTATION_PLAN_PROMPT",
    "THEORIST_PROMPT",
    "ARCHITECT_PROMPT",
    "ENGINEER_PROMPT",
    "REVIEWER_PROMPT",
    "EXPERIMENT_ENGINEER_PROMPT",
    "WRITER_PROMPT",
):
    setattr(_prompt_module, _prompt_name, getattr(_prompt_module, _prompt_name) + _EXTERNAL_RESEARCH_GUARD)

for _name, _module_path in _COMPAT_IMPORTS.items():
    sys.modules.setdefault(_name, import_module(_module_path))

_runtime = import_module("._runtime", __name__)
import_module("._runtime_hardening", __name__).install(_runtime)

# The legacy runtime defines a few helpers later in the file. Rebind those
# globals to canonical implementations so production agents cannot bypass the
# hardened report/path/execution owners by calling the shadowed definitions.
from ..ingestion.pdf import extract_pdf_pages, get_paper_metadata
from ..ingestion.equations import extract_equations
from ..execution.experiments import format_measured_results, run_codebase_validation
from ..execution.sandbox import execute_python_code
from ..reports.pdf import generate_implementation_report
from ..reports.provenance import save_all_modules, save_message_board, save_code_file, sanitize_relative_path

_runtime.extract_pdf_pages = extract_pdf_pages
_runtime.get_paper_metadata = get_paper_metadata
_runtime.extract_equations = extract_equations
_runtime.run_codebase_validation = run_codebase_validation
_runtime.format_measured_results = format_measured_results
_runtime.execute_python_code = execute_python_code
_runtime.save_all_modules = save_all_modules
_runtime.save_code_file = save_code_file
_runtime.save_message_board = save_message_board
_runtime.sanitize_relative_path = sanitize_relative_path
_runtime.generate_implementation_report = generate_implementation_report

from .analyst import analyst_agent
from .theorist import theorist_agent
from .architect import architect_agent
from .engineer import engineer_agent
from .reviewer import reviewer_agent
from .experiment import experiment_engineer_agent
from .writer import writer_agent
from .cro import cro_read_paper, cro_create_plan, cro_evaluate_agent, cro_final_verdict

__all__ = [
    "analyst_agent",
    "theorist_agent",
    "architect_agent",
    "engineer_agent",
    "reviewer_agent",
    "experiment_engineer_agent",
    "writer_agent",
    "cro_read_paper",
    "cro_create_plan",
    "cro_evaluate_agent",
    "cro_final_verdict",
]
