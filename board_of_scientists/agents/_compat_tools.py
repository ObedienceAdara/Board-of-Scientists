"""Compatibility facade for the historical agent runtime.

The legacy runtime still imports a module named ``tools``. This facade keeps
that import working while routing each capability to its canonical bounded
context. No duplicate implementation lives here.
"""

from ..execution.experiments import format_measured_results, run_codebase_validation
from ..ingestion.equations import extract_equations
from ..ingestion.pdf import extract_pdf_pages, get_paper_metadata
from ..reports.pdf import generate_implementation_report
from ..reports.provenance import save_all_modules, save_code_file, save_message_board, sanitize_relative_path
from ..execution.sandbox import execute_python_code

__all__ = [
    "extract_pdf_pages",
    "get_paper_metadata",
    "extract_equations",
    "save_all_modules",
    "save_code_file",
    "save_message_board",
    "sanitize_relative_path",
    "run_codebase_validation",
    "format_measured_results",
    "execute_python_code",
    "generate_implementation_report",
]
