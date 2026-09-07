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

# Patch prompt constants before the legacy runtime imports them. Models must
# never be instructed to fabricate external research, citations, or URLs when
# no search result was supplied by the orchestration layer.
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

# The imported runtime is now backed by canonical ingestion/execution/report
# capabilities and bounded LLM retry/error semantics.
_runtime = import_module("._runtime", __name__)
import_module("._runtime_hardening", __name__).install(_runtime)

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
