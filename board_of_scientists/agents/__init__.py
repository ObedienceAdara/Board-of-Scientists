"""Specialized research-agent boundaries.

The current ``_runtime`` module still contains historical absolute imports.
Install their compatibility aliases only when the agents boundary is loaded,
so importing schemas, evidence, ingestion, or reports remains lightweight and
side-effect free.
"""

from importlib import import_module
import sys

_COMPAT_IMPORTS = {
    "state": "board_of_scientists.schemas.state",
    "schemas": "board_of_scientists.schemas.agents",
    "prompts": "board_of_scientists.agents.prompts",
    "agent_registry": "board_of_scientists.agents.registry",
    "llm_provider": "board_of_scientists.agents._llm",
    "tools": "board_of_scientists.execution._runtime",
}

for _name, _module_path in _COMPAT_IMPORTS.items():
    sys.modules.setdefault(_name, import_module(_module_path))

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
