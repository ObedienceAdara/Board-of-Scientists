"""Board of Scientists application package.

The package is the canonical application boundary. Subsystems are organized by
responsibility: graph orchestration, agents, evidence, execution, ingestion,
schemas, and reporting.

Compatibility aliases point historical absolute imports at active package
implementations. No subsystem depends on an archived implementation.
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

__all__ = ["graph", "agents", "evidence", "execution", "ingestion", "schemas", "reports"]
