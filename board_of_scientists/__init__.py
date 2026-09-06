"""Board of Scientists package.

The package is now the canonical application boundary. During this structural
migration the previous implementation is encapsulated under
``board_of_scientists._legacy`` while stable subsystem modules become the public
entry points.
"""

from importlib import import_module
import sys

# The preserved implementation still uses historical absolute imports. Register
# those names before loading the legacy graph so behavior remains unchanged while
# the new package layout becomes canonical.
for _name in ("state", "schemas", "prompts", "agent_registry", "llm_provider", "tools"):
    sys.modules.setdefault(_name, import_module(f"board_of_scientists._legacy.{_name}"))

# `agents.py` depends on the aliases above. Register it only after its dependencies
# are available, allowing the legacy graph to continue importing unchanged.
sys.modules.setdefault("agents", import_module("board_of_scientists._legacy.agents"))

__all__ = ["graph", "agents", "evidence", "execution", "ingestion", "schemas", "reports"]
