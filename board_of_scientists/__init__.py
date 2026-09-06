"""Board of Scientists package.

The package is the canonical application boundary. The previous implementation
is retained under ``board_of_scientists._legacy`` only as a compatibility
archive while each subsystem is migrated behind the new public module layout.
"""

from importlib import import_module
import sys

# The preserved implementation still uses historical absolute imports. Register
# those names before loading it so the behavior of the existing implementation
# remains intact during the structural migration.
for _name in ("state", "schemas", "prompts", "agent_registry", "llm_provider", "tools"):
    sys.modules.setdefault(_name, import_module(f"board_of_scientists._legacy.{_name}"))
sys.modules.setdefault("agents", import_module("board_of_scientists._legacy.agents"))

__all__ = ["graph", "agents", "evidence", "execution", "ingestion", "schemas", "reports"]
