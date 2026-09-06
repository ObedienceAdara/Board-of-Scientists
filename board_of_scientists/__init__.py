"""Board of Scientists package.

The package is now the canonical application boundary. During this structural
migration the previous root-level implementation is retained under
``board_of_scientists._legacy`` so behavior can be preserved while each public
subsystem is moved behind a stable module boundary.
"""

from importlib import import_module
import sys

# Legacy modules still use their original absolute imports. Register them under
# the original names before a legacy module is imported through a new facade.
for _name in ("state", "schemas", "prompts", "agent_registry", "llm_provider", "tools"):
    try:
        sys.modules.setdefault(_name, import_module(f"board_of_scientists._legacy.{_name}"))
    except Exception:
        # Keep package import lightweight; dependencies are loaded when the
        # affected subsystem is used.
        pass

__all__ = ["graph", "agents", "evidence", "execution", "ingestion", "schemas", "reports"]
