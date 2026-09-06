"""Board of Scientists application package.

The package is the canonical application boundary. Subsystems are organized by
responsibility: graph orchestration, agents, evidence, execution, ingestion,
schemas, and reporting.

Architecture is enforced by repository tests. The package root intentionally
contains no import aliases or compatibility wiring between subsystem layers.
"""

__all__ = ["graph", "agents", "evidence", "execution", "ingestion", "schemas", "reports"]
