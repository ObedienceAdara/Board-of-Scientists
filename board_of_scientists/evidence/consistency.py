"""Global-consistency seam.

Phase 1 deliberately provides data structures and deterministic checks only;
full cross-report reconciliation is the next scientific-system phase.
"""
from dataclasses import dataclass
@dataclass(frozen=True)
class ConsistencyIssue:
    severity: str
    source: str
    message: str
    related: tuple[str, ...] = ()

def check_duplicate_trace_targets(edges):
    seen = {}
    issues = []
    for edge in edges:
        previous = seen.get(edge.target)
        if previous and previous != edge.relation:
            issues.append(ConsistencyIssue("warning", edge.source, f"Conflicting relations for {edge.target}: {previous} vs {edge.relation}", (edge.target,)))
        seen[edge.target] = edge.relation
    return issues
