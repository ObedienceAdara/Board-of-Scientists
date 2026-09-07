"""Deterministic evidence consistency checks."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from board_of_scientists.schemas.evidence import ConsistencyIssue, TraceEdge


def check_duplicate_trace_targets(edges: Iterable[TraceEdge]) -> list[ConsistencyIssue]:
    """Find targets that receive incompatible relations."""
    relations: dict[str, dict[str, str]] = defaultdict(dict)
    for edge in edges:
        relation = edge.relation.strip()
        if relation:
            relations[edge.target][relation] = edge.source

    issues: list[ConsistencyIssue] = []
    for target, rels in relations.items():
        if len(rels) > 1:
            ordered = sorted(rels)
            issues.append(
                ConsistencyIssue(
                    severity="warning",
                    source=rels[ordered[0]],
                    message=f"Conflicting relations for {target}: {', '.join(ordered)}",
                    related=(target, *ordered),
                )
            )
    return issues


__all__ = ["ConsistencyIssue", "check_duplicate_trace_targets"]
