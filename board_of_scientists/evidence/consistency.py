"""Deterministic evidence consistency checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from board_of_scientists.schemas.evidence import TraceEdge


@dataclass(frozen=True)
class ConsistencyIssue:
    severity: str
    source: str
    message: str
    related: tuple[str, ...] = ()


def check_duplicate_trace_targets(edges: Iterable[TraceEdge]) -> list[ConsistencyIssue]:
    """Find targets that are assigned incompatible relations."""
    seen: dict[str, set[str]] = {}
    sources: dict[tuple[str, str], str] = {}
    for edge in edges:
        relation = edge.relation.strip()
        if not relation:
            continue
        seen.setdefault(edge.target, set()).add(relation)
        sources[(edge.target, relation)] = edge.source

    issues: list[ConsistencyIssue] = []
    for target, relations in seen.items():
        if len(relations) > 1:
            ordered = sorted(relations)
            source = sources.get((target, ordered[0]), "traceability")
            issues.append(
                ConsistencyIssue(
                    severity="warning",
                    source=source,
                    message=f"Conflicting relations for {target}: {', '.join(ordered)}",
                    related=(target, *ordered),
                )
            )
    return issues


__all__ = ["ConsistencyIssue", "check_duplicate_trace_targets"]
