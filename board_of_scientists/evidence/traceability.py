"""Traceability edges linking source evidence to implementation artifacts."""
from pydantic import BaseModel, Field


class TraceEdge(BaseModel):
    source: str
    relation: str
    target: str
    evidence: str = ""


class TraceabilityGraph(BaseModel):
    edges: list[TraceEdge] = Field(default_factory=list)


__all__ = ["TraceEdge", "TraceabilityGraph"]
