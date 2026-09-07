"""Evidence-layer domain contracts.

Schemas own all evidence data structures. Concrete evidence operations import
these models; the dependency never points back into the evidence package.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator


def normalize_claim_text(text: str) -> str:
    """Normalize whitespace while preserving semantic wording."""
    return re.sub(r"\s+", " ", str(text).strip())


class Claim(BaseModel):
    id: str
    text: str
    source: str
    kind: Literal["contribution", "method", "result", "assumption", "limitation", "implementation"] = "method"
    evidence_level: int = Field(default=1, ge=0, le=7)
    references: list[str] = Field(default_factory=list)

    @field_validator("text")
    @classmethod
    def _normalize_text(cls, value: str) -> str:
        return normalize_claim_text(value)


class EquationEvidence(BaseModel):
    id: str
    label: str = ""
    expression: str
    page: int | None = Field(default=None, ge=1)
    section: str = ""
    role: str = ""
    implementation_refs: list[str] = Field(default_factory=list)


class TraceEdge(BaseModel):
    source: str
    relation: str
    target: str
    evidence: str = ""


class TraceabilityGraph(BaseModel):
    edges: list[TraceEdge] = Field(default_factory=list)


class ConsistencyIssue(BaseModel):
    severity: Literal["info", "warning", "error", "critical"] = "warning"
    source: str
    message: str
    related: tuple[str, ...] = ()


__all__ = [
    "Claim",
    "EquationEvidence",
    "TraceEdge",
    "TraceabilityGraph",
    "ConsistencyIssue",
    "normalize_claim_text",
]
