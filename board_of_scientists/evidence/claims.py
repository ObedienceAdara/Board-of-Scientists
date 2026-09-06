"""Claim extraction and normalization primitives."""
from typing import Literal
from pydantic import BaseModel, Field
class Claim(BaseModel):
    id: str
    text: str
    source: str
    kind: Literal["contribution","method","result","assumption","limitation","implementation"] = "method"
    evidence_level: int = Field(default=1, ge=0, le=7)
    references: list[str] = Field(default_factory=list)
