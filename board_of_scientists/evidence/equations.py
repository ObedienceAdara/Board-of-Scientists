"""Equation evidence primitives."""
from pydantic import BaseModel, Field
class EquationEvidence(BaseModel):
    id: str
    label: str = ""
    expression: str
    page: int | None = None
    section: str = ""
    role: str = ""
    implementation_refs: list[str] = Field(default_factory=list)
