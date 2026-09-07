"""Domain state and LangGraph state contracts.

The workflow carries one ``ResearchState`` object, but its fields are explicit
bounded contexts rather than one giant mutable bag of unrelated values.
"""

from __future__ import annotations

from typing import Any
from typing_extensions import TypedDict

from pydantic import BaseModel, Field


class PaperSection(TypedDict):
    page: int
    heading: str
    text: str
    figures: list
    tables: list
    equations: list


class CodeModule(TypedDict):
    filename: str
    language: str
    code: str
    description: str
    status: str


class AgentMessage(TypedDict):
    sender: str
    recipient: str
    sender_name: str
    recipient_name: str
    content: str
    message_type: str


class ResearchInput(BaseModel):
    """Input that identifies the research task."""

    pdf_path: str = ""
    paper_title: str = ""
    paper_abstract: str = ""


class PaperCorpus(BaseModel):
    """Canonical representation of the ingested paper."""

    raw_pages: list[PaperSection] = Field(default_factory=list)
    full_paper_text: str = ""
    figures_summary: str = ""
    tables_summary: str = ""
    equations_summary: str = ""
    page_notes: str = ""
    page_notes_list: list[str] = Field(default_factory=list)


class AnalysisState(BaseModel):
    """Scientific interpretation produced before implementation planning."""

    research_report: str = ""
    cro_reading_notes: str = ""
    theoretical_analysis: str = ""


class ArchitectureState(BaseModel):
    """System design produced by the Architect/CRO planning stages."""

    architecture_analysis: str = ""
    implementation_plan: str = ""
    codebase_structure: str = ""
    file_manifest: list[dict[str, Any]] = Field(default_factory=list)


class ImplementationState(BaseModel):
    """Generated source plus the review loop around it."""

    code_modules: dict[str, CodeModule | dict[str, Any]] = Field(default_factory=dict)
    review_feedback: dict[str, Any] = Field(default_factory=dict)
    review_summary: str = ""
    implementation_notes: str = ""


class ValidationState(BaseModel):
    """Measured execution results and their separate interpretation."""

    execution_results: str = ""
    measured_validation: dict[str, Any] = Field(default_factory=dict)
    validation_report: str = ""
    discrepancies: str = ""


class EvidenceState(BaseModel):
    """Evidence artifacts available to future global scientific reasoning."""

    claims: list[dict[str, Any]] = Field(default_factory=list)
    equations: list[dict[str, Any]] = Field(default_factory=list)
    traceability_edges: list[dict[str, Any]] = Field(default_factory=list)
    consistency_issues: list[dict[str, Any]] = Field(default_factory=list)


class CommunicationState(BaseModel):
    """Inter-agent messages and CRO control metadata."""

    message_board: list[AgentMessage | dict[str, Any]] = Field(default_factory=list)
    cro_directives: dict[str, str] = Field(default_factory=dict)
    evaluations: dict[str, dict[str, Any]] = Field(default_factory=dict)
    revision_counts: dict[str, int] = Field(default_factory=dict)
    needs_revision: list[str] = Field(default_factory=list)


class OutputState(BaseModel):
    """Persisted deliverables exposed after a research run."""

    readme: str = ""
    implementation_paper: str = ""
    output_dir: str = ""
    pdf_report_path: str = ""
    final_verdict: str = ""


class ResearchState(TypedDict):
    """LangGraph state envelope containing bounded domain objects."""

    research_input: ResearchInput
    paper_corpus: PaperCorpus
    analysis: AnalysisState
    architecture: ArchitectureState
    implementation: ImplementationState
    validation: ValidationState
    evidence: EvidenceState
    communication: CommunicationState
    output: OutputState


def create_initial_state(pdf_path: str) -> ResearchState:
    """Construct a fully initialized domain state for a new research run."""
    return {
        "research_input": ResearchInput(pdf_path=pdf_path),
        "paper_corpus": PaperCorpus(),
        "analysis": AnalysisState(),
        "architecture": ArchitectureState(),
        "implementation": ImplementationState(),
        "validation": ValidationState(),
        "evidence": EvidenceState(),
        "communication": CommunicationState(),
        "output": OutputState(),
    }


__all__ = [
    "AgentMessage",
    "AnalysisState",
    "ArchitectureState",
    "CodeModule",
    "CommunicationState",
    "create_initial_state",
    "EvidenceState",
    "OutputState",
    "PaperCorpus",
    "PaperSection",
    "ResearchInput",
    "ResearchState",
    "ImplementationState",
    "ValidationState",
]
