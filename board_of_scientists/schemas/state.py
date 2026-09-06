"""Domain state and LangGraph state contracts.

The workflow carries one ``ResearchState`` object, but its fields are explicit
bounded contexts rather than one giant mutable bag of unrelated values.

Agent implementations that still operate on the historical flat runtime shape
use ``to_runtime_state`` / ``from_runtime_state`` at the graph boundary. This
keeps the domain model canonical while allowing behavior-preserving migration
of individual agents.
"""

from __future__ import annotations

from typing import Any, TypedDict

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
    """Immutable-ish user/run input needed to identify the research task."""

    pdf_path: str = ""
    paper_title: str = ""
    paper_abstract: str = ""


class PaperCorpus(BaseModel):
    """Canonical representation of the ingested source paper and analyst notes."""

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
    """System-design contract produced by the Architect/CRO planning stages."""

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
    """Measured execution and the separate narrative interpretation of it."""

    execution_results: str = ""
    measured_validation: dict[str, Any] = Field(default_factory=dict)
    validation_report: str = ""
    discrepancies: str = ""


class EvidenceState(BaseModel):
    """Evidence artifacts that can later support global consistency reasoning."""

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
    """LangGraph state: one orchestration envelope containing bounded domains."""

    research_input: ResearchInput
    paper_corpus: PaperCorpus
    analysis: AnalysisState
    architecture: ArchitectureState
    implementation: ImplementationState
    validation: ValidationState
    evidence: EvidenceState
    communication: CommunicationState
    output: OutputState


RuntimeResearchState = dict[str, Any]


def create_initial_state(pdf_path: str) -> ResearchState:
    """Construct a fully initialized domain state for a new run."""

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


def to_runtime_state(state: ResearchState) -> RuntimeResearchState:
    """Project canonical domain state into the flat shape used by legacy-free agents."""

    ri = state["research_input"]
    pc = state["paper_corpus"]
    an = state["analysis"]
    ar = state["architecture"]
    im = state["implementation"]
    va = state["validation"]
    ev = state["evidence"]
    co = state["communication"]
    out = state["output"]

    return {
        "pdf_path": ri.pdf_path,
        "paper_title": ri.paper_title,
        "paper_abstract": ri.paper_abstract,
        "raw_pages": pc.raw_pages,
        "full_paper_text": pc.full_paper_text,
        "figures_summary": pc.figures_summary,
        "tables_summary": pc.tables_summary,
        "equations_summary": pc.equations_summary,
        "page_notes": pc.page_notes,
        "page_notes_list": pc.page_notes_list,
        "research_report": an.research_report,
        "cro_reading_notes": an.cro_reading_notes,
        "theoretical_analysis": an.theoretical_analysis,
        "architecture_analysis": ar.architecture_analysis,
        "implementation_plan": ar.implementation_plan,
        "codebase_structure": ar.codebase_structure,
        "file_manifest": ar.file_manifest,
        "code_modules": im.code_modules,
        "review_feedback": im.review_feedback,
        "review_summary": im.review_summary,
        "implementation_notes": im.implementation_notes,
        "execution_results": va.execution_results,
        "measured_validation": va.measured_validation,
        "validation_report": va.validation_report,
        "discrepancies": va.discrepancies,
        "claims": ev.claims,
        "equations": ev.equations,
        "traceability_edges": ev.traceability_edges,
        "consistency_issues": ev.consistency_issues,
        "message_board": co.message_board,
        "cro_directives": co.cro_directives,
        "evaluations": co.evaluations,
        "revision_counts": co.revision_counts,
        "needs_revision": co.needs_revision,
        "readme": out.readme,
        "implementation_paper": out.implementation_paper,
        "output_dir": out.output_dir,
        "pdf_report_path": out.pdf_report_path,
        "final_verdict": out.final_verdict,
    }


def _model_dump(value: Any) -> Any:
    return value.model_dump() if isinstance(value, BaseModel) else value


def from_runtime_state(runtime: RuntimeResearchState) -> ResearchState:
    """Rebuild canonical domain state from an agent's flat result."""

    return {
        "research_input": ResearchInput(
            pdf_path=runtime.get("pdf_path", ""),
            paper_title=runtime.get("paper_title", ""),
            paper_abstract=runtime.get("paper_abstract", ""),
        ),
        "paper_corpus": PaperCorpus(
            raw_pages=runtime.get("raw_pages", []),
            full_paper_text=runtime.get("full_paper_text", ""),
            figures_summary=runtime.get("figures_summary", ""),
            tables_summary=runtime.get("tables_summary", ""),
            equations_summary=runtime.get("equations_summary", ""),
            page_notes=runtime.get("page_notes", ""),
            page_notes_list=runtime.get("page_notes_list", []),
        ),
        "analysis": AnalysisState(
            research_report=runtime.get("research_report", ""),
            cro_reading_notes=runtime.get("cro_reading_notes", ""),
            theoretical_analysis=runtime.get("theoretical_analysis", ""),
        ),
        "architecture": ArchitectureState(
            architecture_analysis=runtime.get("architecture_analysis", ""),
            implementation_plan=runtime.get("implementation_plan", ""),
            codebase_structure=runtime.get("codebase_structure", ""),
            file_manifest=runtime.get("file_manifest", []),
        ),
        "implementation": ImplementationState(
            code_modules=runtime.get("code_modules", {}),
            review_feedback=runtime.get("review_feedback", {}),
            review_summary=runtime.get("review_summary", ""),
            implementation_notes=runtime.get("implementation_notes", ""),
        ),
        "validation": ValidationState(
            execution_results=runtime.get("execution_results", ""),
            measured_validation=runtime.get("measured_validation") or {},
            validation_report=runtime.get("validation_report", ""),
            discrepancies=runtime.get("discrepancies", ""),
        ),
        "evidence": EvidenceState(
            claims=runtime.get("claims", []),
            equations=runtime.get("equations", []),
            traceability_edges=runtime.get("traceability_edges", []),
            consistency_issues=runtime.get("consistency_issues", []),
        ),
        "communication": CommunicationState(
            message_board=runtime.get("message_board", []),
            cro_directives=runtime.get("cro_directives", {}),
            evaluations=runtime.get("evaluations", {}),
            revision_counts=runtime.get("revision_counts", {}),
            needs_revision=runtime.get("needs_revision", []),
        ),
        "output": OutputState(
            readme=runtime.get("readme", ""),
            implementation_paper=runtime.get("implementation_paper", ""),
            output_dir=runtime.get("output_dir", ""),
            pdf_report_path=runtime.get("pdf_report_path", ""),
            final_verdict=runtime.get("final_verdict", ""),
        ),
    }


__all__ = [
    "AgentMessage",
    "AnalysisState",
    "ArchitectureState",
    "CodeModule",
    "CommunicationState",
    "create_initial_state",
    "EvidenceState",
    "from_runtime_state",
    "OutputState",
    "PaperCorpus",
    "PaperSection",
    "ResearchInput",
    "ResearchState",
    "RuntimeResearchState",
    "to_runtime_state",
    "ImplementationState",
    "ValidationState",
]
